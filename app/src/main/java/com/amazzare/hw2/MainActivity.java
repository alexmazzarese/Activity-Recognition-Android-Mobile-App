package com.amazzare.hw2;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.hardware.*;

import android.os.Bundle;
import android.os.Environment;
import android.os.FileObserver;
import android.service.autofill.TextValueSanitizer;
import android.util.Log;
import android.widget.EditText;
import android.widget.TextView;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.io.File;
import java.util.Scanner;
import java.util.Random;

import android.view.View;
import android.widget.LinearLayout;
import android.widget.Toast;

import org.w3c.dom.Text;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.core.pmml.jaxbbindings.DecisionTree;


import static weka.core.SerializationHelper.read;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    public static double maximum(double data[]){
        if(data == null || data.length == 0) return 0.0;
        int length = data.length;
        double MAX = data[0];
        for (int i = 1; i < length; i++){
            MAX = data[i]>MAX?data[i]:MAX;
        }
        return MAX;
    }

    public static double minimum(double data[]){
        if(data == null || data.length == 0) return 0.0;
        int length = data.length;
        double MIN = data[0];
        for (int i = 1; i < length; i++){
            MIN = data[i]<MIN?data[i]:MIN;
        }
        return MIN;
    }

    public static double mean(double data[]){
        if(data == null || data.length == 0) return
                0.0;
        int length = data.length;
        double Sum = 0;
        for (int i = 0; i < length; i++)
            Sum = Sum + data[i];
        return Sum / length;
    }

    public static double variance(double data[]){
        if(data == null || data.length == 0) return 0.0;
        int length = data.length;
        double average = 0, s = 0, sum = 0;
        for (int i = 0; i<length; i++)
        {
            sum = sum + data[i];
        }
        average = sum / length;
        for (int i = 0; i<length; i++)
        {
            s = s + Math.pow(data[i] - average, 2);
        }
        s = s / length;
        return s;
    }

    public static double standardDeviation(double data[]){
        if(data == null || data.length == 0) return 0.0;
        double s = variance(data);
        s = Math.sqrt(s);
        return s;
    }

    public static double zeroCrossingRate(double data[]){
        int length = data.length;
        double num = 0;
        for (int i = 0; i < length - 1; i++)
        {
            if (data[i] * data[i + 1]< 0){
                num++;
            }
        }
        return num / (double)length;
    }

    // https://stackoverflow.com/a/6018431/882436
    private static double[] toDoubles(List<Double> doubles ) {
        double[] target = new double[doubles.size()];
        for (int i = 0; i < target.length; i++) {
            target[i] = doubles.get(i);                // java 1.5+ style (outboxing)
        }
        return target;
    }

    // serialize a weka model, overwrites any existing file
    private static void writeFile( Context cx, String fname, Classifier cls ) throws Exception {

        // delete if already exists
        deleteFile(cx, fname);

        File path = cx.getExternalFilesDir(null);
        File file = new File(path, fname);

        weka.core.SerializationHelper.write(file.getAbsolutePath(), cls);
    }

    private static void deleteFile( Context cx, String fname ) {
        File path = cx.getExternalFilesDir(null);
        File file = new File(path, fname);
        file.delete();
    }

    private static void appendToFile( Context cx, String fname, String contents ) {
        File path = cx.getExternalFilesDir(null);
        File file = new File(path, fname);

        try {
            FileOutputStream fs = new FileOutputStream(file, true);
            fs.write( contents.getBytes() );
            fs.close();
        }
        catch ( Exception ex ){
            Log.e("tag", "msg", ex );
        }
    }

    // based on https://stackoverflow.com/a/326448/882436
    private List<String> readFile( Context cx, String fname) throws IOException {

        File path = cx.getExternalFilesDir(null);
        File file = new File(path, fname);

        // StringBuilder fileContents = new StringBuilder((int)file.length());

        List<String> results = new ArrayList<>();

        try (Scanner scanner = new Scanner(file)) {
            while(scanner.hasNextLine()) {
                results.add(scanner.nextLine());
                // fileContents.append(scanner.nextLine() + System.lineSeparator());
            }
            return results;
        }
    }

    public void showMsg( String msg ) {
        Toast.makeText( this.getApplicationContext(), msg, Toast.LENGTH_SHORT).show();
    }

    public void showMsg( Exception ex ) {
        ex.printStackTrace();
        showMsg(ex.getMessage());
    }

    public Instances getWekaInstances( Context cx, String fname ) {

        // based on https://waikato.github.io/weka-wiki/use_weka_in_your_java_code/
        File path = cx.getExternalFilesDir(null);
        File file = new File(path, fname);

        try {

            CSVLoader loader = new CSVLoader();
            loader.setNoHeaderRowPresent(true);
            loader.setSource(file);
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            return data;
        } catch ( Exception ex ) {
            showMsg(ex);
        }

        return null;
    }

    public List<String> getClassLabels() {
        Instances data = getWekaInstances(this.getApplicationContext(), data_fname);
        Enumeration<Object> objectEnumeration = data.classAttribute().enumerateValues();
        ArrayList<String> result = new ArrayList<String>();

        while ( objectEnumeration.hasMoreElements() ) {
            result.add((String)objectEnumeration.nextElement());
        }

        return result;
    }

    public Classifier loadClassifier( String fname ) throws Exception {
        File path = this.getApplicationContext().getExternalFilesDir(null);
        File file = new File(path, fname);
        return (Classifier)weka.core.SerializationHelper.read(file.getAbsolutePath());
    }

    public J48 getJ48Classifier( Instances data ) throws Exception {
        J48 tree = new J48();         // new instance of tree
        tree.setUnpruned(true);
        tree.buildClassifier(data);   // build classifier
        return tree;
    }

    public NaiveBayes getNBClassifier( Instances data ) throws Exception {
        NaiveBayes result = new NaiveBayes();
        result.buildClassifier(data);
        return result;
    }

    public RandomForest getRFClassifier( Instances data ) throws Exception {
        RandomForest result = new RandomForest();
        result.buildClassifier(data);
        return result;
    }

    public double evalClassifier( Classifier cls, Instances data ) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(cls, data, 10, new Random(1));
        //eval.evaluateModel(cls, data);
        return eval.pctCorrect();
    }

    // based on provided Lecture 6 codes
    // usage:
    //      double [] detected=classification(min,max,var,std);
    //      text1.setText("class: " +String.valueOf(detected[0])+" probabilities: "+String.valueOf(detected[1]));
    public double [] classification( Classifier cls, double magnitude, double mean, double var, double min, double max, double std, double zcr) throws Exception {
        int ck=1;
        double [] predicted_class =new double[2];
        predicted_class[0]=0.0;
        predicted_class[1]=0.0;

        // Classifier cls = null;
        // cls = (Classifier) read(getAssets().open("act_rf.model"));

        ArrayList<Attribute> attributes = new ArrayList<>();

        attributes.add(new Attribute("magnitude",0));
        attributes.add(new Attribute("mean",1));
        attributes.add(new Attribute("var",2));
        attributes.add(new Attribute("min",3));
        attributes.add(new Attribute("max",4));
        attributes.add(new Attribute("std",5));
        attributes.add(new Attribute("zcr",6));

        List<String> labels = getClassLabels();

        // attributes.add(new Attribute("label",Arrays.asList("a","b"),7));
        attributes.add(new Attribute("label", labels,7));

        // new instance to classify
        Instance instance = new SparseInstance(7);
        instance.setValue(attributes.get(0), magnitude);
        instance.setValue(attributes.get(1), mean);
        instance.setValue(attributes.get(2), var);
        instance.setValue(attributes.get(3), min);
        instance.setValue(attributes.get(4), max);
        instance.setValue(attributes.get(5), std);
        instance.setValue(attributes.get(6), zcr);

        // Create an empty set
        Instances datasetConfiguration;
        datasetConfiguration = new Instances("data", attributes, 0);

        datasetConfiguration.setClassIndex( attributes.size() - 1 );
        instance.setDataset(datasetConfiguration);

        double[] distribution;
        distribution = cls.distributionForInstance(instance);
        predicted_class[0]=cls.classifyInstance(instance);
        predicted_class[1]=Math.max(distribution[0], distribution[1])*100;

        return predicted_class;
    }

    private SensorManager mSensorMgr;
    private Sensor mAccel;

    long lastDisplaySeconds = 0;

    List<Double> magnitudes = new ArrayList<Double>();

    long startTime = 0;
    int interval=2; // 2 seconds
    boolean is_classifying = false;

    LinearLayout collectLayout = null;
    LinearLayout trainDeployLayout = null;
    EditText txtLabel = null;
    TextView tvRF = null;
    TextView tvNB = null;
    TextView tvDT = null;
    String data_fname = "data.csv";
    String clsRF_fname = "rf.model";
    String clsDT_fname = "dt.model";
    String clsNB_fname = "nb.model";
    String current_label = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        collectLayout = (LinearLayout) findViewById(R.id.collectLayout);
        trainDeployLayout = (LinearLayout)findViewById(R.id.trainDeployLayout);
        txtLabel = (EditText)findViewById(R.id.txtLabel);
        tvRF = (TextView)findViewById(R.id.textRF);
        tvNB = (TextView)findViewById(R.id.textNB);
        tvDT = (TextView)findViewById(R.id.textDT);

    }

    @Override
    protected void onResume() {
        super.onResume();
        //mSensorMgr.registerListener(this, mAccel, SensorManager.SENSOR_DELAY_NORMAL);
    }

    @Override
    protected void onPause() {
        super.onPause();
        //mSensorMgr.unregisterListener(this);
    }

    public void btnCollect_click(View view) {

        trainDeployLayout.setVisibility(View.GONE);
        collectLayout.setVisibility( View.VISIBLE );
        is_classifying = false;
        stopSensor();
    }

    public void btnTrain_click(View view) {

        collectLayout.setVisibility( View.GONE );
        trainDeployLayout.setVisibility(View.VISIBLE);
        is_classifying = false;
        stopSensor();
    }

    public void btnDeploy_click(View view) {
        collectLayout.setVisibility( View.GONE );
        trainDeployLayout.setVisibility( View.VISIBLE );
        is_classifying = true;
        tvNB.setText("");
        tvRF.setText("");
        tvDT.setText("");
        startSensor();
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {

        if (sensorEvent.sensor.getType() != Sensor.TYPE_ACCELEROMETER)
            return;

        // event.values, 0,1,2 = x,y,z
        double magnitude =
                Math.sqrt(sensorEvent.values[0]*sensorEvent.values[0]+sensorEvent.values[1]*sensorEvent.values[1]+sensorEvent.values[2]
                        *sensorEvent.values[2]);
        long millis = System.currentTimeMillis();
        int seconds = (int) (millis / 1000);

        if ( startTime == 0 )
            startTime = seconds;

        if ( (seconds % interval) ==0 && ( lastDisplaySeconds != seconds ) ){

            double[] magArray = toDoubles(magnitudes);
            double min=minimum(magArray);
            double max=maximum(magArray);
            double var=variance(magArray);
            double std=standardDeviation(magArray);
            double mean = mean(magArray);
            double zcr = zeroCrossingRate(magArray);

            if ( !is_classifying ) {

                // validate label exists
                if ( ( current_label == null ) || ( current_label.isEmpty() ) ) {
                    Toast.makeText(this.getApplicationContext(), "Error: label not set.  Discarding data", Toast.LENGTH_SHORT).show();
                    return;
                }

                // write to csv
                //  magnitude,mean,var,min,max,std,zcr,label
                String data = ""
                        // + ( seconds - startTime )
                        + "" + magnitude
                        + ", " + mean
                        + ", " + var
                        + ", " + min
                        + ", " + max
                        + ", " + std
                        + ", " + zcr
                        + ", " + current_label
                        + "\n";

                showMsg( "Writing data for time: " + String.valueOf(seconds - startTime) );

                appendToFile(this.getApplicationContext(), data_fname, data);
            }
            else {  // classifying

                try {

                    List<String> labels = getClassLabels();

                    // load classifiers.  todo:  improve performance by not reloading them on every iteration
                    Classifier dt = loadClassifier(clsDT_fname);
                    double[] detected = classification( dt, magnitude, mean, var, min, max, std, zcr);
                    //tvDT.setText( "class: " + String.valueOf(detected[0])+" probabilities: "+String.valueOf(detected[1]) );
                    tvDT.setText( "class: " + labels.get((int)detected[0]) + " P: "+String.valueOf((int)detected[1]) + "%" );

                    Classifier rf = loadClassifier(clsRF_fname);
                    double[] detected_rf = classification( rf, magnitude, mean, var, min, max, std, zcr);
                    // tvRF.setText( "class: " +String.valueOf(detected_rf[0])+" probabilities: "+String.valueOf(detected_rf[1]) );
                    tvRF.setText( "class: " + labels.get((int)detected_rf[0]) + " P: "+String.valueOf((int)detected_rf[1]) + "%" );

                    Classifier nb = loadClassifier(clsNB_fname);
                    double[] detected_nb = classification( nb, magnitude, mean, var, min, max, std, zcr);
                    //tvNB.setText( "class: " +String.valueOf(detected_nb[0])+" probabilities: "+String.valueOf(detected_nb[1]) );
                    tvNB.setText( "class: " + labels.get((int)detected_nb[0]) + " P: "+String.valueOf((int)detected_nb[1]) + "%" );

                } catch ( Exception ex ) {
                    showMsg(ex);
                }

            }

            magnitudes.clear();
            lastDisplaySeconds = seconds;
        }
        else{
            magnitudes.add(magnitude);
        }

    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
    }

    // start data collection from sensor
    public void btnStartCollection_click(View view) {

        // first verify a label exists
        if ( txtLabel.getText().toString().isEmpty() ) {
            showMsg( "Please enter a label" );
            return;
        }

        current_label = txtLabel.getText().toString();
        showMsg("Starting data collection for label: " + current_label );

        is_classifying = false;
        startTime = 0;

        // initiate sensor listener
        startSensor();
    }

    // end data collection from sensor
    public void btnEndCollection_click(View view) {
        stopSensor();
    }

    private void startSensor() {
        if ( mSensorMgr == null )
            mSensorMgr=(SensorManager)getSystemService(SENSOR_SERVICE);

        mAccel = (Sensor)mSensorMgr.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        if ( mAccel != null ) {
            mSensorMgr.registerListener(this, mAccel, SensorManager.SENSOR_DELAY_NORMAL, SensorManager.SENSOR_DELAY_UI);
            showMsg("Starting sensor listener");
        }
        else
            showMsg( "Error acquiring sensor");
    }

    private void stopSensor() {

        // halt sensor
        // data stored automatically during onSensorChanged
        if ( mSensorMgr != null ) {
            mSensorMgr.unregisterListener(this);
            showMsg("Stopping sensor listener");
        }
    }

    public void btnDelete_click(View view) {
        deleteFile( this.getApplicationContext(), data_fname );
        showMsg("File deleted");
    }

    public void btnShowData_click(View view) {

        try {
            List<String> lines = readFile(this.getApplicationContext(), data_fname);

            if ( lines.size() > 0 )
                showMsg( "First line: " + lines.get(0) );
            else
                showMsg("No data");

        } catch ( IOException ex ) {
            showMsg(ex.toString());
        }

    }

    public void btnDoTraining_click(View view) {

        try {
            Instances data = getWekaInstances(this.getApplicationContext(), data_fname);

            Classifier j48 = getJ48Classifier(data);
            double j48_accuracy = evalClassifier(j48, data);
            tvDT.setText(String.valueOf(j48_accuracy));
            writeFile(this.getApplicationContext(), clsDT_fname, j48);

            Classifier rf = getRFClassifier(data);
            double rf_accuracy = evalClassifier(rf, data);
            tvRF.setText(String.valueOf(rf_accuracy));
            writeFile(this.getApplicationContext(), clsRF_fname, rf);

            Classifier nb = getNBClassifier(data);
            double nb_accuracy = evalClassifier(nb, data);
            tvNB.setText(String.valueOf(nb_accuracy));
            writeFile(this.getApplicationContext(), clsNB_fname, nb);

            showMsg("Training complete");

        } catch ( Exception ex ) {
            showMsg(ex);
        }

    }
}
