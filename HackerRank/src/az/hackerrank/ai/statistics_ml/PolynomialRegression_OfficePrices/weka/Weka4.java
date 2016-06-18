package az.hackerrank.ai.statistics_ml.PolynomialRegression_OfficePrices.weka;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;

import java.util.ArrayList;
import java.util.Scanner;

import java.io.PrintWriter;
import java.io.FileNotFoundException;

public class Weka4 {
    public static void main (String[] args) throws FileNotFoundException, Exception {
        Scanner in;
        PrintWriter fileOut;
        DataSource datasource;
        Instances dataset;
        MultilayerPerceptron classifier;
        int f, n, t;
        
        in = new Scanner(System.in);
        f = in.nextInt();
        n = in.nextInt();
        
        fileOut = new PrintWriter("training.csv");
        for (int i = 0; i < f; i++) {
            fileOut.print(i + ",");
        }
        fileOut.println("price");
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < f; j++) {
                double feature = in.nextDouble();
                fileOut.print(feature + ",");
                //fileOut.print(Math.pow(feature, 2) + ",");
                //fileOut.print(Math.pow(feature, 3) + ",");
            }
            fileOut.println(in.nextDouble());
        }
        fileOut.close();
        
        datasource = new DataSource("training.csv");
        dataset = datasource.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);
        
        classifier = new MultilayerPerceptron();
        classifier.buildClassifier(dataset);
        
        t = in.nextInt();
        
        fileOut = new PrintWriter("test.csv");
        for (int i = 0; i < f; i++) {
            fileOut.print(i + ",");
        }
        fileOut.println("price");
        
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < f; j++) {
                double feature = in.nextDouble();
                fileOut.print(feature + ",");
                //fileOut.print(Math.pow(feature, 2) + ",");
                //fileOut.print(Math.pow(feature, 3) + ",");
            }
            
            fileOut.println(0.0);
        }
        fileOut.close();
        
        datasource = new DataSource("test.csv");
        dataset = datasource.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);
        
        for (int i = 0; i < t; i++) {
            System.out.println(classifier.classifyInstance(dataset.instance(i)) - 5);
        }
    }
}

