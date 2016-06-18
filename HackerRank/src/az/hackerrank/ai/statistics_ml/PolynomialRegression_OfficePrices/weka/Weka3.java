package az.hackerrank.ai.statistics_ml.PolynomialRegression_OfficePrices.weka;

import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

import weka.classifiers.functions.SMOreg;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class Weka3 {

         public static void main(String[] args) throws IOException, Exception {
       // Scanner in = new Scanner(new File("input.txt"));
        Scanner in = new Scanner(System.in);
        int num_attr = in.nextInt();
        int cases = in.nextInt();

        FastVector attrs = new FastVector();

        for (int i = 0; i < num_attr + 1; i++) {
            attrs.addElement(new Attribute("attr" + i));
        }

        Instances dataset = new Instances("my_dataset", attrs, 0);

        for (int i = 0; i < cases; i++) {
            Instance inst = new DenseInstance(num_attr + 1);
            for (int j = 0; j < num_attr + 1; j++) {
                double hodnota = Double.parseDouble(in.next());
                inst.setValue(j, hodnota);
            }
            dataset.add(inst);
        }

      
        
        int num_urcit = in.nextInt();
        for (int i = 0; i < num_urcit; i++) {
           Instance inst = new DenseInstance(num_attr + 1);
            for (int j = 0; j < num_attr; j++) {
                double hodnota = Double.parseDouble(in.next());
                inst.setValue(j, hodnota);
            }
             dataset.add(inst);
        }
        
         //System.out.println(dataset);
        
        dataset.setClassIndex(dataset.numAttributes()-1);
        SMOreg model = new SMOreg();
        
        String neco[] = model.getOptions();
        neco[7] = "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 3.0 -L";
        //System.out.println(Arrays.toString(neco));
        model.setOptions(neco);
                
        model.buildClassifier(dataset);
        //System.out.println(model);
        
        for (int i = 0; i < num_urcit; i++) {
            double price = model.classifyInstance(dataset.instance(cases+i));
            System.out.println(price);
            
        }
  
    }
}