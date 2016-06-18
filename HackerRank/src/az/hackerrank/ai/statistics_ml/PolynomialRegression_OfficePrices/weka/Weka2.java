package az.hackerrank.ai.statistics_ml.PolynomialRegression_OfficePrices.weka;

import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;
import weka.classifiers.*;
import weka.classifiers.functions.LinearRegression;
import weka.core.*;


public class Weka2 {

   public static List<List<Integer>> possExps = new ArrayList<List<Integer>>();
    
    public static int maxD;
    
    public static int F;

    public static List<Integer> getCopy(List<Integer> list){
    	List<Integer> newList = new ArrayList<Integer>();
    	for(int i : list)
    		newList.add(i);
    	return newList;
    }
    
    public static boolean getPossibleExponents(List<Integer> listA, int d){
       
        List<Integer> list = getCopy(listA);        
        
        int preCount = list.size();
        if(preCount >= F){
            possExps.add(list);
            return true;
        }          
        
        int preSum = 0;
        for(int i : list)
            preSum += i;
        
        if(preSum + d <= maxD){
        	list.add(d);
            for(int deg = 0; deg <= maxD; deg++){
                boolean closed = getPossibleExponents(list,deg);
                if(closed)
                	break;
            }
        }
        return false;
    }

    public static void main(String[] args) throws Exception{         
    	
    	Scanner in = new Scanner(System.in);
        F = in.nextInt();
        
        //Possible feature combinations under degree 4
        maxD = 3;
        List<Integer> startList = new ArrayList<Integer>();
        for(int deg = 0; deg <= maxD; deg++){
            getPossibleExponents(startList,deg);
        }
        
        //Create all (combined) attributes + Class
        int numFeats = possExps.size();
        FastVector atts = new FastVector(numFeats+1);
        for(int f = 0; f < numFeats; f++){
            atts.addElement(new Attribute("F" + (f+1)));
        }
        atts.addElement(new Attribute("Price"));
           
        //TRAIN
        int N = in.nextInt();
        Instances train = new Instances("Train", atts, N);
        train.setClassIndex(train.numAttributes() - 1); 
        for(int i = 0; i < N; i++){
            Instance instTrain = new DenseInstance(numFeats+1);
            instTrain.setDataset(train);
            
            double[] vals = new double[F];
            for(int j = 0; j < F; j++){
                vals[j] = in.nextDouble();
            }            
            
            int pos = 0;
            for(List<Integer> exps : possExps){
                double value = 1;
                for(int j = 0; j < F; j++)
                    value *= Math.pow(vals[j],(double)exps.get(j));
                instTrain.setValue((Attribute)atts.elementAt(pos), value);
                pos++;
            }
            instTrain.setClassValue(in.nextDouble());
            train.add(instTrain);
        }
        
        //TEST
        int T = in.nextInt();
        Instances test = new Instances("Test", atts, T);
        test.setClassIndex(test.numAttributes() - 1); 
        for(int t = 0; t < T; t++){
            Instance instTest = new DenseInstance(numFeats+1); 
            instTest.setDataset(test);
            double[] vals = new double[F];
            for(int j = 0; j < F; j++)
                vals[j] = in.nextDouble();
            
            int pos = 0;
            for(List<Integer> exps : possExps){
                double value = 1;
                for(int j = 0; j < F; j++)
                    value *= Math.pow(vals[j],(double)exps.get(j));
                instTest.setValue((Attribute)atts.elementAt(pos++), value);
            }
            instTest.setClassMissing();
            test.add(instTest);
        }
        in.close();            
        
        LinearRegression lr = new LinearRegression();
 	    lr.buildClassifier(train);
        
        @SuppressWarnings("unchecked")
		Enumeration<Instance> insts = test.enumerateInstances();
        while(insts.hasMoreElements()){
        	System.out.println(lr.classifyInstance(insts.nextElement()));
        }
    }
}
/* *--Tail Starts Here-- */