package az.hackerrank.ai.statistics_ml.PolynomialRegression_OfficePrices.weka;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.Scanner;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.core.*;


public class Weka1 {
	private static Hashtable<String,Integer> answer = new Hashtable<String,Integer> ();
	private static void fun(String text,int level,int endlevel,String store){
		if (level==endlevel){
			answer.put(store,1);
		}
		else{
			for(int i=0;i<text.length();i++){
			 String tempStore=store+text.substring(i, i+1);
			fun(text,level+1,endlevel,tempStore); 
			}
		}
		
	}
public static void main(String args[]) throws Exception{

	 
	 Scanner in = new Scanner(System.in);
 	
 	int featureNumber=in.nextInt();
 	int trainNumber=in.nextInt();
 	fun("0123",1,featureNumber+1,"");
 
 	ArrayList<String> toDelete = new ArrayList<String>();
 	for(String ele:answer.keySet()){
 		int count=0;
 		int delete=0;
 		for(int i=0;i<ele.length();i++){
 			count+=Integer.parseInt(ele.substring(i, i+1));
 			if (count>3){delete=1;break;}
 		}
 		if(count==0 || delete==1){
 			toDelete.add(ele);
 		}
 		
 	}
 	for(String ele:toDelete){
 		answer.remove(ele);
 	}
 	//System.out.println(answer);
 	 FastVector fvWekaAttributes = new FastVector(answer.size()+1);
 	for(int i=0;i<answer.size()+1;i++){
 		Attribute Attribute1 = new Attribute(String.valueOf(i));
 		fvWekaAttributes.addElement(Attribute1);
 	}
 	
 	Instances isTrainingSet = new Instances("train", fvWekaAttributes, trainNumber);     
 	isTrainingSet.setClassIndex(answer.size());

	
 	
 	for(int i=0;i<trainNumber;i++){
 		Instance iExample = new DenseInstance(answer.size()+1); 
 		
 		double line[]=new double[featureNumber];
 		for(int j=0;j<featureNumber;j++){
 			line[j]=in.nextDouble();
 		}
 		int pos=0;
 		for(String text:answer.keySet()){
 			
 			double addon=1;
 		
 		
 			for(int z=0;z<featureNumber;z++){ 							
 				addon*=Math.pow(line[z],Double.parseDouble(text.substring(z, z+1)));
 			}
 			
 			iExample.setValue((Attribute)fvWekaAttributes.elementAt(pos++),addon);     
 		
 		} // add the instance
 		iExample.setValue((Attribute)fvWekaAttributes.elementAt(pos++),in.nextDouble());     
 		isTrainingSet.add(iExample);
 	}
 	//LibLINEAR nb = new LibLINEAR();
 	Classifier nb = (Classifier)new LinearRegression();
 	nb.buildClassifier(isTrainingSet);
 	//System.out.println(isTrainingSet.toString());
 	
 	
 	int testNumber=in.nextInt();
 	Instances isTestSet = new Instances("test",fvWekaAttributes,testNumber);           
	 // Set class index

 



 	for(int i=0;i<testNumber;i++){
 		Instance iExample = new DenseInstance(answer.size()+1); 
 		
 		double line[]=new double[featureNumber];
 		for(int j=0;j<featureNumber;j++){
 			line[j]=in.nextDouble();
 		}
 		int pos=0;
 		for(String text:answer.keySet()){
 			
 			double addon=1;
 		
 		
 			for(int z=0;z<featureNumber;z++){ 							
 				addon*=Math.pow(line[z],Double.parseDouble(text.substring(z, z+1)));
 			}
 			
 			iExample.setValue((Attribute)fvWekaAttributes.elementAt(pos++),addon);     
 		
 		} // add the instance
 		//iExample.setValue((Attribute)fvWekaAttributes.elementAt(pos++),in.nextDouble());     
 		isTestSet.add(iExample);
 	}
 
 	/*
	Evaluation eTest = new Evaluation(isTrainingSet);
	 eTest.evaluateModel(nb, isTrainingSet);
	 String strSummary = eTest.toSummaryString();
	 System.out.println(strSummary);
	 */

	

	for(int i=0;i<isTestSet.numInstances();i++){
		
		 double predict = nb.classifyInstance(isTestSet.instance(i));
		System.out.println(predict);
	}
	
	
	
}
}
