package az.hackerrank.ai.statistics_ml.PolynomialRegression_OfficePrices;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

public class Solution_GradDescent {

	public static void main(String[] args) throws FileNotFoundException {
		//File file = new File("sampleInput/PolynomialRegression_OfficePrices.txt");
		//Scanner scan = new Scanner(file);
		Scanner scan = new Scanner(System.in);
		
		int featuresNum = scan.nextInt(); // Number of features
		int trainningRowsNum = scan.nextInt();   // Number of data points

		double[] price = new double[trainningRowsNum]; //Y
		
		double[][] features = new double[trainningRowsNum][featuresNum];
		
		for (int i = 0; i < trainningRowsNum; i++) {
			for (int j = 0; j < featuresNum; j++) {
				features[i][j] = scan.nextDouble();	
			}
			//price[i] = scan.nextDouble() / 1500;
			price[i] = scan.nextDouble();
		}
		
		/*
		 * Think of each such term (x1^a)(x2^b) as a new variable z,and (a+b) <= 4
		 */
		int newFeaturesNum = 0;
		double[][] newFeatures = new double[trainningRowsNum][];
		

		for (int r = 0; r < trainningRowsNum; r++) {	
			int n = 4; // n = dimension
			double[][] f = new double[featuresNum][n];
			for (int i = 0; i < featuresNum; i++){
				for (int j = 0; j < n; j++){
					f[i][j] = Math.pow( features[r][i] , j ); // x^0,x^1,x^2,x^3
				}
			}

			ArrayList<double[]> newFeaturesByRecursion = getNewFeaturesByRecursion(f);
			newFeaturesNum = newFeaturesByRecursion.size();

			double[] f_l = new double[newFeaturesNum];
			for (int i = 0; i < newFeaturesNum; i++){
				f_l[i] = 1.0;
			}
			
			for (int i = 0; i < newFeaturesByRecursion.size(); i++){
				f_l[i] = newFeaturesByRecursion.get(i)[0];
			}	
			newFeatures[r] = f_l;
		}
        
		///* 
		//Grad Descent
		
		double[] theta = new double[newFeaturesNum];
		for (int i = 0; i < newFeaturesNum; i++){
			//theta[i] = Math.random();
			theta[i] = 0.0;
		}

		double cost = 0.0;
		double difference = 0.0;
		boolean gd = false; // Gradient Descent
		double alpha = 0.1; //adjust it

		double[] d = new double[theta.length];
		int iteration = 0;
		do{		
			double oldCost = cost;
			for (int k = 0; k < theta.length; k++) {
				double sumError = 0.0;
				
				for (int i = 0; i < trainningRowsNum; i++) {				
					double y = price[i]; //Expected
					double h = 0.0; //Computed
					
					double[] f = new double[theta.length];
					for (int j = 0; j < theta.length; j++) {
						f[j] = newFeatures[i][j];	
					}
		
					for (int j = 0; j < theta.length; j++) {
						h += f[j] * theta[j];
					}
		
					difference = h - y;
					sumError = sumError + difference * f[k] ;	
					
					//cost += Math.pow(difference, 2);
					cost += difference;
				}
				d[k] = sumError/trainningRowsNum;
			}
			cost = cost/trainningRowsNum;
			
			if (iteration < 100000){
				iteration++;
				gd = true;
				for (int j = 0; j < theta.length; j++) {
					theta[j] = theta[j] - alpha * d[j] ;
					System.out.print(theta[j] + "	");
				}
				System.out.println("");
			}else{
				gd = false;
			}	
			
		}while(gd);
		//end Grad Descent 

		int testNumber = scan.nextInt();
		features = new double[testNumber][featuresNum];
		newFeatures = new double[testNumber][newFeaturesNum+1];
        
        for(int t = 0; t < testNumber; t++){
        	for (int j = 0; j < featuresNum; j++) {
				features[t][j] = scan.nextDouble();	
			}
        	
        	double r = 0.0;
        	int n = 4; // n = dimension
			double[][] f = new double[featuresNum][n];
			for (int i = 0; i < featuresNum; i++){
				for (int j = 0; j < n; j++){
					f[i][j] = Math.pow( features[t][i] , j ); // x^0,x^1,x^2,x^3
				}
			}

			ArrayList<double[]> newFeaturesByRecursion = getNewFeaturesByRecursion(f);
			double[] f_l = new double[newFeaturesNum];
			for (int i = 0; i < newFeaturesNum; i++){
				f_l[i] = 1.0;
			}
			
			for (int i = 0; i < newFeaturesByRecursion.size(); i++){
				f_l[i] = newFeaturesByRecursion.get(i)[0];
			} 	
			newFeatures[t] = f_l;

            for (int k = 0; k < newFeaturesNum; k++) {
            	r = r+newFeatures[t][k]*theta[k];
            }
            
            //System.out.println(r*1500);
            System.out.println(r);
        }

        scan.close();

	}
	
	public static ArrayList<double[]> getNewFeaturesByRecursion(double[][] features) {
	
		ArrayList<double[]> newFeatures = new ArrayList<double[]>();
		int f = features.length;
		int d = features[0].length; // == dimension
		
		if (1 == f ){
			for (int i = 0; i < d; i++){
				double[] element = new double[2];
				element[0] = features[0][i];
				element[1] = i;
				newFeatures.add(element);
			}		
		}else{
			double[][] features_new = new double[f-1][d];
			for (int i = 1; i < f; i++){
				for (int j = 0; j < d; j++){
					features_new[i-1][j] = features[i][j];
				}
			}
			ArrayList<double[]> newFeature = getNewFeaturesByRecursion(features_new);
			
			double[][] features_0 = new double[1][d];
			for (int j = 0; j < d; j++){
				features_0[0][j] = features[0][j];;
			}
			ArrayList<double[]> newFeature_0 = getNewFeaturesByRecursion(features_0);
			
			for (int i = 0; i < newFeature_0.size(); i++){
				for (int j = 0; j < newFeature.size(); j++){
					double dimension = newFeature_0.get(i)[1] + newFeature.get(j)[1];			
					if (dimension < 4){
						double[] element = new double[2];
						element[0] = newFeature.get(j)[0]*newFeature_0.get(i)[0];
						element[1] = dimension;
						newFeatures.add(element);
					}
				}
			}
		}
		
		return newFeatures;
	}
	
}
