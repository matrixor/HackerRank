package az.hackerrank.ai.statistics_ml.PolynomialRegression_OfficePrices;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Solution {

	public static void main(String[] args) throws FileNotFoundException {
		File file = new File("sampleInput/PolynomialRegression_OfficePrices.txt");
		
		Scanner scan = new Scanner(file);
		//Scanner scan = new Scanner(System.in);
		int featuresNumber = scan.nextInt();
		int rowsNumber = scan.nextInt();
		double alpha = 0.02; //adjust it
		
		double[] theta = new double[featuresNumber + 1];
		for (int i = 0; i < theta.length; i++) {
			theta[i] = 1 ; // initial theta
		}
		double difference = 0.0;
		
		double cost = 0.0;
		
		boolean gd = false; // Gradient Descent
		
		double[][] feature = new double[rowsNumber][featuresNumber + 1];
		double[] p = new double[rowsNumber];
		for (int i = 0; i < rowsNumber; i++) {
			for (int j = 0; j < theta.length-1; j++) {
				feature[i][j] = scan.nextDouble();	
			}
			feature[i][theta.length-1] = 1;
			p[i] = scan.nextDouble() / 1500; //normalization
		}
			
		do{		
			for (int i = 0; i < rowsNumber; i++) {				
				double h = 0.0;
				double[] f = new double[theta.length];

				for (int j = 0; j < theta.length; j++) {
					f[j] = feature[i][j];	
				}
				
				double price = p[i];

				for (int j = 0; j < theta.length; j++) {
					theta[j] = theta[j] - alpha * f[j] * difference / rowsNumber ;
					h += f[j] * theta[j];
				}
					
				difference = Math.abs(h - price);
				cost += Math.pow(difference, 2);

			}

			cost = cost/2/rowsNumber;
			
			if(cost > 0.003){ //adjust 0.1
				gd = true;
			}else{
				gd = false;
			}
			
		}while(gd);
		
		int testNumber = scan.nextInt();
        
        for(int i = 0; i < testNumber; i++){
        	double r = 0.0;
        	
            for (int k = 0; k < theta.length-1; k++) {
            	double featursTest = scan.nextDouble();
            	r = r+featursTest*theta[k];
            } 
            
            r = r + theta[theta.length-1];
            
            System.out.println(r*1500);
        }
        scan.close();

	}

}
