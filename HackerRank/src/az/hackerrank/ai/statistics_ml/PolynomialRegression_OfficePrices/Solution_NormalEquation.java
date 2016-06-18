package az.hackerrank.ai.statistics_ml.PolynomialRegression_OfficePrices;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

public class Solution_NormalEquation {

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
		
		double[] theta = new double[newFeaturesNum];
		
		theta = normalEquation(newFeatures,price);
		
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
	
	// condition: a[0].length == b.length
	private static double[][] arrayMultiply(double[][] a, double[][] b){        
        double[][] res = new double[a.length][b[0].length];       
        for(int i=0; i< a.length; i++) {
            for (int j = 0; j < b[0].length; j++){
            	res[i][j] = 0.0;
            	for(int k = 0; k< a[0].length; k++){
            		res[i][j] += a[i][k]*b[k][j];
            	}
            }
        }
        return res;
    }
    
	/*
	 * theta = (X' * X)-1 * X' * y
	 */
    private static double[] normalEquation(double[][] newFeatures, double[] price){
        double[][] newFeatures_transpose = new double[newFeatures[0].length][newFeatures.length]; //x'
        for(int i=0; i< newFeatures[0].length; i++){
            for(int j=0; j<newFeatures.length; j++){
            	newFeatures_transpose[i][j] = newFeatures[j][i];
            }
        }
        
        double[][] xX = arrayMultiply(newFeatures_transpose,newFeatures); // xX = x * x'       
        double[][] Xinverse = solution_2(xX); //(X' * X)-1

        double[][] P = new double[price.length][1];
        for(int i=0; i< price.length; i++){
        	P[i][0] = price[i];
        }
        
        double[][] matrix = arrayMultiply(arrayMultiply(Xinverse, newFeatures_transpose),P);
   
        double[][] matrix_transpose = new double[matrix[0].length][matrix.length]; //matrix'
        for(int i=0; i< matrix[0].length; i++){
            for(int j=0; j<matrix.length; j++){
            	matrix_transpose[i][j] = matrix[j][i];
            }
        }
        double[] pinv = matrix_transpose[0];
        
        return pinv;
    }
    
	public static double[][] solution_2(double[][]in){
		int st_vrs=in.length, st_stolp=in[0].length;
		double[][]out=new double[st_vrs][st_stolp];
		double[][]old=new double[st_vrs][st_stolp*2];
		double[][]news=new double[st_vrs][st_stolp*2];

		
		for (int v=0;v<st_vrs;v++){//ones vector
			for (int s=0;s<st_stolp*2;s++){
				if (s-v==st_vrs) 
					old[v][s]=1;
				if(s<st_stolp)
					old[v][s]=in[v][s];
			}
		}

		for (int v=0;v<st_vrs;v++){
			for (int v1=0;v1<st_vrs;v1++){
				for (int s=0;s<st_stolp*2;s++){
					if (v==v1)
						news[v][s]=old[v][s]/old[v][v];
					else
						news[v1][s]=old[v1][s];
				}
			}
			old=prepisi(news);		
			for (int v1=v+1;v1<st_vrs;v1++){
				for (int s=0;s<st_stolp*2;s++){
					news[v1][s]=old[v1][s]-old[v][s]*old[v1][v];
				}
			}
			old=prepisi(news);
		}

		for (int s=st_stolp-1;s>0;s--){
			for (int v=s-1;v>=0;v--){
				for (int s1=0;s1<st_stolp*2;s1++){
					news[v][s1]=old[v][s1]-old[s][s1]*old[v][s];
				}
			}
			old=prepisi(news);
		}
		for (int v=0;v<st_vrs;v++){//rigt part of matrix is invers
			for (int s=st_stolp;s<st_stolp*2;s++){
				out[v][s-st_stolp]=news[v][s];
			}
		}
		return out;
	}

	public static double[][] prepisi(double[][]in){
		double[][]out=new double[in.length][in[0].length];
		for(int v=0;v<in.length;v++){
			for (int s=0;s<in[0].length;s++){
				out[v][s]=in[v][s];
			}
		}
		return out;
	}
}
