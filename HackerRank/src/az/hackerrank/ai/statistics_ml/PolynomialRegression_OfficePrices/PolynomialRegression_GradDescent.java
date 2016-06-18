package az.hackerrank.ai.statistics_ml.PolynomialRegression_OfficePrices;

import java.io.*;
import java.util.*;

public class PolynomialRegression_GradDescent {

	private static double predict( List<Double> feature , List<Double> weight , int F ) {
        double result = 0 ;
        for( int i = 0 ; i < F ; i ++ ) {
            result += feature.get(i) * weight.get(i) ;
        }
        return result ;
    }

    private static List<Double> updateOnce( List<List<Double>> featureList , List<Double> label ,
                                    List<Double> weight , int N , int F , double a ) {
        List<Double> P = new ArrayList<Double>() ;
        //double J = 0 ;
        for( int i = 0 ; i < N ; i ++ ) {
            double pi = predict( featureList.get( i ) , weight , F ) ;
            P.add( pi ) ;
            //J += ( pi - label.get(i) ) * ( pi - label.get(i) ) ;
            // System.out.println( "pi is " + pi + " " + "label is " + label.get(i) ) ;
        }
        //J /= (2*N) ;
        // System.out.println( "J is " + J ) ;

        List<Double> weight_new = new ArrayList<Double>() ;

        // gradient descent
        for( int i = 0 ; i < F ; i ++ ) {
            double sum = 0 ;
            for( int j = 0 ; j < N ; j ++ ) {
                sum += ( P.get(j) - label.get(j) ) * featureList.get(j).get(i) ;
            }
            Double w = weight.get(i) ;
            weight_new.add(  new Double( w - a * sum / N ) ) ;
        }

        return weight_new ;
    }

    private static List<Double> trainModel( List<List<Double>> featureList , List<Double> label ,
                                    List<Double> weight , int N , int F , double a , int iter ) {
        for( int i = 0 ; i < iter ; i ++ ) {
            // System.out.println( "the " + i + " times iter " ) ;
            weight = updateOnce(featureList, label, weight, N, F, a) ;
        }

        return weight ;
    }

    public static List<Double> Comfeature( List<Double> features ) {
        List<Double> featureNew = new ArrayList<Double>() ;
        int size = ( 1<<(features.size()*2) ) ;
        //System.out.println(size);
        for( int i = 0 ; i < size ; i ++ ) {
            double feature = 1.0 ;
            for( int j = 0 ; j < features.size() ; j ++ ) {
            	int az = (i>>(j<<1))&3 ;
            	//System.out.println("az:" + j + ":" + az);
            	//System.out.println(i + ":feature = " + feature + " * features.get(" + j + ")=" + features.get(j) + " ^" + az );
                feature *= Math.pow( features.get(j) , (i>>(j<<1))&3 ) ;
                //System.out.println(feature);
            }
            //System.out.println(feature);
            featureNew.add( feature ) ;
        }
        return featureNew ;
    }

    public static void main(String[] args) throws FileNotFoundException {

        int N , F , T ;
        //Scanner scanner = new Scanner( System.in ) ;
        File file = new File("sampleInput/PolynomialRegression_OfficePrices.txt");
		
		Scanner scanner = new Scanner(file);

        F = scanner.nextInt() ;
        N = scanner.nextInt() ;

        // get feature && label
        List<List<Double>> featureList = new ArrayList<List<Double>>() ;
        List<Double> label = new ArrayList<Double>() ;
        int Fnum = 0 ;
        for( int i = 0 ; i < N ; i ++ ) {
            List<Double> features = new ArrayList<Double>() ;
            for( int j = 0 ; j < F ; j ++ ) {
                double feature = scanner.nextDouble() ;
                features.add( feature ) ;
            }
            List<Double> featuresCom = Comfeature( features ) ;
            
            for(Double xxx : featuresCom){
            	System.out.println(i + ":" + xxx);
            }
            System.out.println("----------------");
            
            featureList.add( featuresCom ) ;
            Fnum = featuresCom.size() ;
            double labeli = scanner.nextDouble() ;
            label.add( labeli ) ;
        }

        // init weight
        List<Double> weight = new ArrayList<Double>() ;
        for( int i = 0 ; i < Fnum ; i ++ ) {
            weight.add( 0.0 ) ;
        }

        weight = trainModel( featureList , label , weight , N , Fnum , 0.1 , 5000 ) ;

        T = scanner.nextInt() ;

        for( int i = 0 ; i < T ; i ++ ) {
            List<Double> features = new ArrayList<Double>() ;
            for( int j = 0 ; j <  F ; j ++ ) {
                double feature = scanner.nextDouble() ;
                features.add( feature ) ;
            }
            features = Comfeature( features ) ;
            System.out.println( predict( features , weight , Fnum ) ) ;
        }
        
        scanner.close();

    }
}