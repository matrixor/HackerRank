package az.hackerrank.ai.statistics_ml.PolynomialRegression_OfficePrices;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class PolynomialRegression_NormalEquation {
    public static final int ORDER = 3;
    private static double[][] getSubMatrix(double[][] matrix, int a, int b){
        int size = matrix.length, row, col;
        double[][] subMatrix = new double[size-1][size-1];
        for(int i=0; i<size-1; i++){
            for(int j=0; j<size-1; j++){
                row = (i<a) ? i:i+1;
                col = (j<b) ? j:j+1;
                subMatrix[i][j] = matrix[row][col];
            }
        }
        return subMatrix;
    }
    private static double getDeterminant(double[][] matrix){
        int size = matrix.length;
        double sum = 0;
        int mult;
        if(size == 2){
            return matrix[0][0]*matrix[1][1]-(matrix[1][0]*matrix[0][1]);
        }
        else {
            for(int i = 0; i < size; i++){
                mult = (i%2 == 0) ? 1:-1;
                sum += matrix[0][i] * getDeterminant(getSubMatrix(matrix,0,i)) * mult;
            }
        }
        return sum;
    }
    private static double[][] getMatrixOfMinors(double[][] matrix){
        int size = matrix.length;
        double[][] minors = new double[size][size];
        for(int i=0; i< size; i++){
            for(int j=0; j<size; j++){
                minors[i][j] = getDeterminant(getSubMatrix(matrix,i,j));
            }
        }
        return minors;
    }
    private static double[][] getCofactors(double[][] matrix){
        int size = matrix.length;
        int mult;
        double[][] cofactors = new double[size][size];
        for(int i=0; i< size; i++){
            for(int j=0; j<size; j++){
                mult = ((i+j)%2 == 0) ? 1:-1;
                cofactors[i][j] = matrix[i][j] * mult;
            }
        }
        return cofactors;
    }
    private static double[][] getAdjugate(double[][] matrix){
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] adjugate = new double[cols][rows];
        for(int i=0; i< cols; i++){
            for(int j=0; j<rows; j++){
                adjugate[i][j] = matrix[j][i];
            }
        }
        return adjugate;
    }
    private static double[][] scalarMultiply(double[][] matrix, double k){
        int size = matrix.length;
        double[][] res = new double[size][size];
        for(int i=0; i< size; i++) {
            for (int j = 0; j < size; j++){
                res[i][j] = matrix[i][j]*k;
            }
        }
        return res;

    }
    private static double vectorMultiply(double[] v1, double[] v2){
        int len1 = v1.length;
        double sum = 0;
        for(int i = 0; i< len1; i++){
            sum += v1[i]*v2[i];
        }
        return sum;
    }
    private static double[][] matrixMultiply(double[][] mat1, double[][] mat2){
        int row1 = mat1.length;
        int col1 = mat1[0].length;
        int row2 = mat2.length;
        int col2 = mat2[0].length;
        if(col1 != row2){
            System.out.println("Cannot multiply the 2 matrix, no. of cols of 1 must equal no. of rows of 2.");
            System.exit(-1);
        }
        double[][] tmat2 = getAdjugate(mat2);
        double[][] res = new double[row1][col2];
        for(int i=0; i< row1; i++) {
            for (int j = 0; j < col2; j++){
                res[i][j] = vectorMultiply(mat1[i],tmat2[j]);
            }
        }
        return res;
    }
    private static double[][] getInverse(double[][] matrix){
        double[][] mat = getAdjugate(getCofactors(getMatrixOfMinors(matrix)));
        double determinant = getDeterminant(matrix);
        mat = scalarMultiply(mat,1/determinant);
        return mat;
    }
    private static int getVectorSum(int[] vector){
        int len = vector.length;
        int sum = 0;
        for(int i = 0; i<len; i++){
            sum+= vector[i];
        }
        return sum;
    }
    private static ArrayList<int[]> getExponentVectors(int n, int order){
        ArrayList<int[]> res = new ArrayList<>();
        if(n == 1){
            for(int i = 0; i <= order; i++){
                int[] vect = new int[1];
                vect[0] = i;
                res.add(vect);
            }
            return res;
        }
        ArrayList<int[]> vectors = getExponentVectors(n-1,order);
        int len = vectors.size();
        int vecLen = vectors.get(0).length;
        for(int i = 0; i <= order; i++){
            for(int j = 0; j < len; j++){
                int[] vector = new int[vecLen+1];
                vector[0] = i;
                for(int k = 0; k < vecLen; k++){
                    vector[k+1] = vectors.get(j)[k];
                }
                if (getVectorSum(vector)<=order){
                    res.add(vector);
                }
            }
        }
        return res;
    }
    private static double[][] getDataMatrix(ArrayList<int[]> expVectors, double[][] data){
        int cols = expVectors.size();
        int rows = data.length;
        int F = data[0].length;
        int[] expVector;
        double[][] matrix = new double[rows][cols];
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                expVector = expVectors.get(j);
                matrix[i][j] = 1;
                for(int f = 0; f < F; f++){
                    matrix[i][j] *= Math.pow(data[i][f],expVector[f]);
                }
            }
        }
        return matrix;
    }
    private static double[] solveEqFromLU(ArrayList<double[][]> LU, double[] y){
        int len = LU.get(0).length;
        double[] d = new double[len];
        double[] b = new double[len];
        double[][] L = LU.get(0);
        double[][] U = LU.get(1);
        d[0] = y[0]/L[0][0];
        for(int i=1; i<len;i++){
            d[i] = y[i];
            for(int j = 0; j<i; j++){
                d[i] -= L[i][j]*d[j];
            }
        }
        b[len-1] = d[len-1];
        for(int i=len-2;i>=0;i--){
            b[i] = d[i];
            for(int j = i+1; j<len; j++){
                b[i] -= U[i][j]*b[j];
            }
            b[i] /= U[i][i];
        }
        return b;
    }
    private static double[][] getInverseFromLU(ArrayList<double[][]> LU){
        int len = LU.get(0).length;
        double[][] identity = new double[len][len];
        double[][] d = new double[len][len];
        double[][] b = new double[len][len];
        double[][] L = LU.get(0);
        double[][] U = LU.get(1);
        // create identity matrix
        for(int i=0; i<len; i++){
            identity[i][i] = 1;
        }
        for(int m=0; m<len; m++){
            d[m][0] = identity[m][0]/L[0][0];
            for(int i=1; i<len;i++){
                d[m][i] = identity[m][i];
                for(int j = 0; j<i; j++){
                    d[m][i] -= L[i][j]*d[m][j];
                }
            }
            b[m][len-1] = d[m][len-1];
            for(int i=len-2;i>=0;i--){
                b[m][i] = d[m][i];
                for(int j = i+1; j<len; j++){
                    b[m][i] -= U[i][j]*b[m][j];
                }
                b[m][i] /= U[i][i];
            }
        }
        return getAdjugate(b);
        //return b;
    }
    private static double[] getCoefficients(double[][] X, double[] y){
        double[][] Y = new double[1][y.length];
        Y[0] = y;
        Y = getAdjugate(Y);
        double[][] X_abjoint = getAdjugate(X);
        ArrayList<double[][]> LU = LUDecompositionDoolittle(matrixMultiply(X_abjoint,X));
        double[][] Xinverse2 = getInverse(matrixMultiply(X_abjoint, X));
        //double[][] Xinverse2 = getInverseFromLU(LU);
        
        for(int i=0; i< Xinverse2.length; i++){
            for(int j=0; j<Xinverse2[0].length; j++){
            	System.out.print(Xinverse2[i][j] + "	");
            }
            System.out.println();
        }
        System.out.println("-.-.-111.-.-.-");
   
        
        double[][] matrix = matrixMultiply(matrixMultiply(Xinverse2, X_abjoint),Y);
        
        for(int i=0; i< matrix.length; i++){
            for(int j=0; j<matrix[0].length; j++){
            	System.out.print(matrix[i][j] + "	");
            }
            System.out.println();
        }
        System.out.println("-.-.222-.-.-.-");
        
        for(int i=0; i< getAdjugate(matrix)[0].length; i++){
            System.out.println(getAdjugate(matrix)[0][i] + "	");
        }
        System.out.println("-.-.333-.-.-.-");
        
        return getAdjugate(matrix)[0];
    }
    private static double[][] copyOf(double[][] b){
        int L = b.length;
        if(b == null){
            return null;
        }
        double[][] copy = new double[L][L];
        for(int i = 0; i < L; i++){
            for(int j = 0; j < L; j++){
                copy[i][j] = b[i][j];
            }
        }
        return copy;
    }
    private static ArrayList<double[][]> LUDecompositionDoolittle(double[][] X){
        ArrayList<double[][]> LU = new ArrayList<>();
        double[][] a = copyOf(X);
        int N = a.length;
        int i,j,k;
        for(i=0; i<N-1;i++){
            for(k=i+1; k<N; k++){
                a[k][i] = a[k][i]/a[i][i];
                for(j=i+1;j<N;j++){
                    a[k][j] = a[k][j] - a[k][i]*a[i][j];
                }
            }
        }
        double[][] L = new double[N][N];
        double[][] U = new double[N][N];
        for(i = 0; i<N; i++){
            for(j = 0; j<N; j++){
                if(i<=j){
                    U[i][j] = a[i][j];
                }
                if(i==j){
                    L[i][j] = 1;
                }
                if(i>j){
                    L[i][j] = a[i][j];
                }
            }
        }
        LU.add(L);
        LU.add(U);
        return LU;
    }
    private static ArrayList<double[][]> LUDecomposition(double[][] X){
        ArrayList<double[][]> LU = new ArrayList<>();
        int len = X.length;
        int i,j,k;
        double[][] L = new double[len][len];
        double[][] U = new double[len][len];
        for(i = 0; i<len; i++){
            L[i][0] = X[i][0];
            U[0][i] = X[0][i]/L[0][0];
            U[i][i] = 1;
        }
        for(j=1; j<len-1; j++){
            for(i=j; i<len;i++){
                L[i][j] = X[i][j];
                for(k=0; k<j-1; k++){
                    L[i][j] -= L[i][k]*U[k][j];
                }
            }
            for(k=j+1; k<len; k++){
                U[j][k] = X[j][k];
                for(i=0; i<j-1; i++){
                    U[j][k] -= L[j][i]*U[i][k];
                }
                U[j][k] /= L[j][j];
            }
        }
        L[len-1][len-1] = X[len-1][len-1];
        for(k=1; k<len-1; k++){
            L[len-1][len-1] -= L[len-1][k]*U[k][len-1];
        }
        LU.add(L);
        LU.add(U);
        return LU;
    }
    public static void main(String[] args) throws FileNotFoundException {

        //Scanner in = new Scanner(System.in);
    	File file = new File("sampleInput/PolynomialRegression_OfficePrices.txt");
		Scanner in = new Scanner(file);
		
        /*int N = in.nextInt();
        double[][] data = new double[N][N];
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                data[i][j] = in.nextDouble();
            }
        }
        N = in.nextInt();
        double[] y = new double[N];
        for(int i = 0; i < N; i++){
            y[i] = in.nextDouble();
        }
        ArrayList<double[][]> LUtest = LUDecompositionDoolittle(data);
        double[] b = solveEqFromLU(LUtest,y);
        double[][] inverse = getInverseFromLU(LUtest);*/
        int F = in.nextInt();
        int N = in.nextInt();
        double[][] data = new double[N][F];
        double[] y = new double[N];
        for(int i = 0; i < N; i++){
            for(int j = 0; j < F; j++){
                data[i][j] = in.nextDouble();
            }
            y[i] = in.nextDouble();
        }

        ArrayList<int[]> expVectors = getExponentVectors(F,ORDER);
        double[][] dataMatrix = getDataMatrix(expVectors,data);
        double [] coefficients = getCoefficients(dataMatrix,y);

        int T = in.nextInt();
        double[][] test = new double[T][F];
        for(int i = 0; i < T; i++){
            for(int j = 0; j < F; j++){
                test[i][j] = in.nextDouble();
            }
        }
        double prediction, subProduct;
        int[] vector;
        int len = expVectors.size();
        int vectLen = expVectors.get(0).length;
        for(int i = 0; i < T; i++){
            prediction = 0;
            for(int j = 0; j < len; j++){
                vector = expVectors.get(j);
                subProduct = 1;
                for(int k = 0; k < vectLen; k++){
                    subProduct *= Math.pow(test[i][k],vector[k]);
                }
                subProduct*= coefficients[j];
                prediction += subProduct;
            }
            System.out.println(prediction);
        }

    }
}
