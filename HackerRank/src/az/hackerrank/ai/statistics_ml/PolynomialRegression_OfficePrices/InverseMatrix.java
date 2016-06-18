package az.hackerrank.ai.statistics_ml.PolynomialRegression_OfficePrices;

import java.util.ArrayList;

public class InverseMatrix {
    
    ///////////////////Solution 1///////////////////////////////////////////////////////
	
    public static double[][] solution_1(double a[][]){
        int n = a.length;
        double x[][] = new double[n][n];
        double b[][] = new double[n][n];
        int index[] = new int[n];
        for (int i=0; i<n; ++i) 
            b[i][i] = 1;
        // Transform the matrix into an upper triangle
        gaussian(a, index);
        // Update the matrix b[i][j] with the ratios stored
        for (int i=0; i<n-1; ++i)
            for (int j=i+1; j<n; ++j)
                for (int k=0; k<n; ++k)
                    b[index[j]][k]
                    	    -= a[index[j]][i]*b[index[i]][k]; 
        // Perform backward substitutions
        for (int i=0; i<n; ++i) {
            x[n-1][i] = b[index[n-1]][i]/a[index[n-1]][n-1];
            for (int j=n-2; j>=0; --j) {
                x[j][i] = b[index[j]][i];
                for (int k=j+1; k<n; ++k){
                    x[j][i] -= a[index[j]][k]*x[k][i];
                }
                x[j][i] /= a[index[j]][j];
            }
        }
        return x;
    }
 
    // Method to carry out the partial-pivoting Gaussian
    // elimination.  Here index[] stores pivoting order. 
    public static void gaussian(double a[][], int index[]){
        int n = index.length;
        double c[] = new double[n];
        // Initialize the index
        for (int i=0; i<n; ++i) 
            index[i] = i; 
        // Find the rescaling factors, one from each row
        for (int i=0; i<n; ++i){
            double c1 = 0;
            for (int j=0; j<n; ++j){
                double c0 = Math.abs(a[i][j]);
                if (c0 > c1) c1 = c0;
            }
            c[i] = c1;
        }
        // Search the pivoting element from each column
        int k = 0;
        for (int j=0; j<n-1; ++j){
            double pi1 = 0;
            for (int i=j; i<n; ++i){
                double pi0 = Math.abs(a[index[i]][j]);
                pi0 /= c[index[i]];
                if (pi0 > pi1){
                    pi1 = pi0;
                    k = i;
                }
            }
            // Interchange rows according to the pivoting order
            int itmp = index[j];
            index[j] = index[k];
            index[k] = itmp;
            for (int i=j+1; i<n; ++i){
                double pj = a[index[i]][j]/a[index[j]][j]; 
                // Record pivoting ratios below the diagonal
                a[index[i]][j] = pj;
                // Modify other elements accordingly
                for (int l=j+1; l<n; ++l)
                    a[index[i]][l] -= pj*a[index[j]][l];
            }
        }
    }
    
    /////////////////////Solution 2/////////////////////////////////////////////////////
    
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
		//zeros below the diagonal
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
		//zeros above the diagonal
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
    
    /////////////////////Solution 3/////////////////////////////////////////////////////	
	/*
	 *  get inverse matrix by LU Decomposition
	 */
	public static double[][] solution_3(double[][] in){
		//ArrayList<double[][]> LU = LUDecompositionDoolittle(in);
		ArrayList<double[][]> LU =  decomposition(in);
		double[][] inverse = getInverseFromLU(LU);
		return inverse;
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
        /*
        double[][] b_transpose = new double[len][len];
        for(int i=0; i< len; i++){
            for(int j=0; j<len; j++){
            	b_transpose[i][j] = b[j][i];
            }
        }
        
        return b_transpose;
        //*/
        return b;
    }
	
	
    /** 
     * Get matrix L and U. list.get(0) for L, list.get(1) for U 
     * @param a - Coefficient matrix of the equations 
     * @return matrix L and U, list.get(0) for L, list.get(1) for U 
     */  
    private static ArrayList<double[][]> decomposition(double[][] b) {  	
    	double[][] a = cloneArray(b);
    	
        //final double esp = 0.000001;          
        double[][] U = a;  
        double[][] L = createIdentiyMatrix(a.length);  
        ArrayList<double[][]> LU = new ArrayList<>();
          
        for(int j=0; j<a[0].length - 1; j++) {             
           // if(Math.abs(a[j][j]) < esp) {  
           //     throw new IllegalArgumentException("zero pivot encountered.");  
           // }  
              
            for(int i=j+1; i<a.length; i++) {  
                double mult = a[i][j] / a[j][j];   
                for(int k=j; k<a[i].length; k++) {  
                    U[i][k] = a[i][k] - a[j][k] * mult;  
                }  
                L[i][j] = mult;  
            }  
        }  
        LU.add(L);
        LU.add(U);
        
        return LU;  
    }
    
    private static ArrayList<double[][]> LUDecompositionDoolittle(double[][] X){
    	
        ArrayList<double[][]> LU = new ArrayList<>();

        double[][] a = cloneArray(X);
        
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
    
    private static double[][] cloneArray(double[][] b){
        int L = b.length;
        double[][] copy = new double[L][L];
        for(int i = 0; i < L; i++){
            for(int j = 0; j < L; j++){
                copy[i][j] = b[i][j];
            }
        }
        return copy;
    }
    
    private static double[][] createIdentiyMatrix(int n) { 
    	double[][] identyMatrix = new double[n][n];
    	for(int i=0; i<n; i++) {
    		for(int j=0; j<n; j++) {
    			if(i == j){
    				identyMatrix[i][j] = 1;
    			}
    		}
    	}
    	
    	return identyMatrix;
    }
    
    ///////////////////////Solution 4///////////////////////////////////////////////////	
    
	public static double[][] solution_4(double[][] in){
		NiMatrix _robot = new NiMatrix();
        double[][] inverse =  _robot.getNiMatrix(in);
		return inverse;
	}
    
    //////////////////////////////////////////////////////////////////////////	
}
