package test;

public class UpgradedNeuralNetwork {

	public static double sigmoidPackage(double x, boolean deriv) {
		if (deriv) {
			return x*(1-x);
		}
		return 1/(1+Math.pow(Math.E,-x));
	}

	public static double[][] synapseLayer(int inputs, int outputs) {
		double[][] LAYER = new double[inputs][outputs];
		for (int i = 0;i<inputs;i++) {
			for (int j = 0;j<outputs;j++) {
				LAYER[i][j] = (Math.random()*2)-1;
			}
		}
		return LAYER;
	}

	public static double[][] matrixMult(double[][] firstMatrix, double[][] secondMatrix) {
		double[][] newMatrix = new double[firstMatrix.length][secondMatrix[0].length];
		for (int i = 0; i < newMatrix.length; i++) { 
			for (int j = 0; j < newMatrix[0].length; j++) { 
				for (int k = 0; k < firstMatrix[0].length; k++) { 
					newMatrix[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
				}
			}
		}
		return newMatrix;
	}
	public static double[][] arrayMult(double[][]firstMatrix,double[][]secondMatrix) {
		double[][] newMatrix = new double[firstMatrix.length][secondMatrix[0].length];
		for (int i = 0; i < newMatrix.length; i++) { 
			for (int j = 0; j < newMatrix[0].length; j++) { 
				newMatrix[i][j] = firstMatrix[i][j]*secondMatrix[i][j];
			}
		}
		return newMatrix;
	}
	public static double[][] subtract(double[][] firstMatrix, double[][] secondMatrix) {
		double[][] result = new double[firstMatrix.length][firstMatrix[0].length];
		for (int i = 0; i < firstMatrix.length; i++) {
			for (int j = 0; j < firstMatrix[0].length; j++) {
				result[i][j] = firstMatrix[i][j] - secondMatrix[i][j];
			}
		}
		return result;
	}
	public static double[][] add(double[][] firstMatrix, double[][] secondMatrix) {
		double[][] result = new double[firstMatrix.length][firstMatrix[0].length];
		for (int i = 0; i < firstMatrix.length; i++) {
			for (int j = 0; j < firstMatrix[0].length; j++) {
				result[i][j] = firstMatrix[i][j] + secondMatrix[i][j];
			}
		}
		return result;
	}

	public static double[][] transpose(double [][] m){
		double[][] temp = new double[m[0].length][m.length];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				temp[j][i] = m[i][j];
		return temp;
	}

	public static void print(String x, double[][]m) {
		System.out.println(x);
		for (int i = 0;i<m.length;i++) {
			for (int j = 0;j<m[0].length;j++) {
				System.out.println(m[i][j]);
			}
			System.out.println("-----row done-----");
		}		
		System.out.println("\n");
	}

	public static void main(String[] args) {
		System.out.println("NEURAL NETWORK TEST:\n-------------------");

		double[][] INPUT_VALUES = {{0,0,1},{0,1,1},{1,0,1},{1,1,1},{0.5,0.6,0.2}};
		double[][] OUTPUT_VALUES = {{0},{1},{1},{0},{0.4}};
		int iterations = 50000;
		int inputNodes = INPUT_VALUES[0].length;
		int neuronsInMiddleLayer = 5;

		double[][] synapse0 = synapseLayer(inputNodes,neuronsInMiddleLayer);
		//{{-0.16595599,0.44064899,-0.99977125,-0.39533485},{-0.70648822 ,-0.81532281,-0.62747958,-0.30887855},{-0.20646505,0.07763347 ,-0.16161097,0.370439}};
		double[][] synapse1 = synapseLayer(neuronsInMiddleLayer,1);
		//{{-0.5910955},{0.75623487},{-0.94522481},{0.34093502}};

		for (int runs = 0;runs<=iterations;runs++) {
			double[][] layer0 = INPUT_VALUES;
			double[][] rawLayer1 = matrixMult(layer0,synapse0);
			double[][] layer1 = new double[rawLayer1.length][rawLayer1[0].length];
			for (int i = 0;i<rawLayer1.length;i++) {
				for (int j = 0;j<rawLayer1[0].length;j++) {
					layer1[i][j] = sigmoidPackage(rawLayer1[i][j],false);
				}
			}
			//print("LAYER 1",layer1);

			double[][] rawLayer2 = matrixMult(layer1,synapse1);
			double[][] layer2 = new double[rawLayer2.length][rawLayer2[0].length];
			for (int i = 0;i<rawLayer2.length;i++) {
				for (int j = 0;j<rawLayer2[0].length;j++) {
					layer2[i][j] = sigmoidPackage(rawLayer2[i][j],false);
				}
			}
			//print("LAYER 2",layer2);

			//layer2 delta
			double[][] layer2Error = subtract(OUTPUT_VALUES,layer2);
			double[][] sigmoidDerivativeForLayer2 = new double[layer2.length][layer2[0].length];
			for (int i = 0;i<layer2.length;i++) {
				for (int j = 0;j<layer2[0].length;j++) {
					sigmoidDerivativeForLayer2[i][j] = sigmoidPackage(layer2[i][j],true);
				}
			}
			double[][] layer2Delta = arrayMult(layer2Error,sigmoidDerivativeForLayer2);

			//layer 1 delta
			double[][] layer1Error = matrixMult(layer2Delta,transpose(synapse1));
			double[][] sigmoidDerivativeForLayer1 = new double[layer1.length][layer1[0].length];
			for (int i = 0;i<layer1.length;i++) {
				for (int j = 0;j<layer1[0].length;j++) {
					sigmoidDerivativeForLayer1[i][j] = sigmoidPackage(layer1[i][j],true);
				}
			}
			double[][] layer1Delta = arrayMult(layer1Error,sigmoidDerivativeForLayer1);
			double[][] weight1 = matrixMult(transpose(layer1),layer2Delta);
			double[][] weight0 = matrixMult(transpose(layer0),layer1Delta);

			synapse1 = add(synapse1,weight1);
			synapse0 = add(synapse0,weight0);

			if (runs==iterations) {
				//RUN PREDICTION WITH GIVEN ARRAY
				double[][] newData = {{1,1,1}};
				rawLayer1 = matrixMult(newData,synapse0);
				layer1 = new double[rawLayer1.length][rawLayer1[0].length];
				for (int i = 0;i<rawLayer1.length;i++) {
					for (int j = 0;j<rawLayer1[0].length;j++) {
						layer1[i][j] = sigmoidPackage(rawLayer1[i][j],false);
					}
				}
				rawLayer2 = matrixMult(layer1,synapse1);
				layer2 = new double[rawLayer2.length][rawLayer2[0].length];
				for (int i = 0;i<rawLayer2.length;i++) {
					for (int j = 0;j<rawLayer2[0].length;j++) {
						layer2[i][j] = sigmoidPackage(rawLayer2[i][j],false);
					}
				}			
				print("Predicted Value: ",layer2);
			}
		}
	}
}