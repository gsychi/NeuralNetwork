package test;

import java.util.Arrays;

public class MultiLayerNeuralNetwork {

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
		
		//training information
		double[][] INPUT_VALUES = {{0,0,1},{0,1,1},{1,0,1},{1,1,1},{1,0,0},{0,0,0},{0.5,0.6,0.2}};
		double[][] OUTPUT_VALUES = {{0},{1},{1},{0},{0},{1},{0.4}};
		
		//settings
		double[] NEW_VALUE = {0,0,0};
		int iterations = 50000;
		int hiddenLayer1 = 15;

		System.out.println("SETTINGS");
		System.out.println("Iterations: " + iterations);
		System.out.println("Total # of Examples: " + INPUT_VALUES.length);
		System.out.println("Neurons in Hidden Layer: " + hiddenLayer1);
		System.out.println("Predicting Dataset: " + Arrays.toString(NEW_VALUE));
		System.out.println("thinking...");
		double[][] synapse0 = synapseLayer(INPUT_VALUES[0].length,hiddenLayer1);
		double[][] synapse1 = synapseLayer(hiddenLayer1,hiddenLayer1);
		double[][] finalSynapse = synapseLayer(hiddenLayer1,OUTPUT_VALUES[0].length);

		for (int runs = 0;runs<=iterations;runs++) {
			double[][] layer0 = INPUT_VALUES;
			double[][] rawLayer1 = matrixMult(layer0,synapse0);
			double[][] layer1 = new double[rawLayer1.length][rawLayer1[0].length];
			for (int i = 0;i<rawLayer1.length;i++) {
				for (int j = 0;j<rawLayer1[0].length;j++) {
					layer1[i][j] = sigmoidPackage(rawLayer1[i][j],false);
				}
			}
			double[][] rawLayer2 = matrixMult(layer1,synapse1);
			double[][] layer2 = new double[rawLayer2.length][rawLayer2[0].length];
			for (int i = 0;i<rawLayer2.length;i++) {
				for (int j = 0;j<rawLayer2[0].length;j++) {
					layer2[i][j] = sigmoidPackage(rawLayer2[i][j],false);
				}
			}

			double[][] rawFinalLayer = matrixMult(layer2,finalSynapse);
			double[][] finalLayer = new double[rawFinalLayer.length][rawFinalLayer[0].length];
			for (int i = 0;i<rawFinalLayer.length;i++) {
				for (int j = 0;j<rawFinalLayer[0].length;j++) {
					finalLayer[i][j] = sigmoidPackage(rawFinalLayer[i][j],false);
				}
			}

			//finalLayer delta
			double[][] finalLayerError = subtract(OUTPUT_VALUES,finalLayer);
			double[][] sigmoidDerivativeForfinalLayer = new double[finalLayer.length][finalLayer[0].length];
			for (int i = 0;i<finalLayer.length;i++) {
				for (int j = 0;j<finalLayer[0].length;j++) {
					sigmoidDerivativeForfinalLayer[i][j] = sigmoidPackage(finalLayer[i][j],true);
				}
			}
			double[][] finalLayerDelta = arrayMult(finalLayerError,sigmoidDerivativeForfinalLayer);

			//layer 2 delta
			double[][] layer2Error = matrixMult(finalLayerDelta,transpose(finalSynapse));
			double[][] sigmoidDerivativeForLayer2 = new double[layer2.length][layer2[0].length];
			for (int i = 0;i<layer1.length;i++) {
				for (int j = 0;j<layer1[0].length;j++) {
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
			
			
			double[][] finalWeight = matrixMult(transpose(layer2),finalLayerDelta);
			double[][] weight1 = matrixMult(transpose(layer1),layer2Delta);
			double[][] weight0 = matrixMult(transpose(layer0),layer1Delta);

			finalSynapse = add(finalSynapse,finalWeight);
			synapse1 = add(synapse1,weight1);
			synapse0 = add(synapse0,weight0);

			if (runs==iterations) {
				//RUN PREDICTION WITH GIVEN ARRAY
				double[][] newData = {NEW_VALUE};
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
				rawFinalLayer = matrixMult(layer2,finalSynapse);
				finalLayer = new double[rawFinalLayer.length][rawFinalLayer[0].length];
				for (int i = 0;i<rawFinalLayer.length;i++) {
					for (int j = 0;j<rawFinalLayer[0].length;j++) {
						finalLayer[i][j] = sigmoidPackage(rawFinalLayer[i][j],false);
					}
				}			
				System.out.println("Predicted Result: " + finalLayer[0][0]);
				System.out.println("----end test----");
			}
		}
	}
}