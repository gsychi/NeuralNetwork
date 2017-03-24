package test;

import java.util.Arrays;

public class NeuralNetwork {

	double[][] synapse0;
	double[][] synapse1;
	double[][] finalSynapse;
	double[][] INPUT_VALUES;
	double[][] OUTPUT_VALUES;
	int hiddenLayer1;

	public NeuralNetwork(double[][] a, double[][] b, int d) {
		INPUT_VALUES=a;
		OUTPUT_VALUES=b;
		hiddenLayer1=d;
		synapse0 = mxjava.synapseLayer(a[0].length,d);
		synapse1 = mxjava.synapseLayer(d,d);
		finalSynapse = mxjava.synapseLayer(hiddenLayer1,OUTPUT_VALUES[0].length);
	}

	public void trainNetwork(int c) {
		int iterations = c;
		System.out.println("--BEGIN TRAINING--");
		for (int runs = 0;runs<=(iterations-1);runs++) {
			if (runs%(iterations/10) == 0) {
				System.out.println("thinking...");
			}
			double[][] layer0 = INPUT_VALUES;
			double[][] rawLayer1 = mxjava.matrixMult(layer0,synapse0);
			double[][] layer1 = new double[rawLayer1.length][rawLayer1[0].length];
			for (int i = 0;i<rawLayer1.length;i++) {
				for (int j = 0;j<rawLayer1[0].length;j++) {
					layer1[i][j] = mxjava.sigmoidPackage(rawLayer1[i][j],false);
				}
			} //should return a INPUT_VALUES.length x hiddenLayer1 matrix

			double[][] rawLayer2 = mxjava.matrixMult(layer1,synapse1);
			double[][] layer2 = new double[rawLayer2.length][rawLayer2[0].length];
			for (int i = 0;i<rawLayer2.length;i++) {
				for (int j = 0;j<rawLayer2[0].length;j++) {
					layer2[i][j] = mxjava.sigmoidPackage(rawLayer2[i][j],false);
				}
			} //returns INPUT_VALUES.length x hiddenLayer1 matrix

			double[][] rawFinalLayer = mxjava.matrixMult(layer2,finalSynapse);
			double[][] finalLayer = new double[rawFinalLayer.length][rawFinalLayer[0].length];
			for (int i = 0;i<rawFinalLayer.length;i++) {
				for (int j = 0;j<rawFinalLayer[0].length;j++) {
					finalLayer[i][j] = mxjava.sigmoidPackage(rawFinalLayer[i][j],false);
				}
			} //returns INPUT_VALUES.length x OUTPUT_VALUES[0].length matrix.

			//finalLayer delta
			double[][] finalLayerError = mxjava.subtract(OUTPUT_VALUES,finalLayer);
			double[][] sigmoidDerivativeForfinalLayer = new double[finalLayer.length][finalLayer[0].length];
			for (int i = 0;i<finalLayer.length;i++) {
				for (int j = 0;j<finalLayer[0].length;j++) {
					sigmoidDerivativeForfinalLayer[i][j] = mxjava.sigmoidPackage(finalLayer[i][j],true);
				}
			}
			double[][] finalLayerDelta = mxjava.scalarMult(finalLayerError,sigmoidDerivativeForfinalLayer);

			//layer 2 delta
			double[][] layer2Error = mxjava.matrixMult(finalLayerDelta,mxjava.transpose(finalSynapse));
			double[][] sigmoidDerivativeForLayer2 = new double[layer2.length][layer2[0].length];
			for (int i = 0;i<layer1.length;i++) {
				for (int j = 0;j<layer1[0].length;j++) {
					sigmoidDerivativeForLayer2[i][j] = mxjava.sigmoidPackage(layer2[i][j],true);
				}
			}
			double[][] layer2Delta = mxjava.scalarMult(layer2Error,sigmoidDerivativeForLayer2);

			//layer 1 delta
			double[][] layer1Error = mxjava.matrixMult(layer2Delta,mxjava.transpose(synapse1));
			double[][] sigmoidDerivativeForLayer1 = new double[layer1.length][layer1[0].length];
			for (int i = 0;i<layer1.length;i++) {
				for (int j = 0;j<layer1[0].length;j++) {
					sigmoidDerivativeForLayer1[i][j] = mxjava.sigmoidPackage(layer1[i][j],true);
				}
			}
			double[][] layer1Delta = mxjava.scalarMult(layer1Error,sigmoidDerivativeForLayer1);


			double[][] finalWeight = mxjava.matrixMult(mxjava.transpose(layer2),finalLayerDelta);
			double[][] weight1 = mxjava.matrixMult(mxjava.transpose(layer1),layer2Delta);
			double[][] weight0 = mxjava.matrixMult(mxjava.transpose(layer0),layer1Delta);

			finalSynapse = mxjava.add(finalSynapse,finalWeight);
			synapse1 = mxjava.add(synapse1,weight1);
			synapse0 = mxjava.add(synapse0,weight0);
		}
		System.out.println("--END TRAINING--");
	}
	
	public void predict(double[] NEW_VALUE) {
		double[][] newData = {NEW_VALUE};
		double[][] rawLayer1 = mxjava.matrixMult(newData,synapse0);
		double[][] layer1 = new double[rawLayer1.length][rawLayer1[0].length];
		for (int i = 0;i<rawLayer1.length;i++) {
			for (int j = 0;j<rawLayer1[0].length;j++) {
				layer1[i][j] = mxjava.sigmoidPackage(rawLayer1[i][j],false);
			}
		}
		double[][] rawLayer2 = mxjava.matrixMult(layer1,synapse1);
		double[][] layer2 = new double[rawLayer2.length][rawLayer2[0].length];
		for (int i = 0;i<rawLayer2.length;i++) {
			for (int j = 0;j<rawLayer2[0].length;j++) {
				layer2[i][j] = mxjava.sigmoidPackage(rawLayer2[i][j],false);
			}
		}			
		double[][] rawFinalLayer = mxjava.matrixMult(layer2,finalSynapse);
		double[][] finalLayer = new double[rawFinalLayer.length][rawFinalLayer[0].length];
		for (int i = 0;i<rawFinalLayer.length;i++) {
			for (int j = 0;j<rawFinalLayer[0].length;j++) {
				finalLayer[i][j] = mxjava.sigmoidPackage(rawFinalLayer[i][j],false);
			}
		}
		System.out.println("Data Set: " + Arrays.toString(NEW_VALUE));
		System.out.println("Predicted Result: " + Arrays.toString(finalLayer[0]));
	}
		
}
