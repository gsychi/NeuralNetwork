package test;

public class Main {

	public static void main(String[] args) {
		double[][] INPUT_VALUES = {{0,0,1},{0,1,1},{1,0,1},{1,1,1},{1,0,0},{0,0,0},{0.5,0.6,0.2},{0.4,0.5,0.2},{0.1,0.9,0.9},{0.99,0.95,0.95},{0.98,0.97,0.97}};
		double[][] OUTPUT_VALUES = {{0},{1},{1},{0},{0},{1},{0.4},{0.85},{0.6},{0.55},{0.35}};

		//settings
		int hiddenNeuronsPerLayer = 50;
		double[] DATASET = {0.11,0.89,0.9};
		NeuralNetwork a = new NeuralNetwork(INPUT_VALUES,OUTPUT_VALUES,hiddenNeuronsPerLayer);
		a.trainNetwork(200000);
		for (int i = 0;i<INPUT_VALUES.length;i++) {
			a.predict(INPUT_VALUES[i]);
			System.out.println("");
		}
		a.predict(DATASET);
	}
}
