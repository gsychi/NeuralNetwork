package NEURAL_NETWORKS;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class LanguagePredictor {

	private static BufferedReader buffer;
	public static void main(String[] args) throws IOException {
		
		//You can have a different directory, if you so wish
		String documents = System.getProperty ("user.home") + "/Documents/";
		FileReader english = new FileReader (documents + "english.txt");
		
		buffer = new BufferedReader(english);
		ArrayList<String> wordInputBank = new ArrayList<String>();
		ArrayList<Double> confidenceEng = new ArrayList<Double>();
		ArrayList<Double> confidenceChin = new ArrayList<Double>();
		String word;
		while ((word = buffer.readLine()) != null) {
			wordInputBank.add(word.toUpperCase());
			confidenceEng.add((double) 1);
			confidenceChin.add((double) 0);
		}
		FileReader chinese = new FileReader (documents + "chinese.txt");
		buffer = new BufferedReader(chinese);
		while ((word = buffer.readLine()) != null) {
			wordInputBank.add(word.toUpperCase());
			confidenceEng.add((double) 0);
			confidenceChin.add((double) 1);
		}
		
		int indivWordLength = wordInputBank.get(0).length();
		for (int i=0;i<wordInputBank.size();i++) {
			int temp = wordInputBank.get(i).length();
			if (temp>indivWordLength) {
				indivWordLength = temp;
			}
		}

		String[][] wordBank = new String[wordInputBank.size()][indivWordLength];
		for (int i = 0;i<wordInputBank.size();i++) {
			String a = wordInputBank.get(i);
			int length = indivWordLength-wordInputBank.get(i).length();
			for (int j = 1;j<=length;j++) {
				a = a+"0";
			}
			String[] w = a.split("");
			wordBank[i] = w;
		}
		String alphabetString = "0ABCDEFGHIJKLMNOPQRSTUVWXYZ";
		String[] ALPHABET = alphabetString.split("");
		double[][] WORD_BANK_INPUT = new double[wordBank.length][wordBank[0].length];
		for (int i = 0;i<WORD_BANK_INPUT.length;i++) {
			for (int j = 0;j<indivWordLength;j++) {
				for (int k = 0;k<ALPHABET.length;k++) {
					if (wordBank[i][j].equals(ALPHABET[k])) {
						double index = (double) k;
						WORD_BANK_INPUT[i][j] = index/26;
					}
				}
			}
		}
		//Input Values are scaled to an array from 0 to 1.
		
		//NOW FOR OUTPUT
		double[][] CONFIDENCE = new double[wordBank.length][2];
		for (int i = 0;i<CONFIDENCE.length;i++) {
			CONFIDENCE[i][0] = confidenceEng.get(i);
			CONFIDENCE[i][1] = confidenceChin.get(i);
		}
		
		//settings
		int hiddenNeuronsPerLayer = 15;
		/*
		for (int i = 0;i<WORD_BANK_INPUT.length;i++) {
			System.out.println(Arrays.toString(WORD_BANK_INPUT[i]));
		}
		*/
		String NEW_WORD = "XIANG";
		double[] DATASET = process(NEW_WORD,indivWordLength);
		NeuralNetwork a = new NeuralNetwork(WORD_BANK_INPUT,CONFIDENCE,hiddenNeuronsPerLayer);
		a.trainNetwork(100000);
		//words analyzed
		System.out.println("\nWORDS ANALYZED: " + WORD_BANK_INPUT.length+"\n\n");
		
		for (int i = 0;i<WORD_BANK_INPUT.length;i++) {
			System.out.println("word: " + wordInputBank.get(i));
			System.out.println("english to chinese probability:");
			a.predict(WORD_BANK_INPUT[i],false);
			System.out.println("ACTUAL: "+ Arrays.toString(CONFIDENCE[i]));
			System.out.println("");
		}
		System.out.println("word: " + NEW_WORD);
		a.predict(DATASET, false);
		System.out.println("ENGLISH ::: CHINESE. A higher first value denotes that the Network thinks it is English; a higher second value denotes that the Network thinks it is Chinese.");
		
	}
	
	private static double[] process(String word, int indivWordLength) {
		word = word.toUpperCase();
		double[] output = new double[indivWordLength];
		int len = indivWordLength-word.length();
		for (int j = 1;j<=indivWordLength;j++) {
			word = word+"0";
		}
		String[] w = word.split("");
		String alphabetString = "0ABCDEFGHIJKLMNOPQRSTUVWXYZ";
		String[] ALPHABET = alphabetString.split("");
		for (int j = 0;j<indivWordLength;j++) {
			for (int k = 0;k<ALPHABET.length;k++) {
				if (w[j].equals(ALPHABET[k])) {
					double index = (double) k;
					output[j] = index/26;
				}
			}
		}
		
		return output;
	}
}
