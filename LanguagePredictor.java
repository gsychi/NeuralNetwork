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
		
		String[] languages = {"english", "chinese"};
		
		String documents = System.getProperty ("user.home") + "/Documents/";
		FileReader lang1 = new FileReader (documents + "english.txt");
		
		buffer = new BufferedReader(lang1);
		ArrayList<String> wordInputBank = new ArrayList<String>();
		ArrayList<Double> confidenceFirstLang = new ArrayList<Double>();
		ArrayList<Double> confidenceSecondLang = new ArrayList<Double>();

		String word;
		while ((word = buffer.readLine()) != null) {
			wordInputBank.add(word.toUpperCase());
			confidenceFirstLang.add((double) 1);
			confidenceSecondLang.add((double) 0);
		}
		FileReader lang2 = new FileReader (documents + "chinese.txt");
		buffer = new BufferedReader(lang2);
		while ((word = buffer.readLine()) != null) {
			wordInputBank.add(word.toUpperCase());
			confidenceFirstLang.add((double) 0);
			confidenceSecondLang.add((double) 1);
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
			CONFIDENCE[i][0] = confidenceFirstLang.get(i);
			CONFIDENCE[i][1] = confidenceSecondLang.get(i);
		}
		
		//settings
		int hiddenNeuronsPerLayer = 15;
		String explanation = "ENGLISH ::: CHINESE. A higher first value denotes that the Network thinks it is English; a higher second value denotes that the Network thinks it is Chinese.";
		
		for (int i = 0;i<WORD_BANK_INPUT.length;i++) {
			System.out.println(Arrays.toString(WORD_BANK_INPUT[i]));
		}
		
		String NEW_WORD = "XIANG";
		double[] DATASET = process(NEW_WORD,indivWordLength);
		NeuralNetwork a = new NeuralNetwork(WORD_BANK_INPUT,CONFIDENCE,hiddenNeuronsPerLayer);
		a.trainNetwork(10000);
		//words analyzed
		System.out.println("\nWORDS ANALYZED: " + WORD_BANK_INPUT.length+"\n\n");
		
		for (int i = 0;i<WORD_BANK_INPUT.length;i++) {
			System.out.println("english to chinese probability:");
			System.out.println("\nword: " + wordInputBank.get(i));
			a.predict(WORD_BANK_INPUT[i],false);
			System.out.println("ACTUAL: "+ Arrays.toString(CONFIDENCE[i]));
			System.out.println("");
		}
		System.out.println(explanation);
		
		System.out.println("\nword: " + NEW_WORD);
		a.predict(DATASET, false);
		
		NEW_WORD = "HAPPY";
		DATASET = process(NEW_WORD,indivWordLength);
		System.out.println("\nword: " + NEW_WORD);
		a.predict(DATASET, false);

		NEW_WORD = "SHIAN";
		DATASET = process(NEW_WORD,indivWordLength);
		System.out.println("\nword: " + NEW_WORD);
		a.predict(DATASET, false);
		
		NEW_WORD = "STONE";
		DATASET = process(NEW_WORD,indivWordLength);
		System.out.println("\nword: " + NEW_WORD);
		a.predict(DATASET, false);
		
		NEW_WORD = "KAIMEN";
		DATASET = process(NEW_WORD,indivWordLength);
		System.out.println("\nword: " + NEW_WORD);
		a.predict(DATASET, false);
		
		NEW_WORD = "WOMEN";
		DATASET = process(NEW_WORD,indivWordLength);
		System.out.println("\nword: " + NEW_WORD);
		a.predict(DATASET, false);
		
		NEW_WORD = "DEMONS";
		DATASET = process(NEW_WORD,indivWordLength);
		System.out.println("\nword: " + NEW_WORD);
		a.predict(DATASET, false);
		
		NEW_WORD = "TALON";
		DATASET = process(NEW_WORD,indivWordLength);
		System.out.println("\nword: " + NEW_WORD);
		a.predict(DATASET, false);
		
	}
	
	private static double[] process(String word, int indivWordLength) {
		word = word.toUpperCase();
		double[] output = new double[indivWordLength];
		int len = indivWordLength-word.length();
		for (int j = 1;j<=len;j++) {
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
