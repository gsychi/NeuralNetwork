package NEURAL_NETWORKS;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class LanguagePredictor2 {

	private static BufferedReader buffer;
	public static void main(String[] args) throws IOException {

		//settings
		boolean printTrainingProgress = true;
		int hiddenNeuronsPerLayer = 20;  
		int trainedWords = 0;
		int newWords = 50;
		double learningRate = 0.05;
		int iterations = 1000;

		String[] languages = {"english", "chinese"};

		String documents = System.getProperty ("user.home") + "/Documents/Procrastination Box/";
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

		String alphabetString = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; //; 0ABCDEFGHIJKLMNOPQRSTUVWXYZ
		String[] ALPHABET = alphabetString.split("");
		double[][] WORD_BANK_INPUT = new double[wordBank.length][wordBank[0].length*26];
		//set everything to 0 first
		for (int i = 0;i<WORD_BANK_INPUT.length;i++) {
			for (int j = 0;j<WORD_BANK_INPUT[0].length;j++) {
				WORD_BANK_INPUT[i][j] = 0;
			}
		}
		for (int i = 0;i<wordBank.length;i++) {
			for (int j = 0;j<indivWordLength;j++) {
				for (int k = 0;k<ALPHABET.length;k++) {
					if (ALPHABET[k].equals(wordBank[i][j])) {
						WORD_BANK_INPUT[i][(ALPHABET.length*j)+k] = 1;
					}
				}
			}
		}


		double[][] CONFIDENCE = new double[wordBank.length][2];
		for (int i = 0;i<CONFIDENCE.length;i++) {
			CONFIDENCE[i][0] = confidenceFirstLang.get(i);
			CONFIDENCE[i][1] = confidenceSecondLang.get(i);
		}

		int deck[] = new int[WORD_BANK_INPUT.length];
		for (int i = 0;i<deck.length;i++) {
			deck[i] = i;
		}			
		for (int i = 0;i<deck.length;i++) {
			int placement = (int) (Math.random()*deck.length);
			int temp = deck[placement];
			deck[placement] = deck[i];
			deck[i] = temp;
		}
		int totalWords = newWords + trainedWords;

		String[] totalTestWords = new String[totalWords]; // you choose
		for (int i = 0;i<trainedWords;i++) {
			totalTestWords[i] = wordInputBank.get(deck[i]);
		}
		FileReader unseen = new FileReader(documents + "unseen.txt");
		buffer = new BufferedReader(unseen);
		for (int i = trainedWords;i<totalTestWords.length;i++) {
			word = buffer.readLine().toUpperCase();
			totalTestWords[i] = word;
		}
		
		String[][] testWordBank = new String[totalTestWords.length][indivWordLength];
		for (int i = 0;i<testWordBank.length;i++) {
			String ab = totalTestWords[i].toUpperCase();
			int length = indivWordLength-totalTestWords[i].length();
			for (int j = 1;j<=length;j++) {
				ab = ab+"0";
			}
			String[] w = ab.split("");
			testWordBank[i] = w;
		}
		double[][] HOWCORRECT = new double[testWordBank.length][testWordBank[0].length*26];
		//set everything to 0 first
		for (int i = 0;i<HOWCORRECT.length;i++) {
			for (int j = 0;j<HOWCORRECT[0].length;j++) {
				HOWCORRECT[i][j] = 0;
			}
		}
		for (int i = 0;i<HOWCORRECT.length;i++) {
			for (int j = 0;j<indivWordLength;j++) {
				for (int k = 0;k<ALPHABET.length;k++) {
					if (ALPHABET[k].equals(testWordBank[i][j])) {
						HOWCORRECT[i][(ALPHABET.length*j)+k] = 1;
					}
				}
			}
		}


		double[][] NEW_WORDS_CONFIDENCE = new double[totalTestWords.length][2];
		for (int i = 0;i<trainedWords;i++) {
			NEW_WORDS_CONFIDENCE[i][0] = confidenceFirstLang.get(deck[i]);
			NEW_WORDS_CONFIDENCE[i][1] = confidenceSecondLang.get(deck[i]);
		}

		FileReader unseenAns = new FileReader(documents + "unseenAnswers.txt");
		buffer = new BufferedReader(unseenAns);
		for (int i = trainedWords;i<totalTestWords.length;i++) {
			String input = buffer.readLine();
			int ca = Integer.parseInt(input);
			NEW_WORDS_CONFIDENCE[i][0]=ca;
			NEW_WORDS_CONFIDENCE[i][1]=1-ca;
		}

		int test = 1;
		while (test<2) {
			System.out.println("---\nTEST #" + test);
			test++;
			NeuralNetwork a = new NeuralNetwork(WORD_BANK_INPUT,CONFIDENCE,hiddenNeuronsPerLayer,learningRate);

			for (int trials = 0; trials<(iterations/100);trials++) {
				a.trainNetwork(100);

				if (printTrainingProgress) {
					for (int i = 0;i<WORD_BANK_INPUT.length;i++) {
						System.out.println("\nword: " + wordInputBank.get(i));
						a.predict(WORD_BANK_INPUT[i], 0, false);
						System.out.println("ACTUAL: "+ Arrays.toString(CONFIDENCE[i]));
					}

				}
			}
			
			//PREDICTIONS BEGIN
			System.out.println("\nUNSEEN WORDS ---------------------\n");
			for (int i = 0;i<NEW_WORDS_CONFIDENCE.length;i++) {
				System.out.println("\n" + totalTestWords[i]+ "\nACTUAL:" + Arrays.toString(NEW_WORDS_CONFIDENCE[i])+"\nPREDICTION BY COMPUTER:");
				a.predict(HOWCORRECT[i], NEW_WORDS_CONFIDENCE[i][0],true);
			}
			System.out.println("\n"+(a.correct/totalWords*100) + "% correct.");
		}
	}
}
