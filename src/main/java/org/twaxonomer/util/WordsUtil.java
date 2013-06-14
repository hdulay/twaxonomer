package org.twaxonomer.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class WordsUtil
{
	private WordsUtil() {}

	public static RealVector bagOfWords2VecMN(List<String> vocabList, String[] tweet, int size)
	{
		RealMatrix mat = MatrixUtils.createRealMatrix(1, vocabList.size());
		for (String word : tweet)
		{
			if (!word.startsWith("http") && vocabList.contains(word)
				&& word.length() > size)
			{
				int index = vocabList.indexOf(word);
				mat.addToEntry(0, index, 1);
			}
		}
		return mat.getRowVector(0);
	}
	
	public static List<String> createVocabList(List<String> dataSet)
	{
		ArrayList<String> list = new ArrayList<String>();
		for (String string : dataSet)
		{
			if(!list.contains(string)) list.add(string);
		}
		return list;
	}
	
	
	public static WordsData getWordsData(File train, ArrayList<String> vocab, int size)
		throws FileNotFoundException, IOException
	{
		ArrayList<String> strings = new ArrayList<String>();
		ArrayList<Integer> cat = new ArrayList<Integer>();

		FileInputStream fis = new FileInputStream(train);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
		String line;
		while ((line = br.readLine()) != null)
		{
			StringTokenizer st = new StringTokenizer(line, "\t");
			String string = st.nextToken();
			strings.add(string);
			String[] words = string.split(" ");
			for (String word : words)
			{
				if (!vocab.contains(word)) vocab.add(word);
			}
			int isNeg = (st.hasMoreTokens()) ? Integer.parseInt(st.nextToken()) : 0;
			cat.add(isNeg);
		}
		br.close();

		RealMatrix trainMatrix = MatrixUtils.createRealMatrix(strings.size(), vocab.size());
		for (int i = 0; i < strings.size(); i++)
		{
			String string = strings.get(i);
			String[] words = string.split(" ");
			RealVector bag = bagOfWords2VecMN(vocab, words, size);
			trainMatrix.setRow(i, bag.toArray());
		}
		WordsData cd = new WordsData();
		cd.wordsMatrix = trainMatrix;
		double[] data = new double[cat.size()];
		for (int i = 0; i < data.length; i++)
		{
			data[i] = cat.get(i);
		}
		cd.trainCategory = MatrixUtils.createRealVector(data);
		cd.strings = strings;
		return cd;
	}
}
