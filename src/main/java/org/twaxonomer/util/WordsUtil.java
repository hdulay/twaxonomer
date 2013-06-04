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
import org.twaxonomer.bayes.BayesData;

public class WordsUtil
{
	private WordsUtil() {}

	public static RealVector bagOfWords2VecMN(List<String> vocabList, String[] tweet)
	{
		RealMatrix mat = MatrixUtils.createRealMatrix(vocabList.size(), 1);
		for (String word : tweet)
		{
			if(vocabList.contains(word))
			{
				int index = vocabList.indexOf(word);
				mat.addToEntry(index, 0, 1);
			}
		}
		return mat.getColumnVector(0);
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
	
	
	public static BayesData getBayesData(File train, ArrayList<String> vocab)
		throws FileNotFoundException, IOException
	{
		ArrayList<String> tweets = new ArrayList<String>();
		ArrayList<Integer> cat = new ArrayList<Integer>();

		FileInputStream fis = new FileInputStream(train);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
		String line;
		while ((line = br.readLine()) != null)
		{
			StringTokenizer st = new StringTokenizer(line, "\t");
			String tweet = st.nextToken();
			tweets.add(tweet);
			String[] words = tweet.split(" ");
			for (String word : words)
			{
				if (!vocab.contains(word)) vocab.add(word);
			}
			int isNeg = (st.hasMoreTokens()) ? Integer.parseInt(st.nextToken()) : 0;
			cat.add(isNeg);
		}
		br.close();

		RealMatrix trainMatrix = MatrixUtils.createRealMatrix(tweets.size(), vocab.size());
		for (int i = 0; i < tweets.size(); i++)
		{
			String tweet = tweets.get(i);
			String[] tweetarray = tweet.split(" ");
			RealVector bag = bagOfWords2VecMN(vocab, tweetarray);
			trainMatrix.setRow(i, bag.toArray());
		}
		BayesData cd = new BayesData();
		cd.trainMatrix = trainMatrix;
		double[] data = new double[cat.size()];
		for (int i = 0; i < data.length; i++)
		{
			data[i] = cat.get(i);
		}
		cd.trainCategory = MatrixUtils.createRealVector(data);
		return cd;
	}
}
