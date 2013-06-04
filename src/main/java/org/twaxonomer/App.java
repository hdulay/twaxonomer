package org.twaxonomer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.twaxonomer.bayes.Bayes;
import org.twaxonomer.bayes.BayesData;
import org.twaxonomer.bayes.Trained;
import org.twaxonomer.util.TwitterUtil;
import org.twaxonomer.util.WordsUtil;

public class App
{

	public static void main(String[] args)
	{
		try
		{
			if (args.length == 0)
			{
				args = new String[1];
				args[0] = "twaxonomer.properties";  
			}

			classify(args);
			
		}
		catch (Exception e)
		{
			System.err.println(e);
		}
	}

	protected static void classify(String[] args)
		throws FileNotFoundException, IOException, ConfigurationException
	{
		String tweetsDir = "tweets";
		String trainedDirName = "trained";
		
		File dir = new File(tweetsDir);
		File trainedDir = new File(dir, trainedDirName);
		File train = new File(trainedDir, "train");
		ArrayList<String> vocab = new ArrayList<String>();

		buildTrainingData(args, dir, trainedDir, train, 1);
		
		BayesData cd = WordsUtil.getBayesData(train, vocab);
		RealMatrix trainMatrix = cd.trainMatrix;
		
		Bayes bayes = new Bayes();
		Trained trained = bayes.train(trainMatrix, cd.trainCategory);
		Bayes.save(trainedDir, trained);
		
		while(true)
		{
			System.out.print(":>");
			BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
			String line = in.readLine();
			if(line.equals("quit")) 
			{
				in.close();
				break;
			}
			RealVector tweet = WordsUtil.bagOfWords2VecMN(vocab, line.split(" "));
			int classify = bayes.classify(trained, tweet);
			System.out.println((classify == 1) ? "bad" : "not bad");
		}
	}

	protected static void buildTrainingData(String[] args, File tweetsDir,
											File trained, File train, int count)
		throws FileNotFoundException, IOException, ConfigurationException
	{
		PropertiesConfiguration pc = new PropertiesConfiguration(args[0]);
		ArrayList<String> tweets = TwitterUtil.getTweets(pc, tweetsDir, count);
		
		if(!trained.exists()) trained.mkdir();
		BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
		for (String tweet : tweets)
		{
			tweet = tweet
				.toLowerCase()
				.replaceAll("[^a-zA-Z0-9\\s]", "");
			System.out.println(tweet);
			System.out.print(":>");
			while(true)
			{
				try
				{
					String line = in.readLine();
					if(line == null || line.isEmpty()) continue;
					
					int sentiment = Integer.parseInt(line);
					if(!train.exists()) train.createNewFile();
					FileOutputStream fos = new FileOutputStream(train, true);
					fos.write((tweet+"\t"+sentiment+"\n").getBytes());
					fos.close();
				}
				catch (Exception e)
				{
					System.err.println(e);
				}
				break;
			}
		}
	}
}
