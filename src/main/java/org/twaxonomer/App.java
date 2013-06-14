package org.twaxonomer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.clustering.MultiKMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.twaxonomer.bayes.Bayes;
import org.twaxonomer.bayes.Trained;
import org.twaxonomer.util.TwitterUtil;
import org.twaxonomer.util.WordsData;
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

//			classify(args);
			
			kmeans(args);
			
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

	protected static void kmeans(String[] args)
		throws FileNotFoundException, IOException
	{
		
		String tweetsDir = "tweets";
		String trainedDirName = "trained";
		
		File dir = new File(tweetsDir);
		File trainedDir = new File(dir, trainedDirName);
		File train = new File(trainedDir, "train");
		ArrayList<String> vocab = new ArrayList<String>();
		WordsData words = WordsUtil.getWordsData(train, vocab, 3);
		
		DistanceMeasure measure = new EuclideanDistance();

		ArrayList<Clusterable> mat = new ArrayList<Clusterable>();
		for (int i = 0; i < words.wordsMatrix.getRowDimension(); i++)
		{
			DoublePoint dp = new DoublePoint(words.wordsMatrix.getRow(i));
			mat.add(dp);
		}

		KMeansPlusPlusClusterer<Clusterable> km = new KMeansPlusPlusClusterer<Clusterable>(20, 100, measure);
		printKmeans(vocab, words, km.cluster(mat));
		
		MultiKMeansPlusPlusClusterer<Clusterable> mkm = new MultiKMeansPlusPlusClusterer<Clusterable>(km, 10);
		printKmeans(vocab, words, mkm.cluster(mat));
		
		DBSCANClusterer<Clusterable> dbscan = new DBSCANClusterer<Clusterable>(1.3, 1, measure);
		List<Cluster<Clusterable>> clusters = dbscan.cluster(mat);
		printDbscan(vocab, words, clusters);
	}

	private static void printDbscan(ArrayList<String> vocab, WordsData words,
								List<Cluster<Clusterable>> clusters)
	{
		System.out.println("==================================================");
		for (Cluster<Clusterable> cluster : clusters)
		{
			StringBuilder sb = new StringBuilder();
			sb.append("***dbscan:\n");
			sb.append(cluster);
			sb.append("\n***\n");
			List<Clusterable> points = cluster.getPoints();
			for (int i = 0; i < points.size(); i++)
			{
				Clusterable point = points.get(i);
				sb.append("\t"+findTweet(words, point)+"\n");
			}
			System.out.println(sb.toString());
		}
	}

	private static void printKmeans(ArrayList<String> vocab, WordsData words,
								List<CentroidCluster<Clusterable>> centroids)
	{
		System.out.println("==================================================");
		for (CentroidCluster<Clusterable> centroid : centroids)
		{
			StringBuilder sb = new StringBuilder();
			double[] mean = centroid.getCenter().getPoint();
			sb.append("***kmeans:\n");
			for (int i = 0; i < mean.length; i++)
			{
				double mn = mean[i];
				if(mn > .5)
				{
					String string = vocab.get(i);
					sb.append("["+string+"|"+mn+"]");
				}
			}
			sb.append("\n***\n");
			List<Clusterable> points = centroid.getPoints();
			for (int i = 0; i < points.size(); i++)
			{
				Clusterable point = points.get(i);
				sb.append("\t"+findTweet(words, point)+"\n");
			}
			System.out.println(sb.toString());
		}
	}

	private static String findTweet(WordsData words, Clusterable point)
	{
		for (int j = 0; j < words.wordsMatrix.getRowDimension(); j++)
		{
			double[] row = words.wordsMatrix.getRow(j);
			boolean match = true;
			for (int k = 0; k < row.length; k++)
			{
				if(row[k] == point.getPoint()[k]) 
					continue;
				else
				{
					match = false;
					break;
				}
			}
			if(match) return words.strings.get(j);
		}
		return "not found";
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
		
		WordsData cd = WordsUtil.getWordsData(train, vocab, 0);
		RealMatrix trainMatrix = cd.wordsMatrix;
		
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
			RealVector tweet = WordsUtil.bagOfWords2VecMN(vocab, line.split(" "), 0);
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
