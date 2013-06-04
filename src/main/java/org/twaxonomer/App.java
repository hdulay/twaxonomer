package org.twaxonomer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.json.JSONException;
import org.json.JSONObject;
import org.twaxonomer.bayes.Bayes;
import org.twaxonomer.bayes.ClassificationData;
import org.twaxonomer.bayes.Trained;

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

			String tweetsDir = "tweets";
			String trainedDirName = "trained";
			
			File dir = new File(tweetsDir);
			File trainedDir = new File(dir, trainedDirName);
			File train = new File(trainedDir, "train");
			ArrayList<String> vocab = new ArrayList<String>();

			buildTrainingData(args, dir, trainedDir, train, 1);
			
			ClassificationData cd = buildTrainFile(train, vocab);
			RealMatrix trainMatrix = cd.trainMatrix;
			
			Bayes bayes = new Bayes();
			Trained trained = bayes.train(trainMatrix, cd.trainCategory);
			save(trainedDir, trained);
			
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
				RealVector tweet = bagOfWords2VecMN(vocab, line.split(" "));
				int classify = bayes.classify(trained, tweet);
				System.out.println((classify == 1) ? "bad" : "not bad");
			}
			
		}
		catch (Exception e)
		{
			System.err.println(e);
		}
	}

	protected static void print(RealMatrix trainMatrix)
	{
		System.out.println('[');
		for (int i = 0; i < trainMatrix.getRowDimension(); i++)
		{
			System.out.print("  [");
			for (int j = 0; j < trainMatrix.getColumnDimension(); j++)
			{
				System.out.print(trainMatrix.getEntry(i, j)+" ");
			}
			System.out.println("  ]");
		}
		System.out.println(']');
	}

	private static ClassificationData buildTrainFile(File train, ArrayList<String> vocab)
		throws FileNotFoundException, IOException
	{
		ArrayList<String> tweets = new ArrayList<String>();
		ArrayList<Integer> cat = new ArrayList<Integer>();
		
		FileInputStream fis = new FileInputStream(train);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
		String line;
		while((line = br.readLine()) != null)
		{
			StringTokenizer st = new StringTokenizer(line, "\t");
			String tweet = st.nextToken();
			tweets.add(tweet);
			String[] words = tweet.split(" ");
			for (String word : words)
			{
				if(!vocab.contains(word)) vocab.add(word);
			}
			int isNeg = (st.hasMoreTokens()) ? 
					Integer.parseInt(st.nextToken()) : 
						0;
			cat.add(isNeg);
		}
		br.close();
		
		RealMatrix trainMatrix = MatrixUtils.createRealMatrix(tweets.size(),
			vocab.size());
		for (int i = 0; i < tweets.size(); i++)
		{
			String tweet = tweets.get(i);
			String[] tweetarray = tweet.split(" ");
			RealVector bag = bagOfWords2VecMN(vocab, tweetarray);
			trainMatrix.setRow(i, bag.toArray());
		}
		ClassificationData cd = new ClassificationData();
		cd.trainMatrix = trainMatrix;
		double[] data = new double[cat.size()];
		for (int i = 0; i < data.length; i++)
		{
			data[i] = cat.get(i);
		}
		cd.trainCategory = MatrixUtils.createRealVector(data);
		return cd;
	}

	private static void save(File train, Trained trained)
		throws IOException, FileNotFoundException
	{
		File file = new File(train, "trained.dat");
		if(!file.exists()) file.createNewFile();
		FileOutputStream fos = new FileOutputStream(file);
		ObjectOutputStream oos = new ObjectOutputStream(fos);
		oos.writeObject(trained);
		oos.flush();
		oos.close();
	}

	protected static void buildTrainingData(String[] args, File tweetsDir,
											File trained, File train, int count)
		throws FileNotFoundException, IOException, ConfigurationException
	{
		PropertiesConfiguration pc = new PropertiesConfiguration(args[0]);
		ArrayList<String> tweets = getTweets(pc, tweetsDir, count);
		
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

	private static ArrayList<String> getTweets(PropertiesConfiguration pc, File tweetsDir, int max)
		throws MalformedURLException, IOException, FileNotFoundException
	{
		ArrayList<String> tweets = new ArrayList<String>();
		String username = pc.getString("twitter.id");
		String password = pc.getString("twitter.pwd");
		if(!tweetsDir.exists()) tweetsDir.mkdir();

		URL feed = new URL(pc.getString("twitter.url"));

		URLConnection con = feed.openConnection();
		
		con.setReadTimeout(5000);
		String userpass = username + ":" + password;
		String basicAuth = "Basic "
			+ new String(new Base64().encode(userpass.getBytes()));
		con.setRequestProperty("Authorization", basicAuth);
		InputStream is = con.getInputStream();
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		try
		{
			String json;
			int count = 0;
			while ((json = br.readLine()) != null && count < max)
			{
				try
				{
					JSONObject jo = new JSONObject(json);
					
					if(!jo.has("id") && !jo.has("id_str")) continue;
					
					String tweet = jo.getString("text");
					tweets.add(tweet);
					
					count++;
				}
				catch (JSONException e)
				{
					System.out.println(e);
				}
			}
		}
		finally
		{
			br.close();
		}
		return tweets;
	}
	
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
}
