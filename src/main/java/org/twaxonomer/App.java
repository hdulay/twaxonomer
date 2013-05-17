package org.twaxonomer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.json.JSONException;
import org.json.JSONObject;

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

			PropertiesConfiguration pc = new PropertiesConfiguration(args[0]);
			loadData(pc);
		}
		catch (ConfigurationException e)
		{
			System.err.println(e);
		}
		catch (MalformedURLException e)
		{
			System.err.println(e);
		}
		catch (FileNotFoundException e)
		{
			System.err.println(e);
		}
		catch (IOException e)
		{
			System.err.println(e);
		}
	}

	private static void loadData(PropertiesConfiguration pc)
		throws MalformedURLException, IOException, FileNotFoundException
	{
		String username = pc.getString("twitter.id");
		String password = pc.getString("twitter.pwd");
		String dir = pc.getString("twitter.dir");
		File tweets = new File(dir);
		if(!tweets.exists()) tweets.mkdir();

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
			String tweet;
			int count = 0;
			while ((tweet = br.readLine()) != null && count < 10)
			{
				try
				{
					System.out.println(tweet);
					JSONObject jo = new JSONObject(tweet);
					
					String id;
					if(jo.has("id")) id = jo.getString("id");
					else if(jo.has("id_str")) id = jo.getString("id_str");
					else continue;
					
					File tfile = new File(tweets, id+".txt");
					tfile.createNewFile();
					FileOutputStream fos = new FileOutputStream(tfile);
					fos.write(jo.getString("text").getBytes());
					fos.flush();
					fos.close();
					
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
		
	}
	
	protected RealMatrix trainNaiveBayes(List<String> vocabList,
											List<List<String>> messages)
	{
		double[][] m = new double[messages.get(0).size()][vocabList.size()];
		for (int i=0; i < messages.size(); i++)
		{
			RealVector vector = bagOfWords2VecMN(vocabList, messages.get(i));
			m[i] = vector.toArray();
		}
		return MatrixUtils.createRealMatrix(m);
	}
	
	/*
	 * def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

	 */
	protected RealVector bagOfWords2VecMN(List<String> vocabList, List<String> message)
	{
		RealMatrix mat = MatrixUtils.createRealMatrix(vocabList.size(), 1);
		for (String word : message)
		{
			if(vocabList.contains(word))
			{
				int index = vocabList.indexOf(word);
				mat.addToEntry(index, 0, 1);
			}
		}
		return mat.getColumnVector(0);
	}
	

	/*
	 * def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

	 */
	protected List<String> createVocabList(List<String> dataSet)
	{
		ArrayList<String> list = new ArrayList<String>();
		for (String string : dataSet)
		{
			if(!list.contains(string)) list.add(string);
		}
		return list;
	}
}
