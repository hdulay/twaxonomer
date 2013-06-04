package org.twaxonomer.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.json.JSONException;
import org.json.JSONObject;

public class TwitterUtil
{
	private TwitterUtil() {}

	public static ArrayList<String> getTweets(PropertiesConfiguration pc, File tweetsDir, int max)
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

}
