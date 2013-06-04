package org.twaxonomer.bayes;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.StringTokenizer;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import org.twaxonomer.Supervized;
import org.twaxonomer.util.LearningUtil;
import org.twaxonomer.util.VectorUtil;

public class Bayes implements Supervized
{
	/*
	 * def trainNB0(trainMatrix,trainCategory): numTrainDocs = len(trainMatrix)
	 * numWords = len(trainMatrix[0]) pAbusive =
	 * sum(trainCategory)/float(numTrainDocs) p0Num = ones(numWords); p1Num =
	 * ones(numWords) #change to ones() p0Denom = 2.0; p1Denom = 2.0 #change to
	 * 2.0 for i in range(numTrainDocs): if trainCategory[i] == 1: p1Num +=
	 * trainMatrix[i] p1Denom += sum(trainMatrix[i]) else: p0Num +=
	 * trainMatrix[i] p0Denom += sum(trainMatrix[i]) p1Vect = log(p1Num/p1Denom)
	 * #change to log() p0Vect = log(p0Num/p0Denom) #change to log() return
	 * p0Vect,p1Vect,pAbusive
	 */
	public Trained train(RealMatrix trainMatrix, RealVector trainCategory)
	{
		int numTrainDocs = trainMatrix.getRowDimension();
		int numWords = trainMatrix.getColumnDimension();
		double mean = VectorUtil.sum(trainCategory) / (float) numTrainDocs;
		RealVector p0Num = VectorUtil.ones(numWords);
		RealVector p1Num = VectorUtil.ones(numWords);

		double p0Denom = 2.0;
		double p1Denom = 2.0;

		for (int i = 0; i < numTrainDocs; i++)
		{
			if (trainCategory.getEntry(i) == 1)
			{
				p1Num = p1Num.add(trainMatrix.getRowVector(i));
				p1Denom += VectorUtil.sum(trainMatrix.getRowVector(i));
			}
			else
			{
				p0Num = p0Num.add(trainMatrix.getRowVector(i));
				p0Denom += VectorUtil.sum(trainMatrix.getRowVector(i));
			}
		}

		RealVector p1Vect = p1Num.mapDivide(p1Denom).map(VectorUtil.LOG);
		RealVector p0Vect = p0Num.mapDivide(p0Denom).map(VectorUtil.LOG);
		Trained t = new Trained();
		t.mean = mean;
		t.p1Vect = p1Vect;
		t.p0Vect = p0Vect;
		return t;
	}

	/*
	 * p1 = sum(vec2Classify * p1Vec) + log(pClass1) #element-wise mult p0 =
	 * sum(vec2Classify * p0Vec) + log(1.0 - pClass1) if p1 > p0: return 1 else:
	 * return 0
	 */
	public int classify(Trained trained, RealVector data)
	{
		RealVector p1data = data.copy();
		RealVector p2data = data;
		p1data.walkInOptimizedOrder(getVisitor(trained.p1Vect));
		double p1 = VectorUtil.sum(p1data) + Math.log(trained.mean);
		p2data.walkInOptimizedOrder(getVisitor(trained.p0Vect));
		double p0 = VectorUtil.sum(p2data) + Math.log(1.0 - trained.mean);

		if (p1 > p0)
			return 1;
		else
			return 0;
	}

	public static void save(File train, Trained trained)
		throws IOException, FileNotFoundException
	{
		File file = new File(train, "trained.dat");
		if (!file.exists()) file.createNewFile();
		FileOutputStream fos = new FileOutputStream(file);
		ObjectOutputStream oos = new ObjectOutputStream(fos);
		oos.writeObject(trained);
		oos.flush();
		oos.close();
	}

	public static ClassificationData getClassificationData(File train,
															ArrayList<String> vocab)
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

		RealMatrix trainMatrix = MatrixUtils
			.createRealMatrix(tweets.size(), vocab.size());
		for (int i = 0; i < tweets.size(); i++)
		{
			String tweet = tweets.get(i);
			String[] tweetarray = tweet.split(" ");
			RealVector bag = LearningUtil.bagOfWords2VecMN(vocab, tweetarray);
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

	private RealVectorChangingVisitor getVisitor(final RealVector vect)
	{
		RealVectorChangingVisitor multiply = new RealVectorChangingVisitor()
		{
			public double visit(int index, double value)
			{
				return vect.getEntry(index) * value;
			}

			public void start(int dimension, int start, int end)
			{
			}

			public double end()
			{
				return 0;
			}
		};
		return multiply;
	}
}
