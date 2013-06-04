package org.twaxonomer.bayes;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import org.twaxonomer.Supervized;
import org.twaxonomer.util.VectorUtil;

public class Bayes implements Supervized
{
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
