package org.twaxonomer.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class VectorUtil
{
	private VectorUtil() {}
	
	public static UnivariateFunction LOG = new UnivariateFunction()
	{

		public double value(double x)
		{
			double log2 = Math.log(x);
			return log2;
		}
	};
	
	public static double min(RealVector vect)
	{
		double min = Double.NaN;
		double[] data = vect.toArray();
		for (int i = 0; i < data.length; i++)
		{
			if(min > data[i]) min = data[i];
		}
		if (min == Double.NaN)
			throw new IllegalArgumentException("vector doesn't contain a value");
		return min;
	}
	
	public static double max(RealVector vect)
	{
		double max = Double.NaN;
		double[] data = vect.toArray();
		for (int i = 0; i < data.length; i++)
		{
			if(max < data[i]) max = data[i];
		}
		if (max == Double.NaN)
			throw new IllegalArgumentException("vector doesn't contain a value");
		return max;
	}
	
	public static double sum(RealVector vect)
	{
		double sum = 0;
		double[] data = vect.toArray();
		for (int i = 0; i < data.length; i++)
		{
			sum += data[i];
		}
		return sum;
	}
	
	public static RealMatrix rand(int rows, int col)
	{
		RealMatrix m = MatrixUtils.createRealMatrix(rows, col);
		Random r = new Random();
		for (int i = 0; i < m.getColumnDimension(); i++)
		{
			RealVector v = m.getColumnVector(i);
			for (int j = 0; j < v.getDimension(); j++)
			{
				v.setEntry(j, r.nextDouble());
			}
		}
		return m;
	}

	public static RealVector ones(int rows)
	{
		RealMatrix m = MatrixUtils.createRealMatrix(rows, 1);
		for (int i = 0; i < m.getRowDimension(); i++)
		{
			m.setEntry(i, 0, 1);
		}
		return m.getColumnVector(0);
	}

	public static double[] zeros(int count)
	{
		return MatrixUtils.createRealMatrix(count, 1).getColumn(0);
	}
	
	public static List<MatIndex> nonzeros(RealMatrix m)
	{
		ArrayList<MatIndex> list = new ArrayList<MatIndex>();
		double[][] data = m.getData();
		for (int row = 0; row < data.length; row++)
		{
			double[] data2 = data[row];
			for (int col = 0; col < data2.length; col++)
			{
				if(data2[col] != 0)
					list.add(new MatIndex(row, col));
			}
		}
		return list;
	}
	
	public static void print(RealMatrix trainMatrix)
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

	public static RealMatrix getData(List<MatIndex> indices, RealMatrix dataSet)
	{
		double[][] data = new double[1][indices.size()];
		for (int i = 0; i < data.length; i++)
		{
			MatIndex index = indices.get(i);
			double value = dataSet.getEntry(index.row, index.col);
			if(value != 0) data[0][i] = value;
		}
		return MatrixUtils.createRealMatrix(data);
	}

	public static double mean(RealVector dataSet)
	{
		double mean = 0;
		for (int j = 0; j < dataSet.getDimension(); j++)
		{
			mean += dataSet.getEntry(j);
		}
		return mean / dataSet.getDimension();
	}

	public static RealVector mean(RealMatrix dataSet)
	{
		RealMatrix tmp = MatrixUtils.createRealMatrix(1, dataSet.getColumnDimension());
		RealVector v = tmp.getColumnVector(0);
		for (int i = 0; i < dataSet.getColumnDimension(); i++)
		{
			RealVector col = dataSet.getColumnVector(i);
			v.setEntry(i, mean(col));
		}
		
		return v;
	}
}
