package org.twaxonomer.util;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class VectorUtil
{
	public static UnivariateFunction LOG = new UnivariateFunction()
	{

		public double value(double x)
		{
			double log2 = Math.log(x);
			return log2;
		}
	};

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
}
