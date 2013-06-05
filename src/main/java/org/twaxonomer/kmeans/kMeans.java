package org.twaxonomer.kmeans;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.twaxonomer.util.VectorUtil;

public class kMeans
{
	public kMeans() {}
	
	public void cluster(RealMatrix mat, int k, DistanceMeasure distance)
	{
		
	}
	
	protected RealMatrix ramdomCentroids(RealMatrix mat, int k)
	{
		int n = mat.getColumnDimension();
		RealMatrix centroids = MatrixUtils.createRealMatrix(k, n);
		for (int i = 0; i < n; i++)
		{
			RealVector col = mat.getColumnVector(i);
			double minJ = VectorUtil.min(col);
			float rangeJ = (float) (VectorUtil.max(col) - minJ);
			RealVector rand = VectorUtil.rand(k, 1).getColumnVector(0);
			centroids.getColumnVector(i).setSubVector(i, rand.mapMultiply(rangeJ));
		}
		return centroids;
	}
}
