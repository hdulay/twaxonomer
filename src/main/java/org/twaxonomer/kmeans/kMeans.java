package org.twaxonomer.kmeans;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.twaxonomer.util.MatIndex;
import org.twaxonomer.util.VectorUtil;

public class kMeans
{
	public kMeans() {}
	
	/*
	 * def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
	 *  m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                clusterAssmentif distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print centroids
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment
	 */
	protected kMeansData _cluster(RealMatrix dataSet, int k, DistanceMeasure distance)
	{
		int m = dataSet.getColumnDimension();
		RealMatrix clusterAssment = MatrixUtils.createRealMatrix(m, 2);
		RealMatrix centroids = randomCentroids(dataSet, k);
		boolean clusterChanged = true;
		while(clusterChanged)
		{
			clusterChanged = false;
			for (int i = 0; i < m; i++)
			{
				double minDist = Double.POSITIVE_INFINITY;
				double minIndex = -1;
				for (int j = 0; j < k; j++)
				{
					double distJI = distance.compute(centroids.getColumn(j), dataSet.getColumn(i));
					if(distJI < minDist)
					{
						minDist = distJI;
						minIndex = j;
					}
				}
				if(clusterAssment.getEntry(i,0) != minIndex) clusterChanged = true;
				clusterAssment.setEntry(i, 0, Math.pow(minDist,2));
			}
			for (int cent = 0; cent < k; cent++)
			{
				List<MatIndex> nonzeros = VectorUtil.nonzeros(clusterAssment);
				RealMatrix ptsInClust = VectorUtil.getData(nonzeros, dataSet);
				centroids.setRowVector(cent, VectorUtil.mean(ptsInClust));
			}
		}
		kMeansData kdata = new kMeansData();
		kdata.centroids = centroids;
		kdata.clusterAssment = clusterAssment;
		return kdata;
	}
	
	/*
	 * def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment
    
    (non-Javadoc)
	 * @see org.twaxonomer.kmeans.kMeans#cluster(org.apache.commons.math3.linear.RealMatrix, int, org.apache.commons.math3.ml.distance.DistanceMeasure)
	 */
	public kMeansData cluster(RealMatrix dataSet, int k, DistanceMeasure distance)
	{
//		m = shape(dataSet)[0]
		int m = dataSet.getColumnDimension();
//		    clusterAssment = mat(zeros((m,2)))
		RealMatrix clusterAssment = MatrixUtils.createRealMatrix(m, 2);
//		    centroid0 = mean(dataSet, axis=0).tolist()[0]
		double[] centroid0 = VectorUtil.mean(dataSet).toArray();
		//create a list with one centroid
		ArrayList<double[]> centList = new ArrayList<double[]>();
		centList.add(centroid0);
//		    for j in range(m):#calc initial Error
//		        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
//		    while (len(centList) < k):
//		        lowestSSE = inf
//		        for i in range(len(centList)):
//		            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
//		            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
//		            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
//		            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
//		            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
//		            if (sseSplit + sseNotSplit) < lowestSSE:
//		                bestCentToSplit = i
//		                bestNewCents = centroidMat
//		                bestClustAss = splitClustAss.copy()
//		                lowestSSE = sseSplit + sseNotSplit
//		        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
//		        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
//		        print 'the bestCentToSplit is: ',bestCentToSplit
//		        print 'the len of bestClustAss is: ', len(bestClustAss)
//		        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
//		        centList.append(bestNewCents[1,:].tolist()[0])
//		        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
//		    return mat(centList), clusterAssment
		return null;
	}
	
	protected RealMatrix randomCentroids(RealMatrix mat, int k)
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
