package org.twaxonomer.bayes;

import java.io.Serializable;

import org.apache.commons.math3.linear.RealVector;

public class Trained implements Serializable
{
	private static final long serialVersionUID = 908498438648334215L;

	public double mean;
	public RealVector p1Vect;
	public RealVector p0Vect;
	
	public double getMean()
	{
		return mean;
	}
	public void setMean(double mean)
	{
		this.mean = mean;
	}
	public RealVector getP1Vect()
	{
		return p1Vect;
	}
	public void setP1Vect(RealVector p1Vect)
	{
		this.p1Vect = p1Vect;
	}
	public RealVector getP0Vect()
	{
		return p0Vect;
	}
	public void setP0Vect(RealVector p0Vect)
	{
		this.p0Vect = p0Vect;
	}

}
