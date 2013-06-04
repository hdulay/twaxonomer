package org.twaxonomer.util;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class LearningUtil
{
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
