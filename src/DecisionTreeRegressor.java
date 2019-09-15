/**
 * This class implements a Decision Tree Regressor using the CART algorithm.
 * The splitting criterion used in this project is mean squared error. An early
 * stopping threshold can be set when initializing the regressor. The loss 
 * function of the threshold is also MSE
 * 
 * @author Winston Lin
 */
import java.io.*;
import java.util.*;

public class DecisionTreeRegressor 
{
    Node root; // Root node
    boolean[] categorical; // Boolean array of whether feature is categorical
    double errorThreshold = 0; // Cut-off threshold for early stopping
    
    public DecisionTreeRegressor(double errorThreshold)
    {
        this.errorThreshold = errorThreshold;
    }
    
    /**
     * This method fits the regressor on the training set. It determines
     * categorical features in the datasets used in this project and calls
     * the buildTree() method to start building the tree
     * 
     * @param X is the attributes of the training set
     * @param y is the targets of the training set
     * @param fileName is the name of the dataset
     */
    public void fit(ArrayList<String[]> X, ArrayList<String> y, 
            String fileName)
    {
        // All 3 datasets for regression have only numeric features
        categorical = new boolean[X.get(0).length];
        
        root = buildTree(X, y);
    }
    
    /**
     * This method calls the chooseSplitIndex() method to choose the feature
     * and value at each split. It then uses those information to build the
     * tree recursively until we run out of features, impurity = 0, or the
     * cut-off threshold is reached
     * 
     * @param X is the attributes of the data points at each node
     * @param y is the targets of the data points at each node
     * @return a child node to build the tree recursively
     */
    public Node buildTree(ArrayList<String[]> X, ArrayList<String> y)
    {
        Node node = new Node();
        ArrayList<Object> params = chooseSplitIndex(X, y);
        int index;
        double value = 0;
        ArrayList<String> categories = new ArrayList<String>();
        boolean stop = false; // Indicate early stopping
        if (params.isEmpty())
        {
            index = -1;
        }
        else
        {
            index = (int) params.get(0);
            value = (double) params.get(1);
            categories = (ArrayList<String>) params.get(2);
        }
        
        stop = earlyStopping(y);
        if (index == -1 || diffValues(y).size() == 1 || stop)
        {/* Three conditions for leaf: 1. Run out of features 2. Impurity = 0
                                       3. Early stopping threshold is met */
            node.isLeaf = true;
            
            // Set the mean value as the predicted value of a leaf node
            Double mean = 0.0;
            for (String yvalue : y)
            {
                mean += Double.valueOf(yvalue);
            }
            mean /= y.size();
            node.name = mean.toString();
        }
        else
        { 
            node.column = index;
            node.value = value;
            node.categories = categories;
            node.categorical = categorical[index];
            if (node.categorical)
            {
                for (int i = 0; i < node.categories.size(); i++)
                {
                    ArrayList<String[]> Xi = 
                            (ArrayList<String[]>) params.get(i*2 + 3);
                    ArrayList<String> yi = 
                            (ArrayList<String>) params.get(i*2 + 4);
                    node.children.add(buildTree(Xi, yi));
                }
            }
            else
            {
                ArrayList<String[]> X1 = (ArrayList<String[]>) params.get(3);
                ArrayList<String> y1 = (ArrayList<String>) params.get(4);
                ArrayList<String[]> X2 = (ArrayList<String[]>) params.get(5);
                ArrayList<String> y2 = (ArrayList<String>) params.get(6);
                node.children.add(buildTree(X1, y1));
                node.children.add(buildTree(X2, y2));
            }
        }

        return node;
    }
    
    /**
     * This method selects the feature and value to split on. MSE is used as 
     * the splitting criterion
     * 
     * @param @param X is the attributes of the data points at current node
     * @param y is the targets of the data points at current node
     * @return an array of the feature, value, and partitions after the split
     */
    public ArrayList<Object> chooseSplitIndex(ArrayList<String[]> X, 
            ArrayList<String> y)
    {
        ArrayList<Object> params = new ArrayList<Object>();
        int index = -1;
        double value = 0;
        ArrayList<String> categories = new ArrayList<String>();
        
        double mse = Double.MAX_VALUE;
        
        for (int i = 0; i < X.get(0).length; i++)
        {
            ArrayList<String> feature = new ArrayList<String>();
            for (int j = 0; j < X.size(); j++)
            {
                feature.add(X.get(j)[i]);
            } 
            ArrayList<String> uniqueValues = diffValues(feature);
            if (uniqueValues.size() == 1)
            {
                continue;
            }
            
            if (!categorical[i] && uniqueValues.size() > 15)
            {// Use the k-tile method to select split threshold for continuous
                ArrayList<String> temp = new ArrayList<String>();
                Collections.sort(uniqueValues);
                for (int j = 1; j < 16; j++)
                {// Select 15 points evenly from the attribute to be candidates
                    temp.add(uniqueValues.get(j / 16));
                }
                uniqueValues = temp;
            }
            
            if (categorical[i])
            {
                for (String uniqueValue : uniqueValues)
                {
                    ArrayList<String[]> X1 = new ArrayList<String[]>();
                    ArrayList<String> y1 = new ArrayList<String>();
                    ArrayList<String[]> X2 = new ArrayList<String[]>();
                    ArrayList<String> y2 = new ArrayList<String>();
                    
                    for (int j = 0; j < feature.size(); j++)
                    {
                        if (!feature.get(j).equals(uniqueValue))
                        {
                            X1.add(X.get(j));
                            y1.add(y.get(j));
                        }
                        else
                        {
                            X2.add(X.get(j));
                            y2.add(y.get(j));
                        }
                    }
                    ArrayList<ArrayList<String>> yi_s = 
                            new ArrayList<ArrayList<String>>();
                    yi_s.add(y1);
                    yi_s.add(y2);
                    
                    // Update the parameters when MSE is improving
                    double newMse = totalMSE(yi_s);
                    if (newMse < mse && newMse < MSE(y))
                    {
                        params.clear();
                        index = i;
                        params.add(index);
                        params.add(value);
                        params.add(categories);
                        params.add(X1);
                        params.add(y1);
                        params.add(X2);
                        params.add(y2);
                        mse = newMse;
                    }
                }
            }
            else
            {
                for (String uniqueValue : uniqueValues)
                {
                    ArrayList<String[]> X1 = new ArrayList<String[]>();
                    ArrayList<String> y1 = new ArrayList<String>();
                    ArrayList<String[]> X2 = new ArrayList<String[]>();
                    ArrayList<String> y2 = new ArrayList<String>();
                    
                    for (int j = 0; j < feature.size(); j++)
                    {
                        if (Double.parseDouble(feature.get(j)) < 
                                Double.parseDouble(uniqueValue))
                        {
                            X1.add(X.get(j));
                            y1.add(y.get(j));
                        }
                        else
                        {
                            X2.add(X.get(j));
                            y2.add(y.get(j));
                        }
                    }
                    ArrayList<ArrayList<String>> yi_s = 
                            new ArrayList<ArrayList<String>>();
                    yi_s.add(y1);
                    yi_s.add(y2);
                    
                    // Update the parameters when MSE is improving
                    double newMse = totalMSE(yi_s);
                    if (newMse < mse && newMse < MSE(y))
                    {
                        params.clear();
                        index = i;
                        value = Double.parseDouble(uniqueValue);
                        params.add(index);
                        params.add(value);
                        params.add(categories);
                        params.add(X1);
                        params.add(y1);
                        params.add(X2);
                        params.add(y2);
                        mse = newMse;
                    }
                }
            }
        }
        
        return params;
    }
    
    /**
     * This method finds all unique values of an array
     * 
     * @param y is the array of interest
     * @return a new array without redundant values
     */
    public ArrayList<String> diffValues(ArrayList<String> y)
    {
        ArrayList<String> diffValues = new ArrayList<String>();

        for(int i = 0; i < y.size(); i++)
        {
            if(!diffValues.contains(y.get(i)))
            {
                diffValues.add(y.get(i));
            }
        }

       return diffValues;
    }
    
    /**
     * This method calculates the mean squared error of a node
     * 
     * @param y is the node of interest
     * @return the MSE of the node
     */
    public double MSE(ArrayList<String> y)
    {
        // Calculate mean
        double mean = 0;
        int n = y.size();
        if (n == 0)
        {
            return 0;
        }
        for (String value : y)
        {
            mean += Double.parseDouble(value);
        }
        mean /= n;
        
        // Calcuate mean squared error
        double mse = 0;
        for (String value : y)
        {
            mse += Math.pow((Double.parseDouble(value) - mean), 2);
        }
        mse /= n;
        
        return mse;
    }
    
    /**
     * This method calculates the weighted MSE of two or more child nodes
     * 
     * @param yi_s is the list of child nodes
     * @return the weighted, total MSE of the child nodes
     */
    public double totalMSE(ArrayList<ArrayList<String>> yi_s)
    {
        double totalMse = 0;
        int n = 0;
        for (ArrayList<String> yi : yi_s)
        {
            totalMse += MSE(yi) * yi.size();
            n += yi.size();
        }
        if (n == 0)
        {
            return 0;
        }
        totalMse /= n;
        
        return totalMse;
    }
    
    /**
     * This method calls the predictOne() method at the root node of the 
     * trained tree for each test data point in the test set
     * 
     * @param X is the test set
     * @return an array of predictions for all test data points
     */
    public ArrayList<String> predict(ArrayList<String[]> X)
    {
        ArrayList<String> yPredict = new ArrayList<String>();
        for (String[] x : X)
        {
            yPredict.add(root.predictOne(x));
        }
        
        return yPredict;
    }
    
    /**
     * This method performs checks whether the MSE of a node has reached the
     * cut-off threshold predetermined. If so, early stopping is performed
     * 
     * @param y is the target values in the node of interest
     * @return true to indicate that early stopping should take place
     */
    public boolean earlyStopping(ArrayList<String> y)
    {
       if (errorThreshold > MSE(y))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}
