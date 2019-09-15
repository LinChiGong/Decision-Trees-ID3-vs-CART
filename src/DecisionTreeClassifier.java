/**
 * This class implements a Decision Tree Classifier using the ID3 algorithm.
 * The splitting criterion used in this project is gain ratio. The prune()
 * method can be called to perform reduced error pruning which adopts 
 * classification error as the loss function
 * 
 * @author Winston Lin
 */
import java.io.*;
import java.util.*;

public class DecisionTreeClassifier 
{
    Node root; // Root node
    boolean[] categorical; // Boolean array of whether feature is categorical
    
    /**
     * This method fits the classifier on the training set. It determines
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
        categorical = new boolean[X.get(0).length];
        if (fileName.equals("abalone"))
        {// The first feature in the abalone dataset are categorical
            categorical[0] = true;
        }
        else if (fileName.equals("car"))
        {// All features in the car dataset are categorical
            for (int i = 0; i < categorical.length; i++)
            {
                categorical[i] = true;
            }
        }
        
        root = buildTree(X, y);
    }
    
    /**
     * This method calls the chooseSplitIndex() method to choose the feature
     * and value at each split. It then uses those information to build the
     * tree recursively until we run out of features or impurity = 0
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
        
        if (index == -1 || diffValues(y).size() == 1)
        {// Two conditions for leaf: 1. Run out of features 2. Impurity = 0
            node.isLeaf = true;
            for (int i = 0; i < y.size(); i++)
            {
                if (node.classCount.containsKey(y.get(i)))
                {
                    int count = node.classCount.get(y.get(i));
                    count++;
                    node.classCount.put(y.get(i), count);
                }
                else
                {
                    node.classCount.put(y.get(i), 1);
                }
            }
            
            // Set the most common class as the predicted class of a leaf node
            String mostCommon = "";
            int max = -1;
            Iterator<Map.Entry<String, Integer>> it = 
                    node.classCount.entrySet().iterator();
            while(it.hasNext())
            {
                Map.Entry<String, Integer> pair = it.next();
                if (pair.getValue() > max)
                {
                    mostCommon = pair.getKey();
                    max = pair.getValue();
                }
            }
            node.name = mostCommon;
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
     * This method selects the feature and value to split on. Gain ratio is
     * used as the splitting criterion
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
        
        double gainRatio = 0;
        
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
                ArrayList<ArrayList<String[]>> Xi_s = 
                        new ArrayList<ArrayList<String[]>>();
                ArrayList<ArrayList<String>> yi_s = 
                        new ArrayList<ArrayList<String>>();
                for (String uniqueValue : uniqueValues)
                {// For categorical features, each category has its own branch
                    ArrayList<String[]> Xi = new ArrayList<String[]>();
                    ArrayList<String> yi = new ArrayList<String>();
                    for (int j = 0; j < feature.size(); j++)
                    {
                        if (feature.get(j).equals(uniqueValue))
                        {
                            Xi.add(X.get(j));
                            yi.add(y.get(j));
                        }
                    }
                    Xi_s.add(Xi);
                    yi_s.add(yi);
                }
                
                // Update the parameters when we have higher gain ratio
                double newGainRatio = gainRatio(y, yi_s);                
                if (newGainRatio > gainRatio)
                {
                    params.clear();
                    index = i;
                    categories = uniqueValues;
                    params.add(index);
                    params.add(value);
                    params.add(categories);
                    for (int j = 0; j < Xi_s.size(); j++)
                    {
                        params.add(Xi_s.get(j));
                        params.add(yi_s.get(j));
                    }
                    gainRatio = newGainRatio;
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
                    
                    // Update the parameters when we have higher gain ratio
                    double newGainRatio = gainRatio(y, yi_s);
                    if (newGainRatio > gainRatio)
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
                        gainRatio = newGainRatio;
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
     * This method calculates the gain ratio of a split of a parent node into 
     * two or more child nodes
     * 
     * @param y is the parent node
     * @param yi_s is the list of child nodes
     * @return the gain ratio of the split
     */
    public double gainRatio(ArrayList<String> y, 
            ArrayList<ArrayList<String>> yi_s)
    {
        double informationGain = 0;
        double intrinsicInfo = 0;
        
        // Calculate information gain
        int n = y.size();
        double childInfo = 0;
        for (ArrayList<String> yi : yi_s)
        {
            childInfo += entropy(yi) * yi.size() / n;
        }
        informationGain = entropy(y) - childInfo;        
        
        // Calculate intrinsic information
        double branchInfo = 0;
        for (ArrayList<String> yi : yi_s)
        {
            branchInfo += 
                    yi.size()*1.0/n * Math.log(yi.size()*1.0/n) / Math.log(2);
        }
        intrinsicInfo = -branchInfo;
        
        return informationGain / intrinsicInfo;
    }
    
    /**
     * This method calculates the entropy of a node
     * 
     * @param y is the target values in the node of interest
     * @return the entropy of the node
     */
    public double entropy(ArrayList<String> y)
    {
        int n = y.size();
        double summation = 0;
        for (String class_ : diffValues(y))
        {
            double prob = 0;
            for (String value : y)
            {
                if (value.equals(class_))
                {
                    prob++;
                }
            }
            prob /= n;
            summation += prob * Math.log(prob) / Math.log(2);
        }
        
        return -summation;
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
     * This method can be called to perform a reduced error pruning after the
     * tree is fully grown. It compares classification error before and after
     * merging leaves. Merging takes place whenever it improves classification
     * error
     * 
     * @param X is the attributes of the validation set used only for pruning
     * @param y is the targets of the validation set used only for pruning
     * @param node is the root node of the trained tree
     */
    public void prune(ArrayList<String[]> X, ArrayList<String> y, Node node)
    {
        for (Node child : node.children)
        {
            if (!child.isLeaf)
            {
                prune(X, y, child);
            }
        }
        
        boolean allLeaf = true;
        for (Node child : node.children)
        {
            if (!child.isLeaf)
            {
                allLeaf = false;
            }
        }
        
        // Compare classification error before and after merging the leaves
        if (allLeaf)
        {
            // Make predictions with the current tree
            ArrayList<String> leafY = predict(X);
            
            // Merge the leaves and make predictions with the new, merged leaf
            HashMap<String, Integer> mergedClassCount = 
                    new HashMap<String, Integer>();
            for (Node child : node.children)
            {
                Iterator<Map.Entry<String, Integer>> it = 
                        child.classCount.entrySet().iterator();
                while(it.hasNext())
                {
                    Map.Entry<String, Integer> pair = it.next();
                    if (mergedClassCount.containsKey(pair.getKey()))
                    {
                        int count = mergedClassCount.get(pair.getKey());
                        count++;
                        mergedClassCount.put(pair.getKey(), count);
                    }
                    else
                    {
                        mergedClassCount.put(pair.getKey(), 1);
                    }
                }
            }
            String mostCommon = "";
            int max = -1;
            Iterator<Map.Entry<String, Integer>> it = 
                    mergedClassCount.entrySet().iterator();
            while(it.hasNext())
            {
                Map.Entry<String, Integer> pair = it.next();
                if (pair.getValue() > max)
                {
                    mostCommon = pair.getKey();
                    max = pair.getValue();
                }
            }
            
            // Make predictions with the newly pruned tree
            node.isTempLeaf = true;
            node.classCount = mergedClassCount;
            node.name = mostCommon;
            ArrayList<String> mergedY = predict(X);
            
            // Calculate classification accuracies for both cases
            double leafScore = 0;
            double mergedScore = 0;
            for (int i = 0; i < y.size(); i++)
            {
                if (leafY.get(i).equals(y.get(i)))
                {
                    leafScore++;
                }
                if (mergedY.get(i).equals(y.get(i)))
                {
                    mergedScore++;
                }
            }
            leafScore /= y.size();
            mergedScore /= y.size();
            
            // If merging improves classification accuracy, accept the merging
            if (mergedScore > leafScore)
            {
                // For demonstration purpose
                System.out.println("Merging...");
                
                node.isLeaf = true;
                node.children.clear();
            }
            else
            {
                node.classCount.clear();
            }
            
            node.isTempLeaf = false;
        }
    }
}
