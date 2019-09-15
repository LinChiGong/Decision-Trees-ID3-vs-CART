/**
 * This class implements a tree node that is used by the DecisionTreeClassifier
 * class and the DecisionTreeRegressor class. The only method in this class is
 * called during prediction
 * 
 * @author Winston Lin
 */
import java.io.*;
import java.util.*;

public class Node 
{
    int column;   // Index of feature to split on
    double value; // Value of a continuous feature to split on
    
    ArrayList<String> categories = new ArrayList<String>(); // Discrete values
    boolean categorical = false; // Whether node splits on categorical feature
    String name;  // Name of the most common class label or the mean value
    ArrayList<Node> children = new ArrayList<Node>(); // All child nodes
    boolean isLeaf = false; // True if the node is a leaf
    boolean isTempLeaf = false; // Temporary leaf used during pruning
    HashMap<String, Integer> classCount = 
            new HashMap<String, Integer>(); /* Count each class in preparation
                                               for prediction and pruning */
    
    /**
     * This method makes prediction on a single data point. The subtree rooted
     * at the node is traversed until reaching a leaf. The "name" of the leaf
     * node is the predicted value
     * 
     * @param x is a single data point
     * @return the predicted value
     */
    public String predictOne(String[] x)
    {
        if (isTempLeaf)
        {// Used in pruning. Temporary leaf has higher priority than real leaf
            return name;
        }
        else if (isLeaf)
        {
            return name;
        }
        
        String colValue = x[column];
        if (categorical)
        {
            boolean hasBranch = false;
            for (int i = 0; i < categories.size(); i++)
            {
                if (colValue.equals(categories.get(i)))
                {
                    hasBranch = true;
                    return children.get(i).predictOne(x);   
                }
            }
            
            // Handle case when the category has no corresponding branch
            if (!hasBranch)
            {
                int randomIndex = (int) (Math.random() * children.size());
                return children.get(randomIndex).predictOne(x);
            }
        }
        else
        {
            if (Double.parseDouble(colValue) < value)
            {
                return children.get(0).predictOne(x);
            }
            else
            {
                return children.get(1).predictOne(x);
            }
        }
        
        return "This should not be reached";
    }
}
