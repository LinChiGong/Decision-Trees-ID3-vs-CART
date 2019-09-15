/**
 * This class runs the full project. 7 datasets are used in this project - 3
 * for classification and 4 for regression. Classification tasks are performed
 * by the DecisionTreeClassifier class which implements the ID3 algorithm. Both
 * unpruned and pruned (by reduced error pruning) classifiers are built so that
 * their performances can be compared. Regression tasks are performed by the 
 * DecisionTreeRegressor class which implements the CART algorithm. The early
 * stopping threshold is tuned for each regression task. The best threshold
 * along with its performance is recorded and compared with that of the regular 
 * regressor (without early stopping). Note that a 5-fold cross validation is 
 * performed for each task, so it is the average performance that is recorded.
 * All results are written to the output file called "Results.txt"
 * 
 * @author Winston Lin
 */
import java.io.*;
import java.util.*;

public class WriteToFile 
{
    public static void main(String[] args) throws IOException 
    {
        // 7 datasets used in this project
        String[] datasets = {"abalone.data", "car.data", 
                "segmentation.data", "machine.data", "forestfires.data", 
                "winequality-white.csv", "winequality-red.csv"};
        // Write to the output file
        PrintWriter fout = new PrintWriter(
                new BufferedWriter(new FileWriter("Results.txt", true)));
        fout.println("Perform classification and regression tasks. For "
                + "classfication, test datasets with Decision Tree "
                + "Algorithm ID3. Performance is measured by classification "
                + "accuracy. For regression, test datasets with Decision Tree "
                + "Algorithm CART. Performance is measured by MSE.");
        fout.println();
        fout.println(" ---------------");
        fout.println("|CLASSIFICATION|");
        fout.println(" ---------------");
        fout.println();
        
        for (int d = 0; d < datasets.length; d++)
        {
            // Write to the output file
            fout.println(datasets[d].substring(datasets[d].lastIndexOf('/')+1,
                    datasets[d].lastIndexOf('.')));
            fout.println("----------------------");
            
            if (d < 3) // Classification
            {
                // Process and split data into train, test, and validation set
                ArrayList<String[]> records = new ArrayList<String[]>();
                ArrayList<ArrayList<String[]>> partitions = 
                        new ArrayList<ArrayList<String[]>>();
                ETL etl = new ETL();
                records = etl.readCSV(datasets[d]);
                partitions = etl.split(records, true);   
                
                // Perform 5-fold cross validation for classification using ID3
                for (int p = 0; p < 2; p++)
                {
                    double[] accuracies = new double[5];
                    for (int k = 0; k < 5; k++)
                    {
                        ArrayList<String[]> Xtrain = new ArrayList<String[]>();
                        ArrayList<String> ytrain = new ArrayList<String>();
                        ArrayList<String[]> Xtest = new ArrayList<String[]>();
                        ArrayList<String> ytest = new ArrayList<String>();
                        ArrayList<String[]> Xvalid = new ArrayList<String[]>();
                        ArrayList<String> yvalid = new ArrayList<String>();
                        
                        for (int i = 1; i < partitions.size(); i++)
                        {
                            if (i == k + 1)
                            {
                                for (String[] test : partitions.get(i))
                                {
                                    String[] X = new String[test.length - 1];
                                    for (int j = 0; j < X.length; j++)
                                    {
                                        X[j] = test[j];
                                    }
                                    Xtest.add(X);
                                    ytest.add(test[test.length - 1]);
                                }
                            }
                            else
                            {
                                for (String[] train : partitions.get(i))
                                {
                                    String[] X = new String[train.length - 1];
                                    for (int j = 0; j < X.length; j++)
                                    {
                                        X[j] = train[j];
                                    }
                                    Xtrain.add(X);
                                    ytrain.add(train[train.length - 1]);
                                }
                            }
                        }  
                        for (String[] valid : partitions.get(0))
                        {
                            String[] X = new String[valid.length - 1];
                            for (int j = 0; j < X.length; j++)
                            {
                                X[j] = valid[j];
                            }
                            Xvalid.add(X);
                            yvalid.add(valid[valid.length - 1]);
                        }
                        
                        // Fit on train, prune on validation, predict on test
                        DecisionTreeClassifier dtc = 
                                new DecisionTreeClassifier();
                        dtc.fit(Xtrain, ytrain, etl.fileName);
                        if (p == 1)
                        {
                            dtc.prune(Xvalid, yvalid, dtc.root);
                        }
                        ArrayList<String> prediction = dtc.predict(Xtest);
                        double accuracy = 0;
                        for (int i = 0; i < ytest.size(); i++)
                        {  
                            if (ytest.get(i).equals(prediction.get(i)))
                            {
                                accuracy++;
                            }
                        }
                        accuracy /= ytest.size();
                        accuracies[k] = accuracy;
                    }
                    
                    double averageAccuracy = 0;
                    for (double accuracy : accuracies)
                    {
                        averageAccuracy += accuracy;
                    }
                    averageAccuracy /= 5;
                    if (p == 0)
                    {
                        fout.println("Average accuracy (unpruned): " 
                                + Math.round(averageAccuracy * 10000.0) / 100.0 
                                + "%");
                    }
                    else
                    {
                        fout.println("Average accuracy (pruned): " 
                                + Math.round(averageAccuracy * 10000.0) / 100.0 
                                + "%");
                    }
                    fout.println();
                }
                if (d == 2)
                {
                    fout.println(" -----------");
                    fout.println("|REGRESSION|");
                    fout.println(" -----------");
                    fout.println();
                }
            }
            else // Regression
            {
                // Process and split data into train, test, and validation set
                ArrayList<String[]> records = new ArrayList<String[]>();
                ArrayList<ArrayList<String[]>> partitions = 
                        new ArrayList<ArrayList<String[]>>();
                ETL etl = new ETL();
                records = etl.readCSV(datasets[d]);
                partitions = etl.split(records, false);  
                
                // Tune the early stopping threshold for regression using CART
                double[] thresholds = {0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 
                                       1, 5, 10, 50, 100, 500, 1000, 5000, 
                                       10000, 50000};
                ArrayList<Double> averageMSEs = new ArrayList<Double>();
                for (double threshold : thresholds)
                {
                    double[] MSEs = new double[5];
                    for (int k = 0; k < 5; k++)
                    {
                        ArrayList<String[]> Xtrain = new ArrayList<String[]>();
                        ArrayList<String> ytrain = new ArrayList<String>();
                        ArrayList<String[]> Xtest = new ArrayList<String[]>();
                        ArrayList<String> ytest = new ArrayList<String>();
                        ArrayList<String[]> Xvalid = new ArrayList<String[]>();
                        ArrayList<String> yvalid = new ArrayList<String>();
                        
                        for (int i = 1; i < partitions.size(); i++)
                        {
                            if (i == k + 1)
                            {
                                for (String[] test : partitions.get(i))
                                {
                                    String[] X = new String[test.length - 1];
                                    for (int j = 0; j < X.length; j++)
                                    {
                                        X[j] = test[j];
                                    }
                                    Xtest.add(X);
                                    ytest.add(test[test.length - 1]);
                                }
                            }
                            else
                            {
                                for (String[] train : partitions.get(i))
                                {
                                    String[] X = new String[train.length - 1];
                                    for (int j = 0; j < X.length; j++)
                                    {
                                        X[j] = train[j];
                                    }
                                    Xtrain.add(X);
                                    ytrain.add(train[train.length - 1]);
                                }
                            }
                        }  
                        for (String[] valid : partitions.get(0))
                        {
                            String[] X = new String[valid.length - 1];
                            for (int j = 0; j < X.length; j++)
                            {
                                X[j] = valid[j];
                            }
                            Xvalid.add(X);
                            yvalid.add(valid[valid.length - 1]);
                        }
                        
                        // Fit on train, tune the threshold on validation
                        DecisionTreeRegressor dtr = 
                                new DecisionTreeRegressor(threshold);
                        dtr.fit(Xtrain, ytrain, etl.fileName);
                        ArrayList<String> prediction = dtr.predict(Xvalid);
                        double MSE = 0;
                        for (int i = 0; i < yvalid.size(); i++)
                        {
                            MSE += Math.pow((Double.parseDouble(yvalid.get(i)) 
                                   - Double.parseDouble(prediction.get(i))), 2)
                                   / ytest.size();
                        }
                        MSEs[k] = MSE;
                    }
                    double averageMSE = 0;
                    for (double MSE : MSEs)
                    {
                        averageMSE += MSE;
                    }
                    averageMSE /= 5;
                    averageMSEs.add(averageMSE);
                }
                
                int bestThresholdIndex = 0;
                double bestMSE = Double.MAX_VALUE;
                for (int i = 0; i < averageMSEs.size(); i++)
                {
                    if (averageMSEs.get(i) < bestMSE)
                    {
                        bestThresholdIndex = i;
                        bestMSE = averageMSEs.get(i);
                    }
                }
                
                // Use the best threshold to make predictions on the test set
                double[] unprunedMSEs = new double[5];
                double[] prunedMSEs = new double[5];
                for (int k = 0; k < 5; k++)
                {
                    ArrayList<String[]> Xtrain = new ArrayList<String[]>();
                    ArrayList<String> ytrain = new ArrayList<String>();
                    ArrayList<String[]> Xtest = new ArrayList<String[]>();
                    ArrayList<String> ytest = new ArrayList<String>();
                    
                    for (int i = 1; i < partitions.size(); i++)
                    {
                        if (i == k + 1)
                        {
                            for (String[] test : partitions.get(i))
                            {
                                String[] X = new String[test.length - 1];
                                for (int j = 0; j < X.length; j++)
                                {
                                    X[j] = test[j];
                                }
                                Xtest.add(X);
                                ytest.add(test[test.length - 1]);
                            }
                        }
                        else
                        {
                            for (String[] train : partitions.get(i))
                            {
                                String[] X = new String[train.length - 1];
                                for (int j = 0; j < X.length; j++)
                                {
                                    X[j] = train[j];
                                }
                                Xtrain.add(X);
                                ytrain.add(train[train.length - 1]);
                            }
                        }
                    }
                    
                    // Fit on train without early stopping, predict on test
                    DecisionTreeRegressor dtr = 
                            new DecisionTreeRegressor(0);
                    dtr.fit(Xtrain, ytrain, etl.fileName);
                    ArrayList<String> prediction = dtr.predict(Xtest);
                    double MSE = 0;
                    for (int i = 0; i < ytest.size(); i++)
                    {
                        MSE += Math.pow((Double.parseDouble(ytest.get(i)) 
                                - Double.parseDouble(prediction.get(i))), 2)
                                / ytest.size();
                    }
                    unprunedMSEs[k] = MSE;
                    
                    // Fit on train, use the best threshold, predict on test
                    dtr = new DecisionTreeRegressor(
                            thresholds[bestThresholdIndex]);
                    dtr.fit(Xtrain, ytrain, etl.fileName);
                    prediction = dtr.predict(Xtest);
                    MSE = 0;
                    for (int i = 0; i < ytest.size(); i++)
                    {
                        MSE += Math.pow((Double.parseDouble(ytest.get(i)) 
                                - Double.parseDouble(prediction.get(i))), 2) 
                                / ytest.size();
                    }
                    prunedMSEs[k] = MSE;
                }
                
                // Without early stopping
                double averageUnprunedMSE = 0;
                for (double unprunedMSE : unprunedMSEs)
                {
                    averageUnprunedMSE += unprunedMSE;
                }
                averageUnprunedMSE /= 5;
                fout.println("Average MSE (without early stopping): " 
                            + Math.round(averageUnprunedMSE * 100.0) / 100.0);
                
                // With early stopping at the best threshold
                double averagePrunedMSE = 0;
                for (double prunedMSE : prunedMSEs)
                {
                    averagePrunedMSE += prunedMSE;
                }
                averagePrunedMSE /= 5;
                fout.println("Best threshold: " 
                            + thresholds[bestThresholdIndex]);
                fout.println("Average MSE (with early stopping): " 
                            + Math.round(averagePrunedMSE * 100.0) / 100.0);
                fout.println();
            }
        }
        fout.close();
    }
}
