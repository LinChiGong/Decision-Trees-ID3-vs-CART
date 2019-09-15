/**
 * This class performs one classification task and one regression task using
 * the input files and prints the process and results to the console. Note that
 * this class serves the demonstration purpose. For a complete run on all 7 
 * datasets used in this project, run the WriteToFile class
 * 
 * @author Winston Lin
 */
import java.io.*;
import java.util.*;

public class Main 
{
    public static void main(String[] args) throws IOException
    {
        String classificationFile;
        
        // Prompt for the dataset to be used
        Scanner scan = new Scanner(System.in);
        System.out.println();
        System.out.print("Enter input file name for classification: ");
        classificationFile = scan.nextLine().trim();
        System.out.println();
        
        // Preprocess and split the data into train, test, and validation set
        ArrayList<String[]> records = new ArrayList<String[]>();
        ArrayList<ArrayList<String[]>> partitions = 
                new ArrayList<ArrayList<String[]>>();
        ETL etl = new ETL();
        records = etl.readCSV(classificationFile);
        partitions = etl.split(records, true);
        System.out.println("Dataset has been processed and splited");
        System.out.println();
        
        // Checkpoint
        System.out.print("Press 'Enter' to run the ID3 Algorithm:");
        String temp = scan.nextLine().trim();
        System.out.println();
        
        // Perform 5-fold cross validation of ID3 for classification
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
                
                DecisionTreeClassifier dtc = new DecisionTreeClassifier();
                dtc.fit(Xtrain, ytrain, etl.fileName);
                if (p == 1)
                {
                    dtc.prune(Xvalid, yvalid, dtc.root);
                }
                ArrayList<String> prediction = dtc.predict(Xtest);
                double accuracy = 0;
                if (k == 0 && p == 0)
                {
                    System.out.println("\tACTUAL" + "\tPREDICTED");
                    System.out.println("\t------" + "\t---------");
                }
                for (int i = 0; i < ytest.size(); i++)
                {
                    
                    if (k == 0 && p == 0)
                    {// Print out the prediction for each test instance
                        System.out.println("\t" + ytest.get(i) + "\t| " 
                                                + prediction.get(i));
                    }
                    if (ytest.get(i).equals(prediction.get(i)))
                    {
                        accuracy++;
                    }
                }
                accuracy /= ytest.size();
                accuracies[k] = accuracy;
                if (k == 0 && p == 0)
                {
                    System.out.println();
                }
                System.out.println("Accuracy: " 
                            + Math.round(accuracy * 10000.0) / 100.0 + "%");
                System.out.println("---------------------");
            }
            double averageAccuracy = 0;
            for (double accuracy : accuracies)
            {
                averageAccuracy += accuracy;
            }
            averageAccuracy /= 5;
            if (p == 0)
            {
                System.out.println("Average accuracy (unpruned): " 
                        + Math.round(averageAccuracy * 10000.0) / 100.0 + "%");
            }
            else
            {
                System.out.println("Average accuracy (pruned): " 
                        + Math.round(averageAccuracy * 10000.0) / 100.0 + "%");
            }
            System.out.println();
        }
        
        String regressionFile;
        
        // Prompt for the dataset to be used
        scan = new Scanner(System.in);
        System.out.print("Enter input file name for regression: ");
        regressionFile = scan.nextLine().trim();
        System.out.println();
        
        // Preprocess and split the data into train, test, and validation set
        records = new ArrayList<String[]>();
        partitions = new ArrayList<ArrayList<String[]>>();
        etl = new ETL();
        records = etl.readCSV(regressionFile);
        partitions = etl.split(records, false);
        System.out.println("Dataset has been processed and splited");
        System.out.println();
        
        // Checkpoint
        System.out.print("Press 'Enter' to tune the threshold for early "
                + "stopping:");
        temp = scan.nextLine().trim();
        System.out.println();
        
        // Tune the early stopping threshold for regression using CART
        double[] thresholds = {0, 1, 5, 10, 50, 100, 500, 1000, 5000};
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
                DecisionTreeRegressor dtr = 
                        new DecisionTreeRegressor(threshold);
                dtr.fit(Xtrain, ytrain, etl.fileName);
                ArrayList<String> prediction = dtr.predict(Xvalid);
                double MSE = 0;
                for (int i = 0; i < yvalid.size(); i++)
                {
                    MSE += Math.pow((Double.parseDouble(yvalid.get(i)) 
                            - Double.parseDouble(prediction.get(i))), 2)
                            / yvalid.size();
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
            System.out.println("At threshold: " + threshold);
            System.out.println("Average MSE (validation): " + averageMSE);
            System.out.println("--------------------------------");
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
        
        // Checkpoint
        System.out.println();
        System.out.print("Press 'Enter' to run the CART Algorithm with the "
                + "best threshold:");
        temp = scan.nextLine().trim();
        System.out.println();        
        
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
            
            // Print out the prediction for each test instance
            if (k == 0)
            {
                System.out.println("\tACTUAL" + "\tPREDICTED");
                System.out.println("\t------" + "\t---------");
                for (int i = 0; i < ytest.size(); i++)
                {
                    System.out.println("\t" + ytest.get(i) + "\t| " 
                                                + prediction.get(i));
                }
            }
            
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
        System.out.println();
        System.out.println("Average MSE (without early stopping): " 
                    + Math.round(averageUnprunedMSE * 100.0) / 100.0);
        
        // With early stopping at the best threshold
        double averagePrunedMSE = 0;
        for (double prunedMSE : prunedMSEs)
        {
            averagePrunedMSE += prunedMSE;
        }
        averagePrunedMSE /= 5;
        System.out.println("Best threshold: " 
                    + thresholds[bestThresholdIndex]);
        System.out.println("Average MSE (with early stopping): " 
                    + Math.round(averagePrunedMSE * 100.0) / 100.0);
        System.out.println();
        
        scan.close();
    }
}
