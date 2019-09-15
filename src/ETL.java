/**
 * This class reads data from file and preprocesses the data in preparation for
 * being fed to either an ID3 algorithm implemented by DecisionTreeClassifier
 * class or a CART algorithm implemented by the DecisionTreeRegressor class.
 * The method split() also enables cross validation
 * 
 * @author Winston Lin
 */
import java.io.*;
import java.util.*;

public class ETL 
{    
    String fileName; // Name of the file, extracted from file path
    HashMap<String, Integer> classCount = 
            new HashMap<String, Integer>(); /* Count each class in preparation
                                               for stratified split */
    
    /**
     * This method reads the 7 datasets used in this project, performs
     * appropriate processing, and stores each dataset in a 2D String array
     * 
     * @param filePath is the path of the data file
     * @return the 2D String array that stores the dataset
     * @throws FileNotFoundException
     */
    public ArrayList<String[]> readCSV(String filePath) 
            throws FileNotFoundException
    {
        fileName = filePath.substring(filePath.lastIndexOf('/') + 1, 
                filePath.lastIndexOf('.'));
        
        ArrayList<String[]> records = new ArrayList<String[]>();
        String[] record;
        File file = new File(filePath);
        Scanner sc = new Scanner(file);
        while (sc.hasNextLine())
        {
            if (fileName.contains("wine"))
            {
                record = sc.nextLine().split(";", -1);
            }
            else
            {
                record = sc.nextLine().split(",", -1);
            }
            
            if (fileName.equals("segmentation"))
            {
                for (int i = 0; i < record.length - 1; i++)
                {// Move the target column to the last position
                    String temp = record[i];
                    record[i] = record[i + 1];
                    record[i + 1] = temp;
                }
                records.add(record);
            }
            else if (fileName.equals("machine"))
            {
                String[] reducedRecord = new String[record.length - 3];
                for (int i = 0; i < reducedRecord.length; i++)
                {// Ignore columns "Vendor Name", "Model Name", and "ERP"
                    reducedRecord[i] = record[i + 2];
                }
                records.add(reducedRecord);
            }
            else if (fileName.equals("forestfires"))
            {
                String[] month = {"jan", "feb", "mar", "apr", "may", "jun", 
                        "jul", "aug", "sep", "oct", "nov", "dec"};
                String[] day = {"mon", "tue", "wed", "thu", "fri", "sat", 
                        "sun"};
                // Represent columns "month" and "day" as Roman numerals
                for (int i = 0; i < month.length; i++)
                {
                    if (record[2].equals(month[i]))
                    {
                        record[2] = Integer.toString(i + 1);
                    }
                }
                for (int i = 0; i < day.length; i++)
                {
                    if (record[3].equals(day[i]))
                    {
                        record[3] = Integer.toString(i + 1);
                    }
                }
                records.add(record);
            }
            else
            {
                records.add(record);
            }
        }
        
        // Remove headers
        if (fileName.equals("segmentation"))
        {
            for (int i = 0; i < 5; i++)
            {
                records.remove(0);
            }   
        }
        else if (fileName.equals("forestfires") || fileName.contains("wine"))
        {     
            records.remove(0);
        }
        
        sc.close();
        
        return records;
    } 
    
    /**
     * This method splits the data array into 6 partitions. It first pull out
     * 10% of the data which will be used as a validation set (for pruning). It
     * then splits the rest of the data into 5 partitions used in cross 
     * validation 
     * 
     * @param records is the data array
     * @param stratified can be turned on to perform stratified split
     * @return the 6 partitions in an array
     */
    public ArrayList<ArrayList<String[]>> split(ArrayList<String[]> records,
            boolean stratified)
    {
        ArrayList<ArrayList<String[]>> partitions = 
                new ArrayList<ArrayList<String[]>>();
        
        // Pull out 10% of the data to be used as a validation set
        int validSize = records.size() / 10;
        ArrayList<String[]> validationSet = new ArrayList<String[]>();
        for (int i = 0; i < validSize; i++)
        {
            int randomIndex = (int) (Math.random() * records.size());
            validationSet.add(records.get(randomIndex));
            records.remove(randomIndex);
        }
        partitions.add(validationSet);
        
        // Count the number of instances in each class
        if (stratified)
        {
            for (int i = 0; i < records.size(); i++)
            {
                if (classCount.containsKey(
                        records.get(i)[records.get(0).length - 1]))
                {
                    int count = classCount.get(
                            records.get(i)[records.get(0).length - 1]);
                    count++;
                    classCount.put(
                            records.get(i)[records.get(0).length - 1], count);
                }
                else
                {
                    classCount.put(
                            records.get(i)[records.get(0).length - 1], 1);
                }
            }
        }
        
        // Split the data into 5 folds in preparation for cross validation
        int foldSize = records.size() / 5;
        for (int i = 0; i < 5; i++)
        {
            ArrayList<String[]> fold = new ArrayList<String[]>();
            
            if (stratified)
            {// Perform stratified split
                Iterator<Map.Entry<String, Integer>> it = 
                        classCount.entrySet().iterator();
                while(it.hasNext())
                {// For each class, select enough points to current fold
                    Map.Entry<String, Integer> pair = it.next();
                    int classPoints = pair.getValue();
                    if (i == 4)
                    {// Add all remaining data points to the last fold
                        while (records.size() != 0)
                        {
                            fold.add(records.get(0));
                            records.remove(0);
                        }
                        break;
                    }
                    // Make sure we have enough data points in each fold
                    int minPoints = classPoints / 5;
                    if (minPoints < 1)
                    {
                        minPoints = 1;
                    }
                    int j = 0;
                    while(j < records.size() && minPoints > 0)
                    {
                        if (pair.getKey().equals(
                                records.get(j)[records.get(0).length - 1]))
                        {
                            fold.add(records.get(j));
                            records.remove(j);
                            minPoints--;
                        }
                        j++;
                    }
                }
            }
            else
            {// Perform random split
                if (i == 4)
                {// Add all remaining data points to the last fold
                    while (records.size() != 0)
                    {
                        fold.add(records.get(0));
                        records.remove(0);
                    }
                }
                else
                {// Randomly select data points to add to the current fold
                    while(fold.size() < foldSize)
                    {
                        int randomIndex = (int) (Math.random()*records.size());
                        fold.add(records.get(randomIndex));
                        records.remove(randomIndex);
                    }
                }
            }
            partitions.add(fold);
        }
        return partitions;
    }
}
