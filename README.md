# Decision-Trees-ID3-vs-CART

## Abstract

In this project, two algorithms – Iterative Dichotomiser 3 (ID3) and Classification And Regression Tree (CART) – are experimented to solve 3 classification problems and 4 regression problems respectively. All 7 datasets used in the problems are obtained from the UCI Machine Learning Repository [1]. Both algorithms implement a pruning method that can perform post- or pre-pruning on the trees. The performances of pruned and unpruned trees are recorded and compared. In general, pruned trees perform better than unpruned trees. 

## Introduction

The main problems in this project are classification problems based on 3 datasets – “Abalone”, “Car Evaluation”, and “Image Segmentation” – and regression problems based on 4 datasets – “Computer Hardware”, “Forest Fires”, “Wine Quality - White”, and “Wine Quality - Red”. Of the 3 classification datasets, “Abalone” is the only dataset that has mixed attributes. Others’ are either pure categorical or pure numeric. This should not pose a problem since ID3 is good at handling mixed attributes. Also, “Image Segmentation” contains 7 evenly distributed classes, “Car Evaluation” contains 4 classes that are slightly imbalanced, and “Abalone” contains 28 classes that are highly imbalanced. In fact, 9 classes in “Abalone” have less than 10 instances while 11 classes in “Abalone” have more than 100 instances. This severe imbalance may hurt ID3’s performance. On the other hand, all 4 regression datasets have only numeric attributes.

> ID3 is a supervised, nonparametric Decision Tree algorithm. Similar to CART and other Decision Tree algorithms, ID3 makes no assumptions about the distribution of the data and it predicts the target value or label of a query point by walking the query point through a tree structure constructed based on the training set. An important characteristic of ID3 is that it does not limit the number of branches at each split. This is especially useful for categorical feature that has more than 2 categories. Each category can have its own branch. However, this characteristic also causes information gain to be biased toward highly branching features because the subset of each branch is more likely to be pure. To address this problem, gain ratio is used instead of information gain as the splitting criterion for our ID3 algorithm. Note that gain ratio weights the regular information gain by the inverse of the number of branches. As a result, it reduces the bias on highly branching features.

> In this project, we allow post-pruning to be performed after ID3 constructs a fully grown tree. Specifically, the pruning method used by our ID3 is called reduced error pruning. It is worth noting that a fully grown tree usually has high variance and tends to overfit the training data. Reduced error pruning is one way to reduce overfitting and make the model generalize better. Thus, I expect post-pruned trees to perform better than unpruned trees for all problems.

> CART is similar to ID3 in many ways except that CART only makes binary split. As a result, we don’t have to worry about the number of branches and can simply use mean squared error (MSE) as the splitting criterion. For CART, we allow pre-pruning with early stopping. Unlike reduced error pruning, early stopping needs to be done during the tree building process. Yet similar to reduced error pruning, the effect of early stopping is to make the tree smaller so that it does not overfit the data, resulting in better generalization ability. Thus, I also expect pre-pruned trees to perform better than unpruned trees for all problems. 

While ID3 and CART both work for either classification or regression problems. In this project, we use ID3 only for classification tasks, and use CART only for regression tasks. For classification, the predicted label of a leaf is the most common label of all training data points in the leaf. For regression, the predicted value of a leaf is the mean value of all training data points in the leaf.

## Methods

- Data processing:

  The categorical attributes, “month” and “day” in the “Forest Fires” dataset are represented as Roman numerals. Headers are removed. Columns such as “model name” or “vendor name” that are not informative are dropped. There is no missing value in all datasets.

  Prior to running ID3 and CART, a dataset is partitioned into 6 folds – one fold is used as the validation set and the other 5 folds are used in cross validation. The 5-fold partition can be either regular or stratified. In regular partition, instances are randomly assigned to each fold. In stratified partition, the proportion of different classes in the original training set is maintained. Instances of the same label are assigned to each fold evenly. In this project, we turn on stratified partition for all classification problems.

- ID3 [2]:

  Recall that we use ID3 for classification tasks. During the tree building process, gain ratio is used as the splitting criterion and entropy is used as the impurity measure. Note that categorical features and numeric features are handled separately. For categorical features, one branch is created for each category, so a feature is only considered once. For numeric features, a splitting value needs to be determined in order to create two branches, so a feature is considered multiple times with different splitting values. One way to determine the best splitting value is to go through all values of the data points in the node. However, doing so is very time consuming, so in this project, I implement a k-tile method where k = 15 [3]. I first sort the data points by target values and then select 15 data points that are at evenly spaced positions from each other. In this fashion, the number of candidate splitting values reduces from thousands to 15, and the k-tile method effectively reduces the runtime from 3 hours to under 3 minutes. At each split, the categorical feature or the numeric feature-value pair that leads to the highest gain ratio is selected to make the split.

  For reduced error pruning, we use the classification error as the loss function. When the prune() method is called, we recursively inspect each non-leaf node in a bottom-up fashion. For each non-leaf node, we calculate two versions of overall classification accuracy on the validation set. The first version uses the tree as-is and the second version uses a new tree where the leaves of the non-leaf node are merged and the non-leaf node is now treated as a temporary leaf. If the classification accuracy after merging is better than the classification accuracy before merging, we accept the merging and make the non-leaf node a permanent leaf.

- CART [4]:
  
  Recall that we use CART for regression tasks. Instead of gain ratio, mean squared error is used as the splitting criterion. Different from ID3, CART only makes binary split. Thus, for categorical features, only one category can be selected at a time where a data point can either belong to the category or not. That is, CART handles categorical features and numeric features very similarly. At each split, the categorical feature-category pair or the numeric feature-value pair that leads to the smallest MSE is selected to make the split. Note that the k-tile method is also implemented for numeric features to speed up training. 

  For early stopping, we set a cut-off threshold and use MSE as the loss function. Prior to making a split, the MSE of the current node is calculated. If the MSE is less than the threshold, then the current node is immediately assigned as a leaf and no further splits will be made. To determine a good cut-off threshold, we use the validation set to tune the threshold. Specifically, thresholds [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, ... , 10000, 50000] are considered for all regression tasks.

## Results

- Classification:

  ![alt text](output/classification.png)

  Note that the values are average classification accuracies calculated from performing cross validation on the test sets. Pruned trees are ID3 trees that are post-pruned by reduced error pruning using the validation set.

- Regression:

  ![alt text](output/regression.png)

  Here, the values are mean squared errors. Average MSEs of unpruned and pruned trees are calculated from performing cross validation on the test sets. Best cut-off threshold is obtained by tuning the threshold using the validation set. Pruned trees are CART trees that are pre-pruned by early stopping with the best threshold.

## References

1. Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. 
2. Vasudevan. (2014). Iterative Dichotomiser-3 Algorithm In Data Mining Applied To Diabetes Database. Journal of Computer Science,10(7), 1151-1155. doi:10.3844/jcssp.2014.1151.1155 
3. Chickering, D., Meek, C., & Rounthwaite, R. (2001). Efficient determination of dynamic split points in a decision tree. Proceedings 2001 IEEE International Conference on Data Mining. doi:10.1109/icdm.2001.989505 
4. Loh, W. (2011). Classification and regression trees. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery,1(1), 14-23. doi:10.1002/widm.8
5. Chawla, N. V. (n.d.). Data Mining for Imbalanced Datasets: An Overview. Data Mining and Knowledge Discovery Handbook,853-867. doi:10.1007/0-387-25465-x_40

