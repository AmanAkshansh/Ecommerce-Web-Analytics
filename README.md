# Ecommerce-Web-Analytics
Binary classification using Random Forest

# Dataset
The datasets are compressed as rar files because of their big sizes.
The two datasets that are contained inside dataset.rar file are:
(i) Training - train.csv
(ii) Testing - test.csv

# Data description:
1. 'Target' column in the Training set is class label.
2. '-1' signifies missing value in the columns.
3. TOP: time on page(in seconds)
4. Exits: No. of times unique Id has exited the page
5. Binary_var: contains binary values
6. Metric_var: contains continuous variables
7. Page_visited: whether unique Id has visited the page or not

# Business problem:
The objective of this project is to predict successful 'Unique Ids'from unique Id column who have high chance of getting 1 as 'Target' column value.
Following needs to be submitted:
1. output file .csv format which consists of two columns 'Unique Id' and 'Predict_target'
2. Evaluation metrices - (a) confusion matrix (b) F1 score (c) Accuracy score (d) ROC curve (e) AUC score

# Techniques used:
1. Explanatory data analysis and Data cleaning/preparation.
2. Used Random forest algorithm to create the ML model.
3. Created confusion matrix to see the accuracy,sensitivity and misclassification rate.
4. Used ROSE package to handle the class imbalance in 'Target' variable in order to increase the sensitivity of the model.
5. Selected the sample which gave the best sensitivity and created the Random forest model with this sampled datset.
6. Calculated all the required Evaluation metrices.
7. Predicted the result on Testing dataset.


