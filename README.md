# TitanicKagglewith_SVC
This code is used to clean and transform data for the Titanic survival prediction problem. The first part of the 
code computes summary statistics for categorical features and calculates mean survival rates for different features.
The second part drops irrelevant columns, creates a new feature called 'Title', and maps it to numerical values. 
It also imputes missing age values and converts the 'Age' feature into an ordinal 'AgeBand' feature.

The relevance of this code to the support vector machine (SVM) method in machine learning is that SVMs require well
-prepared data that is free from missing values and irrelevant features. Therefore, the data cleaning and feature 
engineering steps performed by this code are essential for improving the accuracy of SVM models. The imputation of 
missing age values and creation of the 'AgeBand' feature also helps to improve the performance of SVMs by providing 
more meaningful features for the model to learn from.


