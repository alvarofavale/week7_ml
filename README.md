# Week 7: Machine Learning Project - Credit Score Classification

## Overview

In this week's project, we will work on building a machine learning model for classifying credit scores. This will involve analyzing a dataset and implementing a machine learning model to predict credit scores based on various factors. The dataset is sourced from [Statso](https://statso.io/credit-score-classification-case-study/).

## Dataset Research and Selection

The dataset used for this project contains information relevant to credit scoring and classification. This dataset can be used to predict whether an individual is likely to have a good or bad credit score, which is essential for determining loan eligibility and financial stability.

## Goal of the Project

The objective is to build a machine learning model that can classify an individual’s credit score into categories like "Good" or "Bad" based on various financial attributes. The project will cover the following steps:

1. **Data Exploration**: Inspect and understand the dataset, including missing values and data types.
2. **Data Preprocessing**: Clean the data, handle missing values, encode categorical variables, and scale numerical values.
3. **Model Training**: Use machine learning algorithms (like Logistic Regression, Random Forest, or Support Vector Machines) to build a classifier.
4. **Model Evaluation**: Evaluate the model using appropriate metrics such as accuracy, precision, recall, and F1-score.
5. **Tuning and Improvement**: Fine-tune the model for better performance using techniques like cross-validation and hyperparameter tuning.

## Approach

1. **Cleaning and Visualizations**: 
   - Primary cleaning and data wrangling.
   - Correlation matrix analysis.

2. **Model Testing**: 
   - Start with KNN.
   - Apply both Classifier and Regressor, evaluate, and decide to keep using KNN Classifier.

3. **Feature Implementation**: 
   - Implement initial features and refine the approach to model usage.

4. **Testing Other Models**: 
   - Test Decision Trees and Linear Regression.
   - Decide to implement an Ensemble model, opting for Random Forest.
   - Test both Regressor and Classifier, ultimately opting for Classifier.

5. **Further Tuning**: 
   - Implement cross-validation and hyperparameters.
   - Perform extensive tests and normalization.

6. **Getting to Final Figures**: 
   - Draw conclusions, print Confusion Matrices, implement parameters.
   - Finalize models and prepare the presentation.

## Repo Organization

### Folder Structure

- **data**  
  - `raw`: CSV file used  
  - `encoding`: CSV file post mapping and categorical column encoding

- **notebooks**  
  - Person name folders: Each team member proceeded to their own model tests  
    - `Tung`  
    - `Aurelie`  
    - `Ceci`

#### Folder Structure for "Ceci"

- **folders**  
  - **figures**: Boxplots and violin plots made for testing features before encoding  
  - **figuresb**: Boxplots and violin plots made for testing features after encoding  
  - **figures_usable**: Useful figures for analysis  
  - **figures_correlations**: Printing final correlation matrix from hyperparameter testing

- **jupyter notebooks**  
  - `ml_firstKNN`: Initial KNN testing and presence of KNN Classifier and Regressor  
  - `encoding`: Encoding of all categorical columns and printing the .csv file post mapping them correctly  
  - `KNN_application_post_encoding`: Testing KNN again after encoding columns with updated .csv file  
  - `KNN_feature_implementation`: Implementing features, testing KNN post encoding, determining splits to use, hyperparameter testing, implementation and fine-tuning, final tests  
  - `linear_regression`: Attempted linear regression model  
  - `full_KNN_implementation_simplified`: Defined functions for implementing features, hyperparameter testing, and confusion matrix printing all in one notebook for both of the kept split types: 70% train/30% test and 80% train/20% test

## Reference

Dataset source: [Statso Credit Score Classification Case Study](https://statso.io/credit-score-classification-case-study/)
