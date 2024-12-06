# Week7: Machine Learning Project - Credit Score Classification

## Table of Contents
- [Description](#description)
- [Data Set Research and Selection](#data-set-research-and-selection)
- [Insights](#insights)
- [Model Results](#model-results)
  - [Linear Regression](#linear-regression)
  - [Random Forest Regressor](#random-forest-regressor)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Gradient Boost Regressor](#gradient-boost-regressor)
  - [Gradient Boost Classifier](#gradient-boost-classifier)
  - [KNN Classifier](#knn-classifier)
- [Conclusion](#conclusion)

### Description

This machine learning project aims to classify customers' credit scores using various regression and classification techniques. The goal is to determine which model offers the best performance in terms of accuracy and efficiency for predicting customers' creditworthiness.

### Data Set Research and Selection

The dataset used for this project was selected from [Statso](https://statso.io/credit-score-classification-case-study/). This dataset includes diverse financial and demographic characteristics of customers, enabling the evaluation of their credit scores. The key features of the dataset include:

- **Age**: The age of the individual
- **Income**: The annual income of the individual
- **Loan Amount**: The loan amount requested or borrowed
- **Credit History**: The past history of the individual’s credit repayment (good or bad)
- **Debt-to-Income Ratio**: The ratio of debt to income, an important factor in determining creditworthiness
- **Employment Status**: Whether the individual is employed, self-employed, or unemployed
- **Other Financial Metrics**: Includes assets, monthly expenses, and other related financial information

The objective is to classify individuals into categories of creditworthiness ("Good" or "Bad") based on these attributes.

### Insights

- **Regressor vs Classifier**:
  - **Classifier models are more efficient with less tuning** compared to regressor models.
  - For binary classification tasks, classification models outperform regression models in terms of performance and simplicity.

- **Model Optimization**:
  - Achieving the highest accuracy in predictive modeling is a journey characterized by "trial and error." With each iteration, models are refined and improved, demanding patience and perseverance to discover the ideal configuration.

- **Model Selection**:
  - For tasks with clearly defined categories, classifiers are your go-to option for efficient modeling with minimal tuning effort.

### Model Results

The following models were tested to assess the performance in predicting credit scores:

#### Linear Regression
- **MAE (Mean Absolute Error):** 0.57
- **MSE (Mean Squared Error):** 0.52
- **RMSE (Root Mean Squared Error):** 0.72
- **Accuracy:** 0.11

#### Random Forest Regressor
- **MAE:** 0.42
- **MSE:** 0.33
- **RMSE:** 0.57
- **Accuracy:** 0.44

#### Random Forest Classifier
- **MAE:** 0.42
- **MSE:** 0.33
- **RMSE:** 0.57
- **Accuracy:** 0.80

#### Gradient Boost Regressor
- **MAE:** 0.36
- **MSE:** 0.32
- **RMSE:** 0.57
- **Accuracy:** -0.65

#### Gradient Boost Classifier
- **MAE:** 0.39
- **MSE:** 0.62
- **RMSE:** 0.79
- **Accuracy:** 0.53

#### KNN Classifier

Evaluating the **KNN Classifier** for different data splits:

**For 70% Train / 30% Test Split**:
- **Accuracy:** 0.974
- **Classification Report**:

  |               | Precision | Recall | F1-Score | Support |
  |---------------|-----------|--------|----------|---------|
  | 0             | 0.99      | 0.96   | 0.97     | 5322    |
  | 1             | 0.97      | 0.98   | 0.98     | 15873   |
  | 2             | 0.96      | 0.98   | 0.97     | 8805    |
  | **Accuracy**  |           |        | **0.97** | 30000   |
  | **Macro Avg** | 0.98      | 0.97   | 0.97     | 30000   |
  | **Weighted Avg** | 0.97   | 0.97   | 0.97     | 30000   |

**For 80% Train / 20% Test Split**:
- **Accuracy:** 0.973
- **Classification Report**:

  |               | Precision | Recall | F1-Score | Support |
  |---------------|-----------|--------|----------|---------|
  | 0             | 0.99      | 0.96   | 0.97     | 3527    |
  | 1             | 0.98      | 0.97   | 0.97     | 10599   |
  | 2             | 0.96      | 0.98   | 0.97     | 5874    |
  | **Accuracy**  |           |        | **0.97** | 20000   |
  | **Macro Avg** | 0.97      | 0.97   | 0.97     | 20000   |
  | **Weighted Avg** | 0.97   | 0.97   | 0.97     | 20000   |

### Conclusion

The results indicate that **classification models**, particularly the **Random Forest Classifier** and **KNN Classifier**, outperform regression models in terms of accuracy for credit score classification. This confirms that for classification tasks with well-defined categories, classifier models are more suitable and require less tuning to achieve good performance.

### Repo Organization

#### Folder Structure

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

### Reference

Dataset source: [Statso Credit Score Classification Case Study](https://statso.io/credit-score-classification-case-study/)