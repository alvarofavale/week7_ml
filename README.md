# Week 7: Machine Learning Project - Credit Score Classification

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
- [Repo Organization](#repo-organization)

### Description

This machine learning project aims to classify customers' credit scores using various regression and classification techniques. The goal is to identify which model offers the best performance in predicting customers' creditworthiness, evaluating both accuracy and efficiency.

### Data Set Research and Selection

The dataset used for this project was sourced from [Statso](https://statso.io/credit-score-classification-case-study/). It includes a wide range of financial and demographic characteristics of customers, which are useful for evaluating their credit scores. The key features include:

- **Age**: Age of the individual
- **Income**: Annual income of the individual
- **Loan Amount**: Loan amount requested or borrowed
- **Credit History**: The individual’s past credit repayment history (good or bad)
- **Debt-to-Income Ratio**: Ratio of debt to income, an important factor in determining creditworthiness
- **Employment Status**: Employment status (employed, self-employed, unemployed)
- **Other Financial Metrics**: Includes assets, monthly expenses, and other financial information

The objective is to classify individuals into categories of creditworthiness ("Good" or "Bad") based on these attributes.

### Insights

- **Regressor vs Classifier**:  
  - Classifier models tend to be more efficient and require less tuning compared to regressor models.  
  - For binary classification tasks, classification models generally outperform regression models in terms of accuracy and simplicity.
  
- **Model Optimization**:  
  - Achieving optimal accuracy is often a process of "trial and error." Through iterative testing and refinement, models are improved over time, demanding both patience and perseverance to achieve the best configuration.

- **Model Selection**:  
  - For tasks with well-defined categories, classifier models are generally preferred due to their ability to provide efficient solutions with minimal tuning.

### Model Results

The following models were evaluated to assess their performance in predicting credit scores:

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
  - `raw`: Original CSV dataset  
  - `encoding`: Encoded CSV file after mapping and handling categorical variables

- **notebooks**
  - Person-specific folders: Each team member tested their own models
    - `Tung`
    - `Aurelie`
    - `Ceci`
  - **ml_cleaning_af**:  
    - **Eliminate Null Values**: Swiftly remove missing data to enhance dataset integrity and analytical precision.  
    - **Assess Unique Values**: Investigate distinct entries to gain insights and ensure data consistency.  
    - **Refine Column Labels**: Standardize and clarify labeling for seamless data manipulation and improved readability.

- **slides**
  - Includes presentation PDF  

- **stills**
  - Includes decision tree figure

#### Folder Structure for "Aurelie"

- **jupyter notebooks**
  - `loan_and_clean`: First exploration of the dataset, including data cleaning and initial analysis.  
  - `ml_supervised`: Testing supervised machine learning models on the dataset.  
  - `Hyperparameter_random_forest`: Fine-tuning the Random Forest model through hyperparameter optimization.

#### Folder Structure for "Ceci"

- **folders**  
  - `figures`: Boxplots and violin plots for testing features before encoding  
  - `figuresb`: Boxplots and violin plots for testing features after encoding  
  - `figures_usable`: Useful figures for analysis  
  - `figures_correlations`: Final correlation matrix from hyperparameter testing

- **jupyter notebooks**
  - `ml_firstKNN`: Initial KNN testing, including both KNN Classifier and Regressor  
  - `encoding`: Encoding categorical columns and saving the updated CSV file  
  - `KNN_application_post_encoding`: Testing KNN after encoding columns with the updated dataset  
  - `KNN_feature_implementation`: Feature implementation, hyperparameter testing, model fine-tuning, and final tests  
  - `linear_regression`: Attempted linear regression model  
  - `full_KNN_implementation_simplified`: Simplified implementation of KNN with functions for hyperparameter tuning and confusion matrix output, for both 70% train/30% test and 80% train/20% test splits

#### Folder Structure for "Tung"

- **jupyter notebooks**
  - `KNN_notebook`: Rough and unfiltered KNN calculation for a first feel, achieving ~45% accuracy. Created a numerical dataframe and replaced target "credit_score" with numerical values.
  - `decision_tree_notebook`: Limited the dataset to a random 1000 `client_ids` to improve performance. Normalized the train/test set and used `DecisionTreeRegressor` along with `graphviz` to visualize the tree.
  - `decision_boundary_notebook`: Applied `StandardScaler` and `RandomForestClassifier` to the dataset. Used a scatterplot to visualize the decision boundary.

### Reference

- Dataset source: [Statso Credit Score Classification Case Study](https://statso.io/credit-score-classification-case-study/)
- **Organization resources**:
  - **Kanban Board**: [Kanban Board for ML Project](https://trello.com/b/md5dLfY4/kanban-template-machine-learning-project)
  - **Presentation**: [Project Presentation](https://www.canva.com/design/DAGYKLZWHtE/fMu7Man2KpF1MNT8d74UFg/view?utm_content=DAGYKLZWHtE&utm_campaign=designshare&utm_medium=link&utm_source=editor)
