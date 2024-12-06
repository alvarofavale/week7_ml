# week7_ml

## Week7: Machine Learning Project - Credit Score Classification

### Table of Contents
- [Description](#description)
- [Data Set Research and Selection](#data-set-research-and-selection)
- [Insights](#insights)
- [Model Results](#model-results)
  - [Linear Regression](#linear-regression)
  - [Random Forest Regressor](#random-forest-regressor)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Gradient Boost Regressor](#gradient-boost-regressor)
  - [Gradient Boost Classifier](#gradient-boost-classifier)
- [Conclusion](#conclusion)


### Description

This machine learning project aims to classify customers' credit scores using various regression and classification techniques. The goal is to determine which model offers the best performance in terms of accuracy and efficiency for predicting customers' creditworthiness.

### Data Set Research and Selection

The dataset used for this project was selected from [Statso](https://statso.io/credit-score-classification-case-study/). This dataset includes diverse financial and demographic characteristics of customers, enabling the evaluation of their credit scores.

### Insights

- **Regressor vs Classifier:**
  - **Classifier models are more efficient with less tuning** compared to regressor models.

- **Model Optimization:**
  - Achieving the highest accuracy in predictive modeling is a journey characterized by "trial and error." With each iteration, models are refined and improved, demanding patience and perseverance to discover the ideal configuration.

- **Model Selection:**
  - For tasks with clearly defined categories, classifiers are your go-to option for efficient modeling with minimal tuning effort.

### Model Results

#### Linear Regression
- **MAE:** 0.57
- **MSE:** 0.52
- **RMSE:** 0.72
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

### Conclusion

The results indicate that classification models, particularly the **Random Forest Classifier**, outperform regression models in terms of accuracy for credit score classification. This confirms that for classification tasks with well-defined categories, classifier models are more suitable and require less tuning to achieve good performance.
