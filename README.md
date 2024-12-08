# Decision-Tree-Ensemble-Algorithms
A Python implementation of ensemble learning algorithms from scratch, including Gradient Boosting Machine (GBM), Random Forest, AdaBoost, and Decision Trees. This repository also showcases XGBoost, CatBoost, LightGBM for classification, regression, and ranking tasks, with visualizations and performance comparisons.

# Machine Learning Models Implementation and Evaluation

This repository contains various notebooks that implement and evaluate machine learning models, primarily focusing on decision trees, ensemble methods like AdaBoost and Gradient Boosting, and ranking algorithms.

## Notebooks Overview:

### 1. **AdaBoost Implementation and Feature Importance**
- **Description**: Demonstrates the AdaBoost algorithm with weak learners (decision stumps). The notebook also explores feature importance, identifying which features are most influential in the model's predictions.
- **Key Techniques**: AdaBoost, Feature Importance

### 2. **Decision Tree and Random Forest Models**
- **Description**: Compares the performance of Decision Tree and Random Forest classifiers. Both models are evaluated on a given dataset to assess their effectiveness in classification tasks.
- **Key Techniques**: Decision Trees, Random Forests, Model Evaluation

### 3. **Decision Tree from Scratch: Gini Impurity and Entropy**
- **Description**: Implements a decision tree from scratch for classification, using Gini Impurity and Entropy as the criteria for splitting the data.
- **Key Techniques**: Decision Trees, Gini Impurity, Entropy

-    Decision Tree  Colab Link:  https://github.com/subhashpolisetti/Decision-Tree-Ensemble-Algorithms/blob/main/Decision_Tree_From_Scratch_Gini_Entropy.ipynb

### 4. **Evaluating Classification Algorithms on the Breast Cancer Dataset**
- **Description**: Evaluates multiple classification algorithms, including Decision Trees, Random Forests, and AdaBoost, on the Breast Cancer dataset. The models are compared based on accuracy.
- **Key Techniques**: Classification, Model Evaluation, Breast Cancer Dataset

### 5. **Gradient Boosting with Decision Trees**
- **Description**: Implements gradient boosting using decision trees, where each tree predicts the residuals of the previous tree. This technique improves prediction accuracy iteratively.
- **Key Techniques**: Gradient Boosting, Decision Trees, Residual Prediction

### 6. **Gradient Boosting Regression Techniques**
- **Description**: Applies gradient boosting regression techniques to predict continuous target variables. The notebook focuses on training and tuning models for regression tasks.
- **Key Techniques**: Gradient Boosting, Regression, Hyperparameter Tuning

- Gradient Boost Regression  - Colab Link: https://github.com/subhashpolisetti/Decision-Tree-Ensemble-Algorithms/blob/main/GradientBoosting_Regression_Techniques.ipynb

### 7. **Gradient Boosting Ranking Models (XGBoost, LightGBM, CatBoost)**
- **Description**: Demonstrates ranking models using XGBoost, LightGBM, and CatBoost. The models are trained on synthetic ranking data to predict the relevance of items in ranked lists.
- **Key Techniques**: Ranking Models, XGBoost, LightGBM, CatBoost

- Gradient Boost Ranking -  Colab Link: https://github.com/subhashpolisetti/Decision-Tree-Ensemble-Algorithms/blob/main/GradientBoostRankingTechniques_XGBoost_LightGBM_CatBoost.ipynb


Vedio Explaination Link: https://www.youtube.com/playlist?list=PL6O21IOHvBmf8ABZKBjLhLXlTDBYtKZwx

## Requirements:
- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `xgboost`
  - `lightgbm`
  - `catboost`
  - `fastai`

## How to Use:
1. Clone this repository to your local machine or use it in a Jupyter notebook environment.
2. Open the relevant notebook in Jupyter or Google Colab.
3. Run the cells to see the model implementations, visualizations, and evaluations.


