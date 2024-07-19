---
layout: post
title: "Analyzing Customer Trends and Predicting Churn in the Fintech Industry"
subtitle: "A Comprehensive Machine Learning Approach"
date: 2024-07-20
author: Harsha Vuppalapati
tags: [data science, machine learning, fintech, churn analysis]
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
comments: true
mathjax: true
---

# Analyzing Customer Trends and Predicting Churn in the Fintech Industry

## Introduction

In our previous sessions, we explored various machine learning algorithms and their applications. Now, it's time to apply that knowledge to a hands-on portfolio project focused on the fintech industry. Customer churn analysis is a critical task for financial institutions, as it helps them understand why customers leave and how to retain them. This project involves analyzing customer trends and predicting churn using machine learning techniques.

## Dataset Overview

The dataset for this project is sourced from Kaggle and contains a wealth of information about customers in the fintech sector. We will leverage this data to perform a comprehensive churn analysis and build predictive models to identify at-risk customers.

### Dataset Details
- *CustomerID*: Unique identifier for each customer
- *Surname*: Customer's surname
- *CreditScore*: Credit score of the customer
- *Geography*: Customer's location
- *Gender*: Gender of the customer
- *Age*: Age of the customer
- *Tenure*: Number of years the customer has been with the bank
- *Balance*: Account balance of the customer
- *NumOfProducts*: Number of products the customer has with the bank
- *HasCrCard*: Whether the customer has a credit card
- *IsActiveMember*: Whether the customer is an active member
- *EstimatedSalary*: Estimated salary of the customer
- *Exited*: Whether the customer has left the bank (target variable)

## Project Workflow

### 1. Data Loading and Preprocessing

We start by loading the dataset into a pandas DataFrame and performing initial data cleaning, handling missing values, and preparing the data for analysis.

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

# Display the first few rows
print(df.head())
```
### 2. Exploratory Data Analysis (EDA)

EDA helps us understand the data distribution, relationships between features, and identify any patterns or anomalies. Visualizing the data is a key part of this process.

```python

import matplotlib.pyplot as plt
import seaborn as sns

# Plot distribution of numerical features
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
for i, ax in enumerate(axes.flatten()):
    sns.histplot(df.iloc[:, i+3], kde=True, ax=ax)
plt.tight_layout()
plt.show()
``` 
### 3. Data Cleaning and Feature Engineering

Next, we handle missing values, encode categorical variables, and scale numerical features to prepare the data for modeling.

```python

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Encode categorical variables
label_encoder = LabelEncoder()
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Scale numerical features
scaler = StandardScaler()
df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']] = scaler.fit_transform(df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']])

#### Drop unnecessary columns
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
```
### 4. Model Building

We begin with a logistic regression model to predict customer churn and evaluate its performance using confusion matrix, classification report, and accuracy score.

```python

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Split the data
X = df.drop(columns=['Exited'])
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#### Make predictions
y_pred = model.predict(X_test)

#### Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
```
### 5. Advanced Modeling

To improve our model's performance, we explore advanced algorithms such as Random Forest and Gradient Boosting, comparing their accuracy and other evaluation metrics.

```python

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#### Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

#### Train Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

#### Evaluate the models
print('Random Forest Accuracy:', accuracy_score(y_test, rf_pred))
print('Gradient Boosting Accuracy:', accuracy_score(y_test, gb_pred))
```
### 6. Hyperparameter Tuning and Model Optimization

To further enhance our model's performance, we perform hyperparameter tuning using techniques such as Grid Search and Random Search.

```python

from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print(grid_search.best_params_)
print(grid_search.best_score_)
```
### 7. Model Evaluation and Interpretation

Finally, we evaluate the best model on the test set and interpret the results, focusing on key metrics such as accuracy, precision, recall, F1-score, and feature importance.

```python

# Evaluate the best model
best_model = grid_search.best_estimator_
y_best_pred = best_model.predict(X_test)

# Print evaluation metrics
print(confusion_matrix(y_test, y_best_pred))
print(classification_report(y_test, y_best_pred))
print('Accuracy:', accuracy_score(y_test, y_best_pred))
```
## Conclusion

This project demonstrates a comprehensive approach to analyzing customer trends and predicting churn in the fintech industry using machine learning. By leveraging a diverse set of features and advanced modeling techniques, we gain valuable insights into customer behavior and churn patterns. This project not only showcases data science and machine learning skills but also highlights domain-specific knowledge in the finance sector.
References

    Kaggle Dataset
    Scikit-learn Documentation
    Matplotlib Documentation
    Seaborn Documentation
    Python Data Science Handbook
    Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
