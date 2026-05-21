# Bank Customer Churn Prediction using Machine Learning

## Overview
Customer churn is one of the biggest challenges faced by banks and financial institutions.  
This project aims to predict whether a customer will leave the bank or continue as a customer using various Machine Learning classification algorithms.

## Business Problem

Customer churn directly impacts a bank’s profitability and customer retention.  
Predicting customer churn helps banks identify high-risk customers and take preventive actions such as personalized offers, customer engagement strategies, and retention campaigns.

The project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Feature engineering
- Model building
- Performance comparison
- Handling imbalanced datasets using SMOTE and undersampling

---

## Dataset Information

The dataset contains information about bank customers such as:

- Credit Score
- Geography
- Gender
- Age
- Balance
- Tenure
- Number of Products
- Credit Card Status
- Activity Status
- Estimated Salary

### Target Variable
- Exited = 1 → Customer left the bank
- Exited = 0 → Customer continues with the bank

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

## Machine Learning Algorithms Used

The following classification models were implemented and compared:

1. K-Nearest Neighbors (KNN)
2. Naive Bayes
3. Support Vector Classifier (SVC)
4. Decision Tree Classifier
5. Random Forest Classifier
6. AdaBoost Classifier

---

## Project Workflow

### 1. Data Preprocessing
- Removed unnecessary columns
- Checked missing values
- Label Encoding for categorical data
- Feature scaling using MinMaxScaler

### 2. Exploratory Data Analysis
Performed visualization and analysis on:
- Customer churn distribution
- Churn by geography
- Churn by gender
- Churn by age
- Churn by activity status
- Churn by number of products

### 3. Model Building
- Train-test split
- Model training and prediction
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

### 4. Handling Imbalanced Data
Implemented:
- SMOTE Oversampling
- Random Undersampling

---

## Model Performance

| Model | Accuracy |
|-------|----------|
| KNN | 81% |
| Naive Bayes | 83% |
| SVC | 84% |
| Decision Tree | 79% |
| Random Forest | 86% |
| AdaBoost | 86% |

## Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Data Visualization
- Feature Engineering
- Classification Algorithms
- Model Evaluation
- Handling Imbalanced Data
- Machine Learning Workflow

## Results

- Random Forest and AdaBoost achieved the highest accuracy of 86%.
- SMOTE improved class balance and model stability.
- Random Forest performed best overall for churn prediction.

---


