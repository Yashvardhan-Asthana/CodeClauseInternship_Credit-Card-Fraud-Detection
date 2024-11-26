# Credit Card Fraud Detection System

## Overview
This project is a **Fraud Detection System** that detects fraudulent transactions using machine learning. It leverages techniques such as oversampling, ensemble learning, and feature scaling to build an efficient classification system.

## Features
- **Multiple Models**: Random Forest, XGBoost, and Logistic Regression.
- **Ensemble Learning**: Combines models using a Voting Classifier for improved accuracy.
- **Interactive Predictions**: Accepts user inputs for real-time fraud detection.
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, F1-score, and confusion matrix for all models.

## Prerequisites
- Python 3.8 or higher
- Libraries: `pandas`, `scikit-learn`, `xgboost`, `imbalanced-learn`

## Dataset
The dataset is sourced from Kaggle and includes features like amt, city, merchant, and category, with is_fraud as the target column.

## Workflow
Load Dataset: Load and analyze the dataset for missing values and class distribution.
Preprocessing: Encode categorical features, balance data using SMOTE, and scale numeric features.
Train Models: Train Random Forest, XGBoost, Logistic Regression, and a Voting Classifier.
Evaluate Models: Assess accuracy and other metrics for all models.
Interactive Prediction: Use the trained Voting Classifier to classify transactions.

## Output
Model metrics and evaluation reports.
Fraud classification based on user input.

## Acknowledgments
Dataset from Kaggle.
Project inspired by real-world fraud detection challenges.
