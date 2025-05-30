#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Risk Prediction Model
Converted from Jupyter notebook to terminal-runnable Python script
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*50)
    print("Financial Risk Prediction Model")
    print("="*50)
    
    # Data Loading
    print("\n1. Loading data...")
    try:
        df = pd.read_csv("financial_risk_assessment.csv")
        print(f"Data loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: financial_risk_assessment.csv not found!")
        print("Please make sure the file is in the same directory as this script.")
        return
    
    # Exploratory Data Analysis
    print("\n2. Exploratory Data Analysis")
    print("-" * 30)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset info:")
    df.info()
    
    print("\nStatistical summary:")
    print(df.describe())
    
    print("\nMissing values:")
    print(df.isna().sum())
    
    print("\nDuplicate values:")
    print(f"Total duplicates: {df.duplicated().sum()}")
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    plt.style.use('default')
    
    # Credit Score vs Risk Rating
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Risk Rating', y='Credit Score', data=df)
    plt.title('Credit Score vs Risk Rating')
    plt.tight_layout()
    plt.savefig('credit_score_vs_risk.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Saved: credit_score_vs_risk.png")
    
    # Employment Status by Risk Rating
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Risk Rating', hue='Employment Status', data=df)
    plt.title('Employment Status by Risk Rating')
    plt.tight_layout()
    plt.savefig('employment_vs_risk.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Saved: employment_vs_risk.png")
    
    # Correlation Matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix (Numerical Features)')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Saved: correlation_matrix.png")
    
    # Risk Rating Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Risk Rating', data=df)
    plt.title('Distribution of Risk Rating')
    plt.tight_layout()
    plt.savefig('risk_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Saved: risk_distribution.png")
    
    # Data Cleaning
    print("\n4. Data cleaning...")
    
    # Handle missing values
    num_cols_missing = ['Income', 'Credit Score', 'Loan Amount', 'Assets Value', 'Number of Dependents', 'Previous Defaults']
    
    imputer = SimpleImputer(strategy='median')
    df[num_cols_missing] = imputer.fit_transform(df[num_cols_missing])
    
    print(f"Missing values after imputation: {df.isna().sum().sum()}")
    
    # Encoding categorical features
    print("\n5. Encoding categorical features...")
    categorical_cols = [
        'Gender', 'Education Level', 'Marital Status', 'Loan Purpose',
        'Employment Status', 'Payment History', 'City', 'State', 'Country', 'Marital Status Change'
    ]
    
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    le_target = LabelEncoder()
    df['Risk Rating'] = le_target.fit_transform(df['Risk Rating'].astype(str))
    
    print("Categorical encoding completed.")
    
    # Normalize numerical features
    print("\n6. Normalizing numerical features...")
    numeric_cols = [
        'Age', 'Income', 'Credit Score', 'Loan Amount', 'Years at Current Job',
        'Debt-to-Income Ratio', 'Assets Value', 'Number of Dependents', 'Previous Defaults'
    ]
    
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print("Numerical normalization completed.")
    
    # Data splitting
    print("\n7. Splitting data...")
    X = df.drop('Risk Rating', axis=1)
    y = df['Risk Rating']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Training set shape: {X_train_resampled.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Model Training
    print("\n8. Training models...")
    print("-" * 30)
    
    # XGBoost
    print("Training XGBoost...")
    class_weights = compute_class_weight(class_weight='balanced',
                                       classes=np.unique(y_train_resampled),
                                       y=y_train_resampled)
    weights_dict = dict(zip(np.unique(y_train_resampled), class_weights))
    sample_weights = pd.Series(y_train_resampled).map(weights_dict)
    
    model_xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
    model_xgb.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train_resampled, y_train_resampled)
    
    # SVM
    print("Training SVM...")
    svm_model = SVC(kernel='rbf', C=1, decision_function_shape='ovr', probability=True)
    svm_model.fit(X_train_resampled, y_train_resampled)
    
    # Naive Bayes
    print("Training Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train_resampled, y_train_resampled)
    
    # Model Evaluation
    print("\n9. Model evaluation...")
    print("="*50)
    
    # XGBoost evaluation
    y_pred_xgb = model_xgb.predict(X_test)
    print("\nXGBoost Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_xgb, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))
    
    # Random Forest evaluation
    y_pred_rf = rf.predict(X_test)
    print("\nRandom Forest Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_rf, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    
    # SVM evaluation
    y_pred_svm = svm_model.predict(X_test)
    print("\nSVM Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_svm, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))
    
    # Naive Bayes evaluation
    y_pred_nb = nb_model.predict(X_test)
    print("\nNaive Bayes Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_nb, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_nb))
    
    # Prediction on new data
    print("\n10. Testing with new data...")
    print("-" * 30)
    
    # Create sample new data
    new_data = pd.DataFrame([
        {
            'Gender': 'Female',
            'Education Level': 'Master',
            'Marital Status': 'Married',
            'Loan Purpose': 'Business',
            'Employment Status': 'Self-employed',
            'Payment History': 'Average',
            'City': 'Bandung',
            'State': 'Jawa Barat',
            'Country': 'Indonesia',
            'Marital Status Change': 'Yes',
            'Age': 35,
            'Income': 8000,
            'Credit Score': 690,
            'Loan Amount': 15000,
            'Years at Current Job': 5,
            'Debt-to-Income Ratio': 0.25,
            'Assets Value': 30000,
            'Number of Dependents': 1,
            'Previous Defaults': 0
        },
        {
            'Gender': 'Male',
            'Education Level': 'Bachelor',
            'Marital Status': 'Single',
            'Loan Purpose': 'Education',
            'Employment Status': 'Employed',
            'Payment History': 'Good',
            'City': 'Jakarta',
            'State': 'DKI Jakarta',
            'Country': 'Indonesia',
            'Marital Status Change': 'No',
            'Age': 28,
            'Income': 5000,
            'Credit Score': 720,
            'Loan Amount': 10000,
            'Years at Current Job': 2,
            'Debt-to-Income Ratio': 0.18,
            'Assets Value': 20000,
            'Number of Dependents': 0,
            'Previous Defaults': 0
        },
        {
            'Gender': 'Female',
            'Education Level': 'PhD',
            'Marital Status': 'Divorced',
            'Loan Purpose': 'Home',
            'Employment Status': 'Unemployed',
            'Payment History': 'Poor',
            'City': 'Surabaya',
            'State': 'Jawa Timur',
            'Country': 'Indonesia',
            'Marital Status Change': 'Yes',
            'Age': 45,
            'Income': 3000,
            'Credit Score': 550,
            'Loan Amount': 20000,
            'Years at Current Job': 0,
            'Debt-to-Income Ratio': 0.4,
            'Assets Value': 10000,
            'Number of Dependents': 3,
            'Previous Defaults': 2
        }
    ])
    
    # Preprocess new data
    new_data_processed = new_data.copy()
    
    # Encode categorical features
    for col in categorical_cols:
        le = label_encoders[col]
        new_data_processed[col] = new_data_processed[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
    
    # Scale numerical features
    new_data_processed[numeric_cols] = scaler.transform(new_data_processed[numeric_cols])
    
    # Ensure column order matches training data
    X_input = new_data_processed[X.columns]
    
    # Make predictions
    xgb_preds = model_xgb.predict(X_input)
    xgb_labels = le_target.inverse_transform(xgb_preds)
    
    rf_preds = rf.predict(X_input)
    rf_labels = le_target.inverse_transform(rf_preds)
    
    svm_preds = svm_model.predict(X_input)
    svm_labels = le_target.inverse_transform(svm_preds)
    
    nb_preds = nb_model.predict(X_input)
    nb_labels = le_target.inverse_transform(nb_preds)
    
    # Display results
    results = pd.DataFrame({
        'Sample': [f'Sample {i+1}' for i in range(len(new_data))],
        'XGB Prediction': xgb_labels,
        'RF Prediction': rf_labels,
        'SVM Prediction': svm_labels,
        'NB Prediction': nb_labels
    })
    
    print("Predictions for new data samples:")
    print(results)
    
    print("\n" + "="*50)
    print("Analysis completed successfully!")
    print("Generated files:")
    print("- credit_score_vs_risk.png")
    print("- employment_vs_risk.png") 
    print("- correlation_matrix.png")
    print("- risk_distribution.png")
    print("="*50)

if __name__ == "__main__":
    main()