import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_dataset(filepath):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(filepath)
        print("Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {filepath}. Please check the path.")


def analyze_dataset(data, target_column):
    """Perform initial analysis on the dataset."""
    print("\nDataset Overview:")
    print(data.head())

    print("\nDataset Information:")
    print(data.info())

    print("\nClass Distribution:")
    if target_column in data.columns:
        print(data[target_column].value_counts())
    else:
        raise ValueError(f"The target column '{target_column}' is missing.")

    print("\nMissing Values:")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])


def preprocess_data(data, target_column, categorical_features, numeric_features):
    """Preprocess the data, including feature encoding, splitting, and scaling."""
    encoding_maps = {}

    # Frequency encode categorical features
    for col in categorical_features:
        encoding_maps[col] = data[col].value_counts(normalize=True).to_dict()
        data[col] = data[col].map(encoding_maps[col])

    # Separate features and target
    X = data[categorical_features + numeric_features]
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Standardize the numeric features
    scaler = StandardScaler()
    X_train_resampled[numeric_features] = scaler.fit_transform(X_train_resampled[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    return X_train_resampled, X_test, y_train_resampled, y_test, scaler, X_train.columns, encoding_maps


def train_model(X_train, y_train):
    """Train both Random Forest and Gradient Boosting models."""
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    print("Random Forest training completed.")

    print("\nTraining Gradient Boosting Classifier (XGBoost)...")
    xgb_model = XGBClassifier(random_state=42, n_estimators=100, use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    print("XGBoost training completed.")

    return rf_model, xgb_model


def evaluate_model(models, X_test, y_test):
    """Evaluate the trained models and return their accuracy scores."""
    accuracies = {}
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        accuracies[model_name] = accuracy
    return accuracies


def predict_transaction(models, accuracies, scaler, numeric_features, encoding_maps):
    """Predict whether a transaction is fraudulent using weighted voting."""
    print("\nEnter transaction details for prediction:")

    try:
        # Input fields
        amt = float(input("Amount: "))
        city = input("City: ").strip()
        merchant = input("Merchant Name: ").strip()
        category = input("Category: ").strip()
    except ValueError:
        print("Invalid input. Please provide numeric values for 'Amount' and valid strings for other fields.")
        return

    # Create input dictionary
    input_data = {
        'amt': amt,
        'city': city,
        'merchant': merchant,
        'category': category
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features using precomputed frequency maps
    for col, mapping in encoding_maps.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].map(mapping).fillna(0)

    # Scale numeric features
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    # Ensure all features match the model's input
    input_df = input_df.reindex(columns=list(models.values())[0].feature_names_in_, fill_value=0)

    # Calculate weighted probabilities
    fraud_score = 0
    total_weight = 0

    for model_name, model in models.items():
        weight = accuracies[model_name]
        probabilities = model.predict_proba(input_df)
        fraud_score += weight * probabilities[0][1]  # Probability of fraud (class 1)
        total_weight += weight

    weighted_average_score = fraud_score / total_weight

    # Threshold-based decision
    print("\nCombined Weighted Score:", weighted_average_score)
    print("Prediction:", "Fraud" if weighted_average_score > 0.5 else "Non-Fraud")


if __name__ == "__main__":
    # Filepath to dataset
    dataset_path = 'dataset.csv'
    target_column = 'is_fraud'
    categorical_features = ['city', 'merchant', 'category']
    numeric_features = ['amt']

    # Load and analyze dataset
    data = load_dataset(dataset_path)
    analyze_dataset(data, target_column)

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_columns, encoding_maps = preprocess_data(
        data, target_column, categorical_features, numeric_features
    )

    # Train models
    rf_model, xgb_model = train_model(X_train, y_train)

    # Evaluate models
    models = {"Random Forest": rf_model, "XGBoost": xgb_model}
    accuracies = evaluate_model(models, X_test, y_test)

    # Predict transactions in a loop
    while True:
        predict_transaction(models, accuracies, scaler, numeric_features, encoding_maps)
        cont = input("\nDo you want to predict another transaction? (yes/no): ").strip().lower()
        if cont != 'yes':
            break


