import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import joblib

def load_data(path):
    df = pd.read_csv(path)
    print(df.shape)
    return df

def preprocess(df, target_column):
    df = df.dropna()
    # one-hot encoding for categorical columns
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_models(X_train_scaled, X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        data = X_train_scaled if name == "LogisticRegression" else X_train
        model.fit(data, y_train)
        joblib.dump(model, f"{name}.joblib")
    return models

def evaluate(models, X_test_scaled, X_test, y_test):
    results = []
    
    for name, model in models.items():
        # Use scaled data for Logistic Regression
        data = X_test_scaled if name == "LogisticRegression" else X_test
        y_pred = model.predict(data)
        auc = roc_auc_score(y_test, model.predict_proba(data)[:,1])
        
        print(f"\n{name} Evaluation")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC: {auc:.4f}")
        print(confusion_matrix(y_test, y_pred))
        
        results.append({"Model": name, "ROC_AUC": auc})
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = load_data("dataset/credit_risk_dataset.csv")
    X_train, X_test, y_train, y_test = preprocess(df, "loan_status")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Pass both scaled and unscaled data
    models = train_models(X_train_scaled, X_train, y_train)
    results = evaluate(models, X_test_scaled, X_test, y_test)
    
    # Save scaler
    joblib.dump(scaler, "scaler.joblib")
    print("\n✓ Models and scaler saved successfully!")
    print(results)