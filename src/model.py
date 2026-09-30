"""
model.py
--------
Model training, evaluation, saving, and inference for species conservation risk.
Uses a scikit-learn RandomForestClassifier.
"""

import os
import sys

# Ensure project root is in sys.path whether executed directly or imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.preprocess import load_data, prepare_features, encode_single_input

MODEL_PATH = "models/model.pkl"

def train_and_evaluate_model(df=None, save_path=MODEL_PATH, test_size=0.2, random_state=42):
    """
    Trains a Random Forest Classifier on the species dataset.
    
    Parameters:
        df (pd.DataFrame, optional): Loaded species dataset.
        save_path (str): Destination for serialized model artifact.
        test_size (float): Proportion of data used for testing.
        random_state (int): Seed for reproducibility.
        
    Returns:
        dict: Evaluation metrics, trained model, and encoders.
    """
    if df is None:
        df = load_data()

    # Preprocess features and targets
    X, y, encoders = prepare_features(df)

    # Stratified Train/Test split ensures balanced class distribution in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Initialize Random Forest Classifier
    # Ensemble of 100 decision trees to prevent overfitting and handle nonlinear relationships
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state
    )

    # Train model
    clf.fit(X_train, y_train)

    # Predictions for evaluation
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(
        y_test, 
        y_pred, 
        target_names=encoders["status_classes"], 
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()

    # Feature Importances
    feature_importance_df = pd.DataFrame({
        "Feature": encoders["feature_names"],
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    results = {
        "model": clf,
        "encoders": encoders,
        "accuracy": round(acc * 100, 2),
        "confusion_matrix": cm,
        "classification_report_df": report_df,
        "feature_importances": feature_importance_df,
        "classes": encoders["status_classes"]
    }

    # Save to models/model.pkl
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(results, save_path)
    print(f"Model successfully saved to '{save_path}' with Accuracy: {results['accuracy']}%")

    return results

def load_or_train_model(save_path=MODEL_PATH, force_retrain=False):
    """
    Loads saved model bundle if exists, otherwise trains and saves a new model.
    """
    if os.path.exists(save_path) and not force_retrain:
        try:
            return joblib.load(save_path)
        except Exception:
            pass
    return train_and_evaluate_model(save_path=save_path)

def predict_conservation_risk(model_bundle, category, population_count, 
                              habitat_loss_percent, avg_temperature, 
                              human_activity_index, protected_area):
    """
    Predicts conservation status and confidence probabilities for single sample input.
    
    Returns:
        predicted_status (str): Predicted IUCN category name.
        probabilities (dict): Dictionary mapping class name to predicted probability.
    """
    clf = model_bundle["model"]
    encoders = model_bundle["encoders"]

    # Transform input into feature vector
    input_vector = encode_single_input(
        category=category,
        population_count=population_count,
        habitat_loss_percent=habitat_loss_percent,
        avg_temperature=avg_temperature,
        human_activity_index=human_activity_index,
        protected_area=protected_area,
        encoders=encoders
    )

    pred_idx = clf.predict(input_vector)[0]
    predicted_status = encoders["status_classes"][pred_idx]

    # Predict class probabilities
    probs = clf.predict_proba(input_vector)[0]
    prob_dict = {
        cls_name: round(float(prob) * 100, 2)
        for cls_name, prob in zip(encoders["status_classes"], probs)
    }

    return predicted_status, prob_dict

if __name__ == "__main__":
    train_and_evaluate_model()
