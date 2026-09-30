"""
preprocess.py
-------------
Data loading, cleaning, validation, and feature preparation routines
for the Biodiversity Conservation Analytics project.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Standard column definitions
FEATURE_COLUMNS = [
    "category",
    "population_count",
    "habitat_loss_percent",
    "avg_temperature",
    "human_activity_index",
    "protected_area"
]

TARGET_COLUMN = "conservation_status"

STATUS_ORDER = ["Least Concern", "Vulnerable", "Endangered", "Critically Endangered"]
CATEGORY_ORDER = ["Mammal", "Bird", "Reptile", "Amphibian", "Plant"]

def load_data(csv_path="data/species_data.csv"):
    """
    Loads species data from CSV file. Raises FileNotFoundError if missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at: {csv_path}. Please run data_generator.py first.")
    df = pd.read_csv(csv_path)
    return df

def get_data_summary(df):
    """
    Computes basic metadata, missing values count, and statistics for the dataset.
    """
    summary = {
        "total_records": len(df),
        "total_species": df["species_name"].nunique() if "species_name" in df else 0,
        "total_regions": df["region"].nunique() if "region" in df else 0,
        "categories": df["category"].value_counts().to_dict() if "category" in df else {},
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_stats": df.describe().to_dict()
    }
    return summary

def prepare_features(df):
    """
    Preprocesses DataFrame into feature matrix X and target array y.
    Encodes categorical features and saves label encodings.
    
    Returns:
        X (np.ndarray): Processed feature values.
        y (np.ndarray): Target labels.
        feature_names (list): List of feature names matching columns in X.
        encoders (dict): Dictionary storing encoders and class order mappings.
    """
    data = df.copy()

    # Drop any null rows if present
    data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    # Category encoding (Map to fixed integer order for reproducibility)
    category_map = {cat: idx for idx, cat in enumerate(CATEGORY_ORDER)}
    data["category_encoded"] = data["category"].map(category_map).fillna(0).astype(int)

    # Protected area binary encoding (Yes -> 1, No -> 0)
    data["protected_encoded"] = data["protected_area"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Define model features
    model_feature_names = [
        "category_encoded",
        "population_count",
        "habitat_loss_percent",
        "avg_temperature",
        "human_activity_index",
        "protected_encoded"
    ]

    X = data[model_feature_names].values

    # Target Label Encoding
    label_encoder = LabelEncoder()
    # Fit strictly on predefined classes so class ordering is guaranteed
    label_encoder.fit(STATUS_ORDER)
    y = label_encoder.transform(data[TARGET_COLUMN])

    encoders = {
        "category_map": category_map,
        "status_classes": list(label_encoder.classes_),
        "label_encoder": label_encoder,
        "feature_names": [
            "Category",
            "Population Count",
            "Habitat Loss (%)",
            "Avg Temperature (°C)",
            "Human Activity Index",
            "Protected Area"
        ],
        "raw_feature_names": model_feature_names
    }

    return X, y, encoders

def encode_single_input(category, population_count, habitat_loss_percent, 
                        avg_temperature, human_activity_index, protected_area, encoders):
    """
    Transforms user inputs from Streamlit into a 2D numpy array ready for model prediction.
    """
    category_map = encoders.get("category_map", {cat: idx for idx, cat in enumerate(CATEGORY_ORDER)})
    cat_enc = category_map.get(category, 0)
    prot_enc = 1 if str(protected_area).strip().lower() in ["yes", "1", "true"] else 0

    feature_array = np.array([[
        cat_enc,
        float(population_count),
        float(habitat_loss_percent),
        float(avg_temperature),
        float(human_activity_index),
        prot_enc
    ]], dtype=float)

    return feature_array
