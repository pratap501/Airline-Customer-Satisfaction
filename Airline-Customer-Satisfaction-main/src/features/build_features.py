"""
Feature Engineering Pipeline for Airline Customer Satisfaction
Author: You
Purpose: Production-ready feature creation & preprocessing
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# =========================
# 1. Custom Transformers
# =========================

class ServiceScoreEngineer(BaseEstimator, TransformerMixin):
    """
    Creates composite service experience scores
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Composite scores
        X["comfort_score"] = X[["Seat comfort", "Leg room service"]].mean(axis=1)

        X["digital_score"] = X[
            [
                "Online boarding",
                "Online support",
                "Ease of Online booking",
                "Inflight wifi service",
            ]
        ].mean(axis=1)

        X["inflight_score"] = X[
            [
                "Food and drink",
                "Inflight entertainment",
                "Cleanliness",
                "On-board service",
            ]
        ].mean(axis=1)

        X["ground_score"] = X[
            [
                "Gate location",
                "Checkin service",
                "Baggage handling",
            ]
        ].mean(axis=1)

        return X


class DelayFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Handles skewed and zero-inflated delay features
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Fill missing delays with 0 (industry assumption)
        X["Departure Delay in Minutes"] = X["Departure Delay in Minutes"].fillna(0)
        X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(0)

        # Binary flags
        X["has_departure_delay"] = (X["Departure Delay in Minutes"] > 0).astype(int)
        X["has_arrival_delay"] = (X["Arrival Delay in Minutes"] > 0).astype(int)

        # Log transform for skewness
        X["log_departure_delay"] = np.log1p(X["Departure Delay in Minutes"])
        X["log_arrival_delay"] = np.log1p(X["Arrival Delay in Minutes"])

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical variables with domain knowledge
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Binary encoding
        X["Gender"] = X["Gender"].map({"Female": 0, "Male": 1})
        X["Customer Type"] = X["Customer Type"].map(
            {"disloyal Customer": 0, "Loyal Customer": 1}
        )
        X["Type of Travel"] = X["Type of Travel"].map(
            {"Personal Travel": 0, "Business travel": 1}
        )

        # Ordinal encoding
        X["Class"] = X["Class"].map(
            {"Eco": 0, "Eco Plus": 1, "Business": 2}
        )

        return X


# =========================
# 2. Feature Engineering Pipeline
# =========================

def build_feature_pipeline():
    """
    Full feature engineering pipeline
    """

    numeric_features = [
        "Age",
        "Flight Distance",
        "comfort_score",
        "digital_score",
        "inflight_score",
        "ground_score",
        "log_departure_delay",
        "log_arrival_delay",
        "has_departure_delay",
        "has_arrival_delay",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features)
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline(
        steps=[
            ("service_features", ServiceScoreEngineer()),
            ("delay_features", DelayFeatureEngineer()),
            ("categorical_encoder", CategoricalEncoder()),
            ("preprocessor", preprocessor),
        ]
    )

    return full_pipeline


# =========================
# 3. Data Loading Utility
# =========================

def load_and_prepare_data(csv_path):
    """
    Loads data and separates target
    """
    df = pd.read_csv(csv_path)

    y = df["satisfaction"].map(
        {"dissatisfied": 0, "satisfied": 1}
    )

    X = df.drop(columns=["satisfaction"])

    return X, y
