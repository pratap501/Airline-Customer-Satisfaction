"""
SHAP Explainability Script
Airline Customer Satisfaction
Explains the best-performing model (XGBoost)
"""

import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from src.features.build_features import load_and_prepare_data

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -------------------------
# Paths
# -------------------------
DATA_PATH = "data/Invistico_Airline.csv"
MODEL_PATH = "models/xgboost_best_model.joblib"
PIPELINE_PATH = "models/feature_pipeline.joblib"
OUTPUT_DIR = "reports/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# 1. Load model & pipeline
# -------------------------
logging.info("Loading model and feature pipeline...")

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

# -------------------------
# 2. Load data
# -------------------------
X, y = load_and_prepare_data(DATA_PATH)

# Use a sample for SHAP (industry standard for speed)
X_sample = X.sample(n=3000, random_state=42)

X_sample_transformed = pipeline.transform(X_sample)

logging.info(f"SHAP sample size: {X_sample_transformed.shape}")

# -------------------------
# 3. Feature names (CRITICAL)
# -------------------------
feature_names = [
    "Age",
    "Flight Distance",
    "Comfort Score",
    "Digital Experience Score",
    "Inflight Experience Score",
    "Ground Service Score",
    "Log Departure Delay",
    "Log Arrival Delay",
    "Has Departure Delay",
    "Has Arrival Delay",
]

# -------------------------
# 4. SHAP Explainer
# -------------------------
logging.info("Initializing SHAP explainer...")

explainer = shap.Explainer(
    model,
    X_sample_transformed,
    feature_names=feature_names
)

shap_values = explainer(X_sample_transformed)

# -------------------------
# 5. Global Feature Importance
# -------------------------
logging.info("Generating SHAP summary plot...")

plt.figure()
shap.summary_plot(
    shap_values,
    features=X_sample_transformed,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=300)
plt.close()

# -------------------------
# 6. Bar Plot (Executive Friendly)
# -------------------------
logging.info("Generating SHAP bar plot...")

plt.figure()
shap.summary_plot(
    shap_values,
    features=X_sample_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_bar.png", dpi=300)
plt.close()

# -------------------------
# 7. Dependence Plots (Key Drivers)
# -------------------------
top_features = [
    "Comfort Score",
    "Digital Experience Score",
    "Inflight Experience Score",
    "Log Arrival Delay",
]

for feature in top_features:
    logging.info(f"Generating dependence plot for {feature}")

    plt.figure()
    shap.dependence_plot(
        feature,
        shap_values.values,
        X_sample_transformed,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dependence_{feature.replace(' ', '_')}.png", dpi=300)
    plt.close()

logging.info("SHAP explainability completed successfully")
