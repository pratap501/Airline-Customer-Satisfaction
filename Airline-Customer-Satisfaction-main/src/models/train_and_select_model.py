"""
Model Training & Selection Script (Production Safe)
Airline Customer Satisfaction
"""

# =========================
# 0. Imports & Logging
# =========================

import time
import logging
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Optional advanced models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


from src.features.build_features import (
    build_feature_pipeline,
    load_and_prepare_data,
)

# -------------------------
# Logging Configuration
# -------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# =========================
# 1. Load Data
# =========================

logging.info("Loading dataset...")
X, y = load_and_prepare_data("data/Invistico_Airline.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

logging.info("Dataset loaded and split completed")

# =========================
# 2. Feature Engineering
# =========================

logging.info("Building feature pipeline...")
feature_pipeline = build_feature_pipeline()

X_train_tr = feature_pipeline.fit_transform(X_train)
X_test_tr = feature_pipeline.transform(X_test)

logging.info(f"Feature matrix shape: {X_train_tr.shape}")

# =========================
# 3. Define Models (SAFE)
# =========================

models = {
    "LogisticRegression": LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
    ),

    # Linear SVM replacement (scales well)
    "LinearSVM": SGDClassifier(
        loss="hinge",
        max_iter=2000,
        tol=1e-3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=15,
        weights="distance",
        algorithm="auto",
    ),

    "DecisionTree": DecisionTreeClassifier(
        max_depth=12,
        class_weight="balanced",
        random_state=42,
    ),

    # Reduced size RF (laptop-safe)
    "RandomForest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

if LIGHTGBM_AVAILABLE:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

# =========================
# 4. Train & Evaluate
# =========================

results = []

for name, model in models.items():
    logging.info(f"Training {name} started...")
    start_time = time.time()

    model.fit(X_train_tr, y_train)

    train_time = time.time() - start_time
    logging.info(f"Training {name} finished in {train_time:.2f} seconds")

    y_pred = model.predict(X_test_tr)

    # Probability handling (some models don’t support predict_proba)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_tr)[:, 1]
    else:
        scores = model.decision_function(X_test_tr)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min())

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Train_Time_sec": round(train_time, 2),
    }

    results.append(metrics)

# =========================
# 5. Model Comparison
# =========================

results_df = pd.DataFrame(results).sort_values(
    by=["F1", "ROC_AUC"],
    ascending=False,
)

logging.info("\nMODEL COMPARISON TABLE\n")
logging.info("\n" + results_df.to_string(index=False))

# =========================
# 6. Select Best Model
# =========================

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

logging.info(f"BEST MODEL SELECTED: {best_model_name}")

# =========================
# 7. Stress Testing
# =========================

def stress_test(model, X_test, y_test):
    """
    Robust stress testing across different data slices
    """

    scenarios = {
        "All Data": np.ones(len(X_test), dtype=bool),
        "Long Flights": X_test["Flight Distance"] > 1500,
        "Short Flights": X_test["Flight Distance"] <= 1500,
        "Business Travel": X_test["Type of Travel"] == "Business travel",
        "Personal Travel": X_test["Type of Travel"] == "Personal Travel",
    }

    logging.info("Running stress tests...")

    for name, mask in scenarios.items():
        # mask is now ALWAYS a boolean array / Series
        sample_size = mask.sum()

        if sample_size < 200:
            logging.info(f"{name} | Skipped (only {sample_size} samples)")
            continue

        X_sub = X_test.loc[mask]
        y_sub = y_test.loc[mask]

        X_sub_tr = feature_pipeline.transform(X_sub)

        y_pred = model.predict(X_sub_tr)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_sub_tr)[:, 1]
        else:
            scores = model.decision_function(X_sub_tr)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min())

        logging.info(
            f"{name} | Samples={sample_size} "
            f"| F1={f1_score(y_sub, y_pred):.3f} "
            f"| ROC_AUC={roc_auc_score(y_sub, y_prob):.3f}"
        )

stress_test(best_model, X_test, y_test)

# =========================
# 8. Save Best Model
# =========================

joblib.dump(best_model, f"models/{best_model_name.lower()}_best_model.joblib")
joblib.dump(feature_pipeline, "models/feature_pipeline.joblib")

logging.info("Best model and feature pipeline saved successfully")
