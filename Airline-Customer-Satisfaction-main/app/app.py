# ======================================
# Fix Python path for src/ imports
# ======================================
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ======================================
# Imports
# ======================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ======================================
# Page Configuration
# ======================================
st.set_page_config(
    page_title="Airline Customer Satisfaction ML App",
    page_icon="✈️",
    layout="wide"
)

# ======================================
# Load Model & Pipeline
# ======================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgboost_best_model.joblib")
    pipeline = joblib.load("models/feature_pipeline.joblib")
    return model, pipeline

model, pipeline = load_artifacts()

# ======================================
# SHAP Explainer (FIXED)
# ======================================
@st.cache_resource
def load_shap_explainer(_model):
    """
    TreeExplainer is the correct explainer for XGBoost
    `_model` is ignored by Streamlit hashing
    """
    explainer = shap.TreeExplainer(_model)
    return explainer

explainer = load_shap_explainer(model)

# ======================================
# Sidebar Navigation
# ======================================
st.sidebar.title("✈️ Airline ML App")
page = st.sidebar.radio(
    "Choose Mode",
    ["Single Prediction", "Batch Prediction"]
)

st.sidebar.markdown("---")
# ======================================
# Main Title
# ======================================
st.title("✈️ Airline Customer Satisfaction Prediction")

# ======================================================
# 1️⃣ SINGLE PREDICTION + SHAP
# ======================================================
if page == "Single Prediction":

    st.subheader("🧍 Single Passenger Prediction")

    with st.form("single_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            customer_type = st.selectbox(
                "Customer Type", ["Loyal Customer", "disloyal Customer"]
            )
            age = st.slider("Age", 5, 85, 30)

        with col2:
            travel_type = st.selectbox(
                "Type of Travel", ["Business travel", "Personal Travel"]
            )
            travel_class = st.selectbox(
                "Class", ["Eco", "Eco Plus", "Business"]
            )
            flight_distance = st.slider(
                "Flight Distance (km)", 50, 5000, 1000
            )

        with col3:
            dep_delay = st.number_input(
                "Departure Delay (min)", 0, 500, 0
            )
            arr_delay = st.number_input(
                "Arrival Delay (min)", 0, 500, 0
            )

        st.markdown("### ⭐ Service Ratings (0–5)")

        service_features = [
            "Seat comfort", "Food and drink", "Inflight wifi service",
            "Inflight entertainment", "Online support", "Ease of Online booking",
            "On-board service", "Leg room service", "Baggage handling",
            "Checkin service", "Cleanliness", "Online boarding",
            "Gate location", "Departure/Arrival time convenient"
        ]

        ratings = {}
        cols = st.columns(4)
        for i, f in enumerate(service_features):
            ratings[f] = cols[i % 4].slider(f, 0, 5, 3)

        submit_single = st.form_submit_button("🔮 Predict")

    if submit_single:
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Customer Type": customer_type,
            "Age": age,
            "Type of Travel": travel_type,
            "Class": travel_class,
            "Flight Distance": flight_distance,
            "Departure Delay in Minutes": dep_delay,
            "Arrival Delay in Minutes": arr_delay,
            **ratings,
        }])

        X_tr = pipeline.transform(input_df)

        # ---- Correct prediction ----
        pred = model.predict(X_tr)[0]
        class_idx = list(model.classes_).index(1)
        prob = model.predict_proba(X_tr)[0][class_idx]

        st.markdown("---")
        if pred == 1:
            st.success(f"✅ **Satisfied Customer**  \nConfidence: {prob:.2%}")
        else:
            st.error(f"❌ **Dissatisfied Customer**  \nConfidence: {(1 - prob):.2%}")

        # ---------- SHAP ----------
        st.markdown("### 🔍 Why this prediction?")
        shap_values = explainer.shap_values(X_tr)

        fig, ax = plt.subplots()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_tr[0]
            ),
            show=False
        )
        st.pyplot(fig)

# ======================================================
# 2️⃣ BATCH PREDICTION
# ======================================================
else:
    st.subheader("📂 Batch Prediction (CSV Upload)")

    st.markdown(
        """
        Upload a CSV file with the **same columns as training data**.
        The app will return predictions and probabilities.
        """
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)

        X_batch = pipeline.transform(batch_df)

        preds = model.predict(X_batch)
        class_idx = list(model.classes_).index(1)
        probs = model.predict_proba(X_batch)[:, class_idx]

        batch_df["prediction"] = np.where(preds == 1, "Satisfied", "Dissatisfied")
        batch_df["satisfaction_probability"] = probs

        st.success("Batch prediction completed!")
        st.dataframe(batch_df.head(20))

        csv = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions",
            csv,
            "airline_predictions.csv",
            "text/csv"
        )

# ======================================
# Footer
# ======================================
st.markdown("---")
st.caption("🚀 End-to-End ML App | XGBoost | SHAP | Streamlit")