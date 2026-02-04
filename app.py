# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # =====================================
# # LOAD MODELS
# # =====================================
# rf_model = joblib.load("rf_regression_model.pkl")
# classifier = joblib.load("logistic_classifier.pkl")
# cls_scaler = joblib.load("scaler.pkl")
# pca_scaler = joblib.load("pca_scaler.pkl")
# pca = joblib.load("pca_model.pkl")
# kmeans = joblib.load("kmeans_model.pkl")
# feature_names = joblib.load("cls_feature_names.pkl")

# # =====================================
# # USER INPUT
# # =====================================
# st.title("Alloy Property Predictor")
# Fe = st.number_input("Fe (%)", 0.0, 100.0, 70.0)
# Cr = st.number_input("Cr (%)", 0.0, 100.0, 18.0)
# Ni = st.number_input("Ni (%)", 0.0, 100.0, 8.0)
# C = st.number_input("C (%)", 0.0, 2.0, 0.1)
# user_input = {"Fe": Fe, "Cr": Cr, "Ni": Ni, "C": C}

# # =====================================
# # BUILD FULL INPUT VECTOR
# # =====================================
# X_full = np.zeros((1, len(feature_names)))
# for i, feat in enumerate(feature_names):
#     if feat in user_input:
#         X_full[0, i] = user_input[feat]

# # =====================================
# # SCALE AND PREDICT
# # =====================================
# X_cls_scaled = cls_scaler.transform(X_full)
# X_pca_scaled = pca_scaler.transform(X_full)
# X_pca = pca.transform(X_pca_scaled)

# if st.button("Predict"):
#     # Regression
#     ts, mp = rf_model.predict(X_full)[0]
#     st.success(f"Tensile Strength: {ts:.2f}")
#     st.success(f"Melting Point: {mp:.2f}")

#     # Classification
#     prob = classifier.predict_proba(X_cls_scaled)[0][1]
#     label = "High Performance Alloy" if prob > 0.5 else "Standard Alloy"
#     st.info(label)

#     # Clustering
#     cluster = kmeans.predict(X_pca)[0]
#     st.write(f"Cluster Group: {cluster}")










# ===================================== 
# STREAMLIT APP — HYBRID ALLOY PREDICTOR
# =====================================

import streamlit as st
import joblib
import numpy as np
from PIL import Image

# ---------------------------
# LOAD MODELS
# ---------------------------
rf_model = joblib.load("rf_regression_model.pkl")
classifier = joblib.load("logistic_classifier.pkl")
cls_scaler = joblib.load("scaler.pkl")
pca_scaler = joblib.load("pca_scaler.pkl")
pca = joblib.load("pca_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")

# Feature names used for training classifier
feature_names = [
    "Fe", "Cr", "Ni", "C", "Mn", "Si", "Mo", "Cu", "V", "Nb",
    "Ti", "Al", "Co", "W", "B", "P", "S", "Sn", "Zn", "Pb",
    "As", "Mg", "Zr", "Ce", "La", "Cr2O3", "NiO", "CuO", "MnO", "SiO2"
]

# ---------------------------
# TITLE
# ---------------------------
st.title("🧪 Alloy Hybrid Predictor")
st.markdown(
    "Predicts **Tensile Strength**, **Melting Point**, **Performance Class**, "
    "and **Cluster Group**. Visualizations show the underlying analysis."
)

# ---------------------------
# USER INPUT — MAIN ELEMENTS
# ---------------------------
st.header("Step 1: Enter Main Composition")
main_elements = ["Fe", "Cr", "Ni", "C"]
user_input = {}

for elem in main_elements:
    user_input[elem] = st.number_input(f"{elem} (%)", 0.0, 100.0, 0.0)

# Optional: Other elements
with st.expander("Advanced: Enter Other Elements (Optional)"):
    for elem in feature_names:
        if elem not in main_elements:
            user_input[elem] = st.number_input(f"{elem} (%)", 0.0, 100.0, 0.0)

# Build full input array
X_full = np.zeros((1, len(feature_names)))
for i, feat in enumerate(feature_names):
    X_full[0, i] = user_input.get(feat, 0.0)

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Predict"):

    # --- Regression: Tensile & Melting ---
    ts, mp = rf_model.predict(X_full)[0]  # FULL 30 features
    st.subheader("🔹 Regression Results")
    st.success(f"Tensile Strength: {ts:.2f} psi")
    st.success(f"Melting Point: {mp:.2f} °C")

    # --- Classification ---
    X_cls_scaled = cls_scaler.transform(X_full)
    prob = classifier.predict_proba(X_cls_scaled)[0][1]
    label = "High Performance Alloy" if prob > 0.5 else "Standard Alloy"
    st.subheader("🔹 Performance Prediction")
    st.info(f"{label} (Probability: {prob:.2f})")

    # --- Clustering ---
    X_pca_scaled = pca_scaler.transform(X_full)
    X_pca = pca.transform(X_pca_scaled)
    cluster = kmeans.predict(X_pca)[0]
    st.subheader("🔹 Cluster Assignment")
    st.write(f"Cluster Group: {cluster}")


    # ---------------------------
    # SHOW VISUALIZATIONS
    # ---------------------------
    st.header("Step 2: Visualizations")

    # Feature Importance
    st.subheader("Feature Importance (Random Forest)")
    feat_img = Image.open("plots/feature_importance.png")
    st.image(feat_img)

    # Regression Plots
    st.subheader("Regression: Actual vs Predicted")
    reg_ts_img = Image.open("plots/regression_plot_ts.png")
    st.image(reg_ts_img)
    reg_mp_img = Image.open("plots/regression_plot_mel.png")
    st.image(reg_mp_img)

    # Confusion Matrix
    st.subheader("Classification: Confusion Matrix")
    cm_img = Image.open("plots/confusion_matrix.png")
    st.image(cm_img)

    # ROC Curve
    st.subheader("Classification: ROC Curve")
    roc_img = Image.open("plots/roc_curve.png")
    st.image(roc_img)

    # PCA Clustering
    st.subheader("Clustering: PCA + K-Means")
    cluster_img = Image.open("plots/pca_cluster.png")
    st.image(cluster_img)

    # Top Logistic Regression Coefficients
    st.subheader("Top Logistic Regression Features")
    top_coef_img = Image.open("plots/Top_element_Logistic_regression.png")
    st.image(top_coef_img)

st.markdown("---")
st.caption(
    "Note: All models are pre-trained. Input values for elements not entered default to zero."
)
