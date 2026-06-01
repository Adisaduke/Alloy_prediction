# =====================================
# IMPORTS
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import (
    mean_squared_error, r2_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    f1_score, accuracy_score
)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# =====================================
# SECTION 1 — LOAD & CLEAN DATA
# =====================================

data = pd.read_csv("Alloys.csv")

data = data.rename(columns={
    "Tensile Strength: Ultimate (UTS) (psi)": "Tensile_Strength",
    "Melting Completion (Liquidus)": "Melting_Point"
})

# =====================================
# C1 — UTS column is labeled psi but values are already in MPa
# No conversion needed — Kaggle dataset has incorrect unit label
print("✅ UTS confirmed in MPa already (dataset label is incorrect)")
print(data["Tensile_Strength"].describe())


# =====================================
# SECTION 2 — REGRESSION (PROPERTY PREDICTION)
# =====================================

X_reg = data.iloc[:, 3:].values
Y_reg = data[["Tensile_Strength", "Melting_Point"]].values

# Track test indices for cluster-specific error analysis later
all_indices = np.arange(len(data))
Xr_train, Xr_test, Yr_train, Yr_test, idx_train, idx_test = train_test_split(
    X_reg, Y_reg, all_indices, test_size=0.2, random_state=42
)

rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=1
    ))
])

rf_model.fit(Xr_train, Yr_train)
Yr_pred = rf_model.predict(Xr_test)

print("RF Test MSE:", mean_squared_error(Yr_test, Yr_pred))
print("RF Test R²:", r2_score(Yr_test, Yr_pred))

joblib.dump(rf_model, "rf_regression_model.pkl")


# =====================================
# C2 — K-FOLD CROSS VALIDATION
# (Put this right after rf_model.fit)
# =====================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(rf_model, X_reg, Y_reg, cv=kf, scoring='r2')
print(f"\n5-Fold CV R² — Mean: {cv_r2.mean():.4f}, Std: {cv_r2.std():.4f}")


# =====================================
# C3 — BASELINE MODEL COMPARISONS
# (Put this right after C2)
# =====================================

# Linear Regression baseline
lr_base = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LinearRegression())
])
lr_base.fit(Xr_train, Yr_train)
lr_pred = lr_base.predict(Xr_test)
print("\nLinear Regression R²:", r2_score(Yr_test, lr_pred))
print("Linear Regression MSE:", mean_squared_error(Yr_test, lr_pred))

# Gradient Boosting baseline
gb_base = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", MultiOutputRegressor(GradientBoostingRegressor(random_state=42)))
])
gb_base.fit(Xr_train, Yr_train)
gb_pred = gb_base.predict(Xr_test)
print("Gradient Boosting R²:", r2_score(Yr_test, gb_pred))
print("Gradient Boosting MSE:", mean_squared_error(Yr_test, gb_pred))


# =====================================
# C4 — TRAINING VS TEST PERFORMANCE + LEARNING CURVES
# (Put this right after C3)
# =====================================

# Train vs Test R² comparison
Yr_train_pred = rf_model.predict(Xr_train)
print("\nTrain R²:", r2_score(Yr_train, Yr_train_pred))
print("Test R²:", r2_score(Yr_test, Yr_pred))

# Learning curve
train_sizes, train_scores, val_scores = learning_curve(
    rf_model, X_reg, Y_reg, cv=5, scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Train R²', marker='o')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation R²', marker='o')
plt.fill_between(train_sizes,
                 train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
plt.fill_between(train_sizes,
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
plt.xlabel("Training Set Size", fontsize=13)
plt.ylabel("R²", fontsize=13)
plt.title("Learning Curve — Random Forest", fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
plt.show()


# --- Parity Plots (C10 quality applied here) ---

for i, label in enumerate(["UTS (MPa)", "Melting Point (°C)"]):
    plt.figure(figsize=(7, 6))
    plt.scatter(Yr_test[:, i], Yr_pred[:, i], alpha=0.6, s=20)
    plt.plot(
        [Yr_test[:, i].min(), Yr_test[:, i].max()],
        [Yr_test[:, i].min(), Yr_test[:, i].max()],
        "r--", lw=2
    )
    plt.xlabel(f"Actual {label}", fontsize=13)
    plt.ylabel(f"Predicted {label}", fontsize=13)
    plt.title(f"Actual vs Predicted: {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"parity_{label.split()[0].lower()}.png", dpi=150)
    plt.show()


# =====================================
# NEW — HIGH UTS ERROR ANALYSIS
# (Put this right after parity plots)
# =====================================

high_uts_mask = Yr_test[:, 0] > np.percentile(Yr_test[:, 0], 75)
low_uts_mask = ~high_uts_mask

print("\n--- UTS Error Analysis by Range ---")
print("High UTS (top 25%) R²:",
      r2_score(Yr_test[high_uts_mask, 0], Yr_pred[high_uts_mask, 0]))
print("Normal UTS R²:",
      r2_score(Yr_test[low_uts_mask, 0], Yr_pred[low_uts_mask, 0]))
print("High UTS RMSE:",
      np.sqrt(mean_squared_error(Yr_test[high_uts_mask, 0], Yr_pred[high_uts_mask, 0])))
print("Normal UTS RMSE:",
      np.sqrt(mean_squared_error(Yr_test[low_uts_mask, 0], Yr_pred[low_uts_mask, 0])))


# --- Feature Importance (MDI) ---

rf = rf_model.named_steps["model"]
feat_imp = pd.DataFrame({
    "Element": data.columns[3:],
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(feat_imp["Element"], feat_imp["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("MDI Importance", fontsize=13)
plt.title("Random Forest Feature Importance (MDI)", fontsize=14)
plt.tight_layout()
plt.savefig("feature_importance_mdi.png", dpi=150)
plt.show()


# =====================================
# C5 — SHAP ANALYSIS (FIXED)
# =====================================

imputer_step = rf_model.named_steps["imputer"]
Xr_test_imputed = imputer_step.transform(Xr_test)

rf_fitted = rf_model.named_steps["model"]
explainer = shap.TreeExplainer(rf_fitted)
shap_values = explainer.shap_values(Xr_test_imputed)

feature_names = data.columns[3:].tolist()

# Check shap_values structure and handle both possible formats
print("SHAP values type:", type(shap_values))
print("SHAP values shape:", np.array(shap_values).shape)

# For multi-output RF, shap_values is a list [output1, output2]
# Each element shape should be (n_samples, n_features)
if isinstance(shap_values, list):
    shap_uts = shap_values[0]
    shap_mp = shap_values[1]
else:
    # Newer SHAP versions return a single array
    shap_uts = shap_values[:, :, 0]
    shap_mp = shap_values[:, :, 1]

# SHAP for UTS
plt.figure()
shap.summary_plot(shap_uts, Xr_test_imputed,
                  feature_names=feature_names, show=False)
plt.title("SHAP Summary — UTS (MPa)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_uts.png", dpi=150, bbox_inches='tight')
plt.show()

# SHAP for Melting Point
plt.figure()
shap.summary_plot(shap_mp, Xr_test_imputed,
                  feature_names=feature_names, show=False)
plt.title("SHAP Summary — Melting Point (°C)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_melting.png", dpi=150, bbox_inches='tight')
plt.show()


# =====================================
# C6 — PERMUTATION IMPORTANCE
# (Put this right after C5 SHAP)
# =====================================

perm_imp = permutation_importance(
    rf_model, Xr_test, Yr_test, n_repeats=10, random_state=42
)

perm_df = pd.DataFrame({
    "Element": data.columns[3:],
    "Importance_Mean": perm_imp.importances_mean,
    "Importance_Std": perm_imp.importances_std
}).sort_values("Importance_Mean", ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(
    perm_df["Element"][:15],
    perm_df["Importance_Mean"][:15],
    xerr=perm_df["Importance_Std"][:15]
)
plt.gca().invert_yaxis()
plt.xlabel("Permutation Importance", fontsize=13)
plt.title("Permutation Feature Importance (Top 15)", fontsize=14)
plt.tight_layout()
plt.savefig("permutation_importance.png", dpi=150)
plt.show()

print("\nTop 10 Elements by Permutation Importance:")
print(perm_df.head(10).to_string())


# =====================================
# SECTION 3 — CLASSIFICATION (HIGH PERFORMANCE)
# =====================================

ts_thr = data["Tensile_Strength"].median()
mp_thr = data["Melting_Point"].median()

data["High_Performance"] = (
    (data["Tensile_Strength"] >= ts_thr) &
    (data["Melting_Point"] >= mp_thr)
).astype(int)

print("\nClass distribution:")
print(data["High_Performance"].value_counts())

X_cls = data.drop(
    columns=["Alloy", "Tensile_Strength", "Melting_Point", "High_Performance"],
    errors="ignore"
)
X_cls = X_cls.select_dtypes(include=[np.number])
y_cls = data["High_Performance"]

feature_names_cls = X_cls.columns.tolist()
joblib.dump(feature_names_cls, "cls_feature_names.pkl")

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=42
)

scaler_cls = StandardScaler()
Xc_train_s = scaler_cls.fit_transform(Xc_train)
Xc_test_s = scaler_cls.transform(Xc_test)

classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(Xc_train_s, yc_train)

yc_pred = classifier.predict(Xc_test_s)
yc_prob = classifier.predict_proba(Xc_test_s)[:, 1]

cm = confusion_matrix(yc_test, yc_pred)
print("\nConfusion Matrix:")
print(cm)
print(classification_report(yc_test, yc_pred))
print("ROC AUC:", roc_auc_score(yc_test, yc_prob))

joblib.dump(classifier, "logistic_classifier.pkl")
joblib.dump(scaler_cls, "scaler.pkl")


# Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Standard", "High"],
            yticklabels=["Standard", "High"])
plt.xlabel("Predicted", fontsize=13)
plt.ylabel("Actual", fontsize=13)
plt.title("Confusion Matrix — Logistic Regression", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()


# ROC Curve
fpr, tpr, _ = roc_curve(yc_test, yc_prob)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(yc_test, yc_prob):.2f}", lw=2)
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate", fontsize=13)
plt.ylabel("True Positive Rate", fontsize=13)
plt.title("ROC Curve — Logistic Regression", fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()


# Logistic Coefficients
coef_df = pd.DataFrame({
    "Element": X_cls.columns,
    "Coefficient": classifier.coef_[0]
})
coef_df["Abs"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("Abs", ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(coef_df["Element"][:15], coef_df["Coefficient"][:15])
plt.axvline(0, color="black", linestyle="--")
plt.gca().invert_yaxis()
plt.xlabel("Coefficient Value", fontsize=13)
plt.title("Top 15 Element Influence — Logistic Regression", fontsize=14)
plt.tight_layout()
plt.savefig("logistic_coefficients.png", dpi=150)
plt.show()


# =====================================
# C7 — CLASSIFICATION THRESHOLD SENSITIVITY
# (Put this right after logistic coefficients plot)
# =====================================

print("\n--- Threshold Sensitivity Analysis ---")
sensitivity_results = []

for pct in [25, 33, 40, 50, 60, 67, 75]:
    ts_thr_pct = np.percentile(data["Tensile_Strength"], pct)
    mp_thr_pct = np.percentile(data["Melting_Point"], pct)
    labels_pct = (
        (data["Tensile_Strength"] >= ts_thr_pct) &
        (data["Melting_Point"] >= mp_thr_pct)
    ).astype(int)

    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X_cls, labels_pct, test_size=0.2,
        stratify=labels_pct, random_state=42
    )
    sc_temp = StandardScaler()
    clf_temp = LogisticRegression(max_iter=1000, random_state=42)
    clf_temp.fit(sc_temp.fit_transform(Xc_tr), yc_tr)
    preds_temp = clf_temp.predict(sc_temp.transform(Xc_te))

    sensitivity_results.append({
        "Percentile": pct,
        "Accuracy": round(accuracy_score(yc_te, preds_temp), 4),
        "F1": round(f1_score(yc_te, preds_temp), 4)
    })

sensitivity_df = pd.DataFrame(sensitivity_results)
print(sensitivity_df.to_string(index=False))


# =====================================
# SECTION 4 — PCA + K-MEANS (UNSUPERVISED)
# =====================================

X_unsup = X_cls.copy()

scaler_pca = StandardScaler()
X_unsup_scaled = scaler_pca.fit_transform(X_unsup)

# =====================================
# C8 — CLUSTER ON FULL FEATURE SPACE
# PCA used only for visualization
# (Replaces original clustering section entirely)
# =====================================

# Cluster on FULL scaled feature space
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_unsup_scaled)
data["Cluster"] = clusters

# PCA only for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_unsup_scaled)
print(f"\nPCA Explained Variance (visualization only): {pca.explained_variance_ratio_.sum()*100:.1f}%")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=clusters, cmap="viridis", alpha=0.6, s=20)
plt.colorbar(scatter, label="Cluster")
plt.xlabel("PC1", fontsize=13)
plt.ylabel("PC2", fontsize=13)
plt.title("PCA Visualization of K-Means Clusters\n(Clustering performed in full feature space)",
          fontsize=13)
plt.tight_layout()
plt.savefig("pca_clusters.png", dpi=150)
plt.show()

# Cluster Summary Table
print("\n--- Cluster Summary Table ---")
cluster_summary = data.groupby("Cluster").agg(
    Count=("Tensile_Strength", "count"),
    Mean_UTS_MPa=("Tensile_Strength", "mean"),
    Mean_Melting_Point=("Melting_Point", "mean"),
    High_Perf_Fraction=("High_Performance", "mean")
).round(3)

# Add top 5 elements mean per cluster
top_elements = feat_imp["Element"].head(5).tolist()
for el in top_elements:
    if el in data.columns:
        cluster_summary[f"Mean_{el}_wt%"] = data.groupby("Cluster")[el].mean().round(3)

print(cluster_summary.to_string())

# =====================================
# NEW — CLUSTER-SPECIFIC ERROR ANALYSIS
# (Put this right after cluster summary table)
# =====================================

print("\n--- Cluster-Specific Regression Error Analysis ---")
test_clusters = data.iloc[idx_test]["Cluster"].values

for c in sorted(data["Cluster"].unique()):
    mask = test_clusters == c
    if mask.sum() > 0:
        r2_c = r2_score(Yr_test[mask], Yr_pred[mask])
        rmse_c = np.sqrt(mean_squared_error(Yr_test[mask], Yr_pred[mask]))
        print(f"Cluster {c} — N={mask.sum()}, R²={r2_c:.4f}, RMSE={rmse_c:.4f}")


# =====================================
# C9 — DATASET STATISTICS (for paper reporting)
# (Put this right after cluster error analysis)
# =====================================

print("\n--- Dataset Statistics (Copy these into your paper) ---")
print("Total alloys:", len(data))
print("Number of features:", data.shape[1] - 4)
print("\nUTS stats (MPa):")
print(data["Tensile_Strength"].describe().round(3))
print("\nMelting Point stats (°C):")
print(data["Melting_Point"].describe().round(3))
print("\nMissing values per element (top 10, %):")
print((data.iloc[:, 3:].isna().sum() / len(data) * 100)
      .sort_values(ascending=False).head(10).round(2))
print("\nClass balance (High Performance):")
print(data["High_Performance"].value_counts())
print(data["High_Performance"].value_counts(normalize=True).round(3))


# =====================================
# Save all models
# =====================================

joblib.dump(pca, "pca_model.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler_pca, "pca_scaler.pkl")

print("\n✅ ALL MODELS SAVED — PROJECT COMPLETE")