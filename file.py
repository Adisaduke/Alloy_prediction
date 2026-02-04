# =====================================
# IMPORTS
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    mean_squared_error, r2_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score
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
# SECTION 2 — REGRESSION (PROPERTY PREDICTION)
# =====================================

X_reg = data.iloc[:, 3:].values
Y_reg = data[["Tensile_Strength", "Melting_Point"]].values

Xr_train, Xr_test, Yr_train, Yr_test = train_test_split(
    X_reg, Y_reg, test_size=0.2, random_state=42
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

print("RF MSE:", mean_squared_error(Yr_test, Yr_pred))
print("RF R2:", r2_score(Yr_test, Yr_pred))

joblib.dump(rf_model, "rf_regression_model.pkl")


# --- Regression Plots ---

for i, label in enumerate(["Tensile Strength", "Melting Point"]):
    plt.scatter(Yr_test[:, i], Yr_pred[:, i], alpha=0.7)
    plt.plot(
        [Yr_test[:, i].min(), Yr_test[:, i].max()],
        [Yr_test[:, i].min(), Yr_test[:, i].max()],
        "r--"
    )
    plt.xlabel(f"Actual {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(f"Actual vs Predicted: {label}")
    plt.show()


# Feature Importance
rf = rf_model.named_steps["model"]
feat_imp = pd.DataFrame({
    "Element": data.columns[3:],
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(feat_imp["Element"], feat_imp["Importance"])
plt.gca().invert_yaxis()
plt.title("Random Forest Feature Importance")
plt.show()


# =====================================
# SECTION 3 — CLASSIFICATION (HIGH PERFORMANCE)
# =====================================

ts_thr = data["Tensile_Strength"].median()
mp_thr = data["Melting_Point"].median()

data["High_Performance"] = (
    (data["Tensile_Strength"] >= ts_thr) &
    (data["Melting_Point"] >= mp_thr)
).astype(int)

X_cls = data.drop(
    columns=["Alloy", "Tensile_Strength", "Melting_Point", "High_Performance"],
    errors="ignore"
)
X_cls = X_cls.select_dtypes(include=[np.number])
y_cls = data["High_Performance"]


# Save numeric feature names used for classifier
feature_names = X_cls.columns.tolist()
joblib.dump(feature_names, "cls_feature_names.pkl")
print("✅ Classifier feature names saved!")


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
print(cm)
print(classification_report(yc_test, yc_pred))
print("ROC AUC:", roc_auc_score(yc_test, yc_prob))

joblib.dump(classifier, "logistic_classifier.pkl")
joblib.dump(scaler_cls, "scaler.pkl")


# Confusion Matrix Plot
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "High"],
            yticklabels=["Low", "High"])
plt.title("Confusion Matrix")
plt.show()


# ROC Curve
fpr, tpr, _ = roc_curve(yc_test, yc_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(yc_test, yc_prob):.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.legend()
plt.title("ROC Curve")
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
plt.title("Top Element Influence (Logistic Regression)")
plt.show()


# =====================================
# SECTION 4 — PCA + K-MEANS (UNSUPERVISED)
# =====================================

X_unsup = X_cls.copy()

scaler_pca = StandardScaler()
X_unsup_scaled = scaler_pca.fit_transform(X_unsup)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_unsup_scaled)

print("Explained Variance:", pca.explained_variance_ratio_.sum())

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

data["Cluster"] = clusters

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA + K-Means Clustering")
plt.show()

print(
    data.groupby("Cluster")[[
        "Tensile_Strength",
        "Melting_Point",
        "High_Performance"
    ]].mean()
)

joblib.dump(pca, "pca_model.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler_pca, "pca_scaler.pkl")

print("✅ ALL MODELS SAVED — PROJECT COMPLETE")
