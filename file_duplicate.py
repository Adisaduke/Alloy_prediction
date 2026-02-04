import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans







data = pd.read_csv('Alloys.csv')


data = data.rename(columns={
    "Tensile Strength: Ultimate (UTS) (psi)":
    "Tensile_Strength",
    "Melting Completion (Liquidus)": "Melting_Point"
})


X = data.iloc[:, 3:].values
Y = data.iloc[:, 1:3].values

X_train, X_test, Y_train, Y_test = train_test_split(X ,Y, random_state=42, test_size=0.2)


rf_model = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=1
    ))
])


rf_model.fit(X_train, Y_train)

y_pred_rf = rf_model.predict(X_test)

print("RF MSE:", mean_squared_error(Y_test, y_pred_rf))
print("RF R2:", r2_score(Y_test, y_pred_rf))

import joblib

joblib.dump(rf_model, "rf_regression_model.pkl")
print("✅ Regression model saved")


# Tensile Strength
plt.scatter(Y_test[:,0], y_pred_rf[:,0], alpha=0.7)
plt.plot([Y[:,0].min(), Y_test[:,0].max()],
         [Y_test[:,0].min(), Y_test[:,0].max()],
         'r--')
plt.xlabel("Actual Tensile Strength")
plt.ylabel("Predicted Tensile Strength")
plt.title("Actual vs Predicted: Tensile Strength")
plt.show()


# Melting Point
plt.scatter(Y_test[:,1], y_pred_rf[:,1], alpha=0.7)
plt.plot([Y_test[:,1].min(), Y_test[:,1].max()],
         [Y_test[:,1].min(), Y_test[:,1].max()],
         'r--')
plt.xlabel("Actual Melting Point")
plt.ylabel("Predicted Melting Point")
plt.title("Actual vs Predicted: Melting Point")
plt.show()


# Residual = Actual - Predicted
residuals_ts = Y_test[:,0] - y_pred_rf[:,0] 
residuals_mp = Y_test[:,1] - y_pred_rf[:,1] 


# Tensile Strength Residuals
plt.scatter(Y_test[:,0], residuals_ts, alpha=0.7)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Actual Tensile Strength")
plt.ylabel("Residuals")
plt.title("Residual Plot: Tensile Strength")
plt.show()


# Melting Point Residuals
plt.scatter(Y_test[:,1], residuals_mp, alpha=0.7)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Actual Melting Point")
plt.ylabel("Residuals")
plt.title("Residual Plot: Melting Point")
plt.show()


# Tensile Strength Errors
plt.hist(residuals_ts, bins=20, alpha=0.7, color='blue')
plt.xlabel("Error (MPa)")
plt.ylabel("Frequency")
plt.title("Error Distribution: Tensile Strength")
plt.show()


# Melting Point Errors
plt.hist(residuals_mp, bins=20, alpha=0.7, color='green')
plt.xlabel("Error (°C)")
plt.ylabel("Frequency")
plt.title("Error Distribution: Melting Point")
plt.show()


# Get feature importance (ONLY FIXED PART)
rf = rf_model.named_steps['model']
importances = rf.feature_importances_

# Get correct feature names from original DataFrame
feature_names = data.columns[3:]

# Combine into DataFrame
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(8, 5))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='orange')
plt.xlabel("Importance")
plt.ylabel("Element")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()






# # Classification
ts_threshold = data["Tensile_Strength"].median()
mp_threshold = data["Melting_Point"].median()

data["High_Performance"] = (
    (data["Tensile_Strength"] >= ts_threshold) &
    (data["Melting_Point"] >= mp_threshold)
).astype(int)

# features composition
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

composition_columns = [
    col for col in numeric_cols
    if col not in ["Tensile_Strength", "Melting_Point", "High_Performance"]
]

X = data.drop(
    columns=["Tensile_Strength", "Melting_Point", "High_Performance"]
)
X = X.select_dtypes(include=[np.number])
y = data["High_Performance"]




# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

classifier = LogisticRegression(
    max_iter=1000,
    random_state=42
)

classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)
y_prob = classifier.predict_proba(X_test_scaled)[:, 1]
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", roc_auc)

joblib.dump(classifier, "logistic_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Classification model and scaler saved")

coefficients = classifier.coef_[0]
feature_importance = pd.DataFrame({
    "Element": composition_columns,
    "Coefficient": coefficients
})

feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values(
    by="Abs_Coefficient",
    ascending=False
)





# 2️⃣ Rename columns if necessary (optional)
data = data.rename(columns={
    "Tensile Strength: Ultimate (UTS) (psi)": "Tensile_Strength",
    "Melting Completion (Liquidus)": "Melting_Point"
})

# 3️⃣ Create the target variable
ts_threshold = data["Tensile_Strength"].median()
mp_threshold = data["Melting_Point"].median()

data["High_Performance"] = (
    (data["Tensile_Strength"] >= ts_threshold) &
    (data["Melting_Point"] >= mp_threshold)
).astype(int)

# 4️⃣ Select feature columns (composition elements)
# Drop text and target columns
X = data.drop(
    columns=["Alloy", "Tensile_Strength", "Melting_Point", "High_Performance"]
)
# Keep only numeric columns (just in case)
X = X.select_dtypes(include=[np.number])

# Target
y = data["High_Performance"]

# 5️⃣ Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6️⃣ Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7️⃣ Train Logistic Regression classifier
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train_scaled, y_train)

# 8️⃣ Predictions and probabilities
y_pred = classifier.predict(X_test_scaled)
y_prob = classifier.predict_proba(X_test_scaled)[:, 1]

# 9️⃣ Evaluation metrics
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", roc_auc)

# 🔟 Feature importance (coefficient magnitude)
coefficients = classifier.coef_[0]  # logistic regression coefficients
feature_importance = pd.DataFrame({
    "Element": X.columns,       # use X.columns directly
    "Coefficient": coefficients
})
feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values(
    by="Abs_Coefficient",
    ascending=False
)
print("\nFeature Importance:\n", feature_importance)








plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low", "High"],
    yticklabels=["Low", "High"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: High-Performance Classification")
plt.show()



fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: High-Performance Classification")
plt.legend()
plt.show()




top_features = feature_importance.head(15)

plt.figure(figsize=(8, 5))
plt.barh(
    top_features["Element"],
    top_features["Coefficient"],
    color="purple"
)
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Coefficient Value")
plt.ylabel("Element")
plt.title("Element Influence on High-Performance Classification")
plt.gca().invert_yaxis()
plt.show()



# ============================
# 1️⃣ PREPARE DATA FOR PCA
# ============================

# Use the SAME numeric composition features
X_unsupervised = data.drop(
    columns=["Alloy", "Tensile_Strength", "Melting_Point", "High_Performance"],
    errors="ignore"
)

# Keep only numeric columns
X_unsupervised = X_unsupervised.select_dtypes(include=[np.number])

# Scale features (VERY IMPORTANT for PCA & K-Means)
scaler_pca = StandardScaler()
X_scaled_pca = scaler_pca.fit_transform(X_unsupervised)

# ============================
# 2️⃣ PCA (Dimensionality Reduction)
# ============================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_pca)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", pca.explained_variance_ratio_.sum())

# ============================
# 3️⃣ K-MEANS CLUSTERING
# ============================

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Add cluster labels to original data
data["Cluster"] = clusters

# ============================
# 4️⃣ PCA SCATTER PLOT (CLUSTERS)
# ============================

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA + K-Means Clustering of Alloys")
plt.colorbar(scatter, label="Cluster")
plt.show()

# ============================
# 5️⃣ CLUSTER vs PERFORMANCE
# ============================

cluster_summary = data.groupby("Cluster")[
    ["Tensile_Strength", "Melting_Point", "High_Performance"]
].mean()

print("\nCluster Performance Summary:\n")
print(cluster_summary)

# ============================
# 6️⃣ SAVE PCA & K-MEANS MODELS (FOR UI)
# ============================

import joblib

joblib.dump(pca, "pca_model.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler_pca, "pca_scaler.pkl")

print("✅ PCA and K-Means models saved")


