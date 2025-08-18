import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ================
# 1. Load and Prepare Data
# ================
df = pd.read_csv("aggregated_results.csv")

# Inverse problem: from deformed PCA + pressures → undeformed PCA + material properties
feature_cols = [f"defPCA{i}" for i in range(1, 11)] + ["LV_ED", "LV_ES", "RV_ED", "RV_ES"]
target_cols = [f"PCA{i}" for i in range(1, 11)] + ["a", "a_f"]

X = df[feature_cols].values
y = df[target_cols].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# ================
# 2. Train Model
# ================
mlp = MLPRegressor(hidden_layer_sizes=(64, 64),
                   activation='relu',
                   solver='adam',
                   alpha=0.001,
                   max_iter=500,
                   random_state=42)

mlp.fit(X_train_scaled, y_train_scaled)

# ================
# 3. Evaluate
# ================
y_pred_scaled = mlp.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
print("\nPerformance:")
for i, score in enumerate(rmse, start=1):
    if i <= 10:
        print(f"PCA{i} RMSE: {score:.4f}")
    elif i == 11:
        print(f"a RMSE: {score:.4f}")
    else:
        print(f"a_f RMSE: {score:.4f}")

print(f"Mean RMSE: {rmse.mean():.4f}")
print(f"Overall R²: {r2_score(y_test, y_pred):.4f}")

import matplotlib.pyplot as plt

# Combine PCA + material properties for plotting
output_names = [f"PCA{i}" for i in range(1, 11)] + ["a", "a_f"]
n_outputs = len(output_names)

# --------- Predicted vs Actual Plot ---------
fig, axes = plt.subplots(3, 4, figsize=(20, 12))  # 12 panels (10 PCA + 2 materials)
axes = axes.flatten()

for i in range(n_outputs):
    axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
    min_val = min(y_test[:, i].min(), y_pred[:, i].min())
    max_val = max(y_test[:, i].max(), y_pred[:, i].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[i].set_xlabel("Actual")
    axes[i].set_ylabel("Predicted")
    axes[i].set_title(f"{output_names[i]} (RMSE: {rmse[i]:.3f})")

# Hide any empty subplots if less than 12 outputs
for j in range(n_outputs, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Inverse Model: Predicted vs Actual", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --------- Residual Plots ---------
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()

for i in range(n_outputs):
    residuals = y_pred[:, i] - y_test[:, i]
    axes[i].scatter(y_pred[:, i], residuals, alpha=0.5)
    axes[i].axhline(0, color='red', linestyle='--')
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Residual")
    axes[i].set_title(f"{output_names[i]} Residuals")

for j in range(n_outputs, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Inverse Model: Residuals", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ================
# 4. Save Model & Scalers
# ================
joblib.dump(mlp, "inverse_model.pkl")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print("\nModel and scalers saved!")

# ===============================
# 5. Later... Load & Predict New Data
# ===============================
# Load objects
loaded_model = joblib.load("inverse_model.pkl")
loaded_scaler_X = joblib.load("scaler_X.pkl")
loaded_scaler_y = joblib.load("scaler_y.pkl")

# Example: predict for a random single case
new_sample = pd.DataFrame({
    **{f"defPCA{i}": [np.random.normal()] for i in range(1, 11)},
    "LV_ED": [np.random.uniform(80, 200)],
    "LV_ES": [np.random.uniform(30, 150)],
    "RV_ED": [np.random.uniform(80, 200)],
    "RV_ES": [np.random.uniform(30, 150)]
})

# Scale input
new_sample_scaled = loaded_scaler_X.transform(new_sample)

# Predict and inverse scale
pred_scaled = loaded_model.predict(new_sample_scaled)
pred = loaded_scaler_y.inverse_transform(pred_scaled)

# Split prediction into PCA and material properties
pred_pca = pred[0, :10]
pred_a = pred[0, 10]
pred_a_f = pred[0, 11]

print("\nRandom Sample Input:")
print(new_sample)
print("\nPredicted Undeformed PCA Scores:", pred_pca)
print(f"Predicted Material Property a: {pred_a:.4f}")
print(f"Predicted Material Property a_f: {pred_a_f:.4f}")
