import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ======================
# 1. Load and prepare training data
# ======================
df = pd.read_csv("aggregated_results_minus_5.csv")

feature_cols = [f"PCA{i}" for i in range(1, 11)] + ["LV_ED", "LV_ES", "RV_ED", "RV_ES", "a", "a_f"]
target_cols = [f"defPCA{i}" for i in range(1, 11)]

X = df[feature_cols].values
y = df[target_cols].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features and targets
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# ======================
# 2. Define MLP and hyperparameter grid
# ======================
mlp = MLPRegressor(random_state=42, max_iter=200000)

param_grid = {
    'hidden_layer_sizes': [(64, 64), (128, 64), (128, 128)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.001, 0.01, 0.1],
    'learning_rate_init': [0.001, 0.01]
}

# ======================
# 3. Grid search with 5-fold CV
# ======================
grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_scaled, y_train_scaled)

print("\nBest hyperparameters found:")
print(grid_search.best_params_)

# ======================
# 4. Evaluate best model on test set
# ======================
best_mlp = grid_search.best_estimator_

y_pred_scaled = best_mlp.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
overall_rmse = rmse.mean()
overall_r2 = r2_score(y_test, y_pred)

print("\nTest Set Performance:")
for i, score in enumerate(rmse, start=1):
    print(f"DefPCA{i} RMSE: {score:.4f}")
print(f"Mean RMSE: {overall_rmse:.4f}")
print(f"Overall R²: {overall_r2:.4f}")

# ======================
# 5. Patient-specific prediction
# ======================
patient_df = pd.read_csv("patient_5.csv")
patient_features = patient_df[feature_cols].values
patient_features_scaled = scaler_X.transform(patient_features)

patient_prediction_scaled = best_mlp.predict(patient_features_scaled)
patient_prediction = scaler_y.inverse_transform(patient_prediction_scaled)

true_patient_values = patient_df[target_cols].values

# print("\nPatient-specific predictions vs true values (for each sample):")
# for sample_idx in range(len(patient_prediction)):
#     print(f"\nSample {sample_idx + 1}:")
#     for i in range(len(target_cols)):
#         pred_val = patient_prediction[sample_idx, i]
#         true_val = true_patient_values[sample_idx, i]
#         print(f"  DefPCA{i+1}: Predicted = {pred_val:.4f}, True = {true_val:.4f}")

# Choose a random sample test case to visualize
import random
# Choose a random sample index from 1 to 43
sample_idx = random.randint(1, 42)
print(f"\nVisualizing Sample {sample_idx + 1}:")
print("Predicted values:", patient_prediction[sample_idx])
print("True values:", true_patient_values[sample_idx])
for i in range(len(target_cols)):
    pred_val = patient_prediction[sample_idx, i]
    true_val = true_patient_values[sample_idx, i]
    print(f"  DefPCA{i+1}: Predicted = {pred_val:.4f}, True = {true_val:.4f}")

rmse = np.sqrt(mean_squared_error(patient_prediction[sample_idx], true_patient_values[sample_idx], multioutput='raw_values'))
overall_rmse = rmse.mean()
overall_r2 = r2_score(true_patient_values[sample_idx], patient_prediction[sample_idx])

print("\nTest Set Performance:")
for i, score in enumerate(rmse, start=1):
    print(f"DefPCA{i} RMSE: {score:.4f}")
print(f"Mean RMSE: {overall_rmse:.4f}")
print(f"Overall R²: {overall_r2:.4f}")
