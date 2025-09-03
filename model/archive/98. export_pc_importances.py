import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

# === CONFIG ===
MODEL_FILE = "../../pkl/logk_model_hydroxide.pkl"
X_FILE = "X_hydroxide_train.csv"
Y_FILE = "y_hydroxide_train.csv"  # replace with your y-values file
OUTPUT_FILE = "../../data/pc_feature_importances.csv"

# === LOAD MODEL AND DATA ===
model = joblib.load(MODEL_FILE)
X = pd.read_csv(X_FILE)
y = pd.read_csv(Y_FILE)

# === Align columns (optional but recommended) ===
# Ensure the column order matches what the model expects
if hasattr(model, "feature_names_in_"):
    X = X[model.feature_names_in_]

# === COMPUTE PERMUTATION IMPORTANCE ===
result = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring='neg_mean_squared_error')

# === SAVE IMPORTANCE SCORES ===
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": result.importances_mean
})
importance_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved PC importances to: {OUTPUT_FILE}")
