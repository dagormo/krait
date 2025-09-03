import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

# ----- CONFIG -----
VC_INPUT_FILE = "../../.venv/AS28-Fast-4um_Predict.csv"
PCA_FILE = "pca_space_95pct.csv"
MODEL_FILE = "../../pkl/logk_model_hydroxide.pkl"
X_TRAIN_FILE = "X_hydroxide_train.csv"
Y_TRAIN_FILE = "y_hydroxide_train.csv"
OUTPUT_FILE = "predictions.csv"

CATEGORICAL_COLS = [
    "Functional group",
    "Functional group characteristics",
    "Resin composition",
    "Chemistry"
]
RARE_THRESH = 10

# ----- HELPER FUNCTION -----
def clean_and_encode_categoricals(df, categorical_columns, rare_thresh=0):
    df = df.copy()
    for col in categorical_columns:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        if rare_thresh > 0:
            counts = df[col].value_counts()
            rare_vals = counts[counts < rare_thresh].index
            df[col] = df[col].apply(lambda x: "Other" if x in rare_vals else x)
    df = pd.get_dummies(df, columns=categorical_columns)
    return df

# ----- LOAD FILES -----
print("📥 Loading input files...")
vc_df = pd.read_csv(VC_INPUT_FILE)
pca_df = pd.read_csv(PCA_FILE)
X_train = pd.read_csv(X_TRAIN_FILE)
y_train = pd.read_csv(Y_TRAIN_FILE)
model = joblib.load(MODEL_FILE)

# ----- CLEAN HEADERS -----
vc_df.columns = vc_df.columns.str.strip()
pca_df.columns = pca_df.columns.str.strip()

# ----- MERGE ANALYTE DESCRIPTORS -----
print("🔗 Merging analyte PCA descriptors...")
merged = pd.merge(vc_df, pca_df, left_on="Analyte", right_on="Name", how="left", validate="many_to_one")

# ----- DROP ROWS MISSING PCA INFO -----
pca_cols = [col for col in merged.columns if col.startswith("PC")]
missing_pca = merged[pca_cols].isnull().any(axis=1)

if missing_pca.any():
    print("⚠️ Dropping rows with missing PCA descriptors:")
    print(merged.loc[missing_pca, "Analyte"].values)
    merged = merged[~missing_pca]

if merged.empty:
    raise ValueError("❌ No rows left for prediction after PCA merge. Exiting.")

# ----- DROP UNUSED IDENTIFIERS -----
merged = merged.drop(columns=["Name", "SMILES"], errors="ignore")

# ----- ENCODE CATEGORICALS -----
print("🧼 Encoding categorical variables...")
encoded = clean_and_encode_categoricals(merged, CATEGORICAL_COLS, rare_thresh=RARE_THRESH)

# ----- ALIGN FEATURES TO MODEL -----
print("🧮 Aligning features to model input...")
model_features = model.feature_names_in_
for col in model_features:
    if col not in encoded.columns:
        encoded[col] = 0
encoded = encoded[model_features]

# ----- PREDICT -----
print("🤖 Predicting log(k)...")
merged["Predicted log(k)"] = model.predict(encoded)
merged["k'"] = 10 ** merged["Predicted log(k)"]

if "Void" in merged.columns:
    merged["Predicted Retention Time (min)"] = merged["Void"] * (1 + merged["k'"])
else:
    merged["Predicted Retention Time (min)"] = np.nan
    print("⚠️ 'Void' column not found — cannot compute retention time.")


# Align features to model spec
X_train = X_train[model_features]

# Drop rows with any NaNs
X_train_clean = X_train.dropna()
y_train_clean = y_train.loc[X_train_clean.index].iloc[:, 0]

encoded = encoded.fillna(0)  # Or use a better imputation strategy if appropriate

# Check for NaNs in the input to NearestNeighbors
if encoded.isnull().any().any():
    raise ValueError("❌ 'encoded' DataFrame still contains NaNs. Cannot compute neighbors.")

# Fit NearestNeighbors only on clean data
nn = NearestNeighbors(n_neighbors=10)
nn.fit(X_train_clean)
# ----- CONFIDENCE INTERVALS (from residuals) -----
print("📐 Estimating confidence intervals...")
y_train_pred = model.predict(X_train_clean)
residuals = y_train_clean - y_train_pred
nn = NearestNeighbors(n_neighbors=10)
nn.fit(X_train_clean)
_, indices = nn.kneighbors(encoded)
local_errors = np.abs(np.take(residuals.values, indices))
error_std = local_errors.std(axis=1)

# CI bounds in log(k)
merged["log(k)_lower"] = merged["Predicted log(k)"] - 1.96 * error_std
merged["log(k)_upper"] = merged["Predicted log(k)"] + 1.96 * error_std

# Convert to k' and retention time
merged["k'_lower"] = 10 ** merged["log(k)_lower"]
merged["k'_upper"] = 10 ** merged["log(k)_upper"]
if "Void" in merged.columns:
    merged["RT_lower"] = merged["Void"] * (1 + merged["k'_lower"])
    merged["RT_upper"] = merged["Void"] * (1 + merged["k'_upper"])
else:
    merged["RT_lower"] = merged["RT_upper"] = np.nan

# ----- DISPLAY SUMMARY -----
for _, row in merged.iterrows():
    print(f"\n🔍 {row['Analyte']}")
    print(f"Predicted log(k): {row['Predicted log(k)']:.4f}")
    print(f"Predicted RT: {row['Predicted Retention Time (min)']:.2f} min")
    print(f"95% CI: [{row['RT_lower']:.2f}, {row['RT_upper']:.2f}] min")

# ----- SAVE -----
merged.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Predictions and confidence intervals saved to: {OUTPUT_FILE}")