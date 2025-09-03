#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ─── SETTINGS ───────────────────────────────────────────────────────────────────
PCA_REPS = ['n-Butyrate', 'Tartrate', 'Citraconate', 'Ethanesulfonate', 'Mesaconate', 'Sucrose', 'Chromate', 'Trifluroacetate', 'Fluoroacetate', 'Benzoate', 'Malonate', 'Sorbose', 'Benzenesulfonate', 'Bromoacetate', 'Pyruvate']
MERGED_CSV = "merged_hydroxide.csv"
TARGET     = "log(k)"
MODEL_OUT  = "hydroxide_reps_model.pkl"
PLOT_OUT   = "reps_performance_hydroxide.png"
# ────────────────────────────────────────────────────────────────────────────────

# 1) load merged data
df = pd.read_csv(MERGED_CSV)

# 2) rename if needed
if "Analyte" in df.columns:
    df = df.rename(columns={"Analyte": "Name"})
df = df.loc[:, ~df.columns.duplicated()]

# 3) split into reps vs others
train_df = df[df["Name"].isin(PCA_REPS)].copy()
test_df  = df[~df["Name"].isin(PCA_REPS)].copy()

print(f"Training on {len(train_df)} reps; testing on {len(test_df)} others")

# 4) separate X / y
drop_cols = [
    'Name', 'SMILES', TARGET,
    'Filename', 'Retention time', 'Peak Area', 'Asymmetry', 'ESI',
    '1/T', 'Void', 'Plates', 'Column i.d.', 'Column length',
    'Chemistry', 'log_hydrophobicity', 'Eluent',
    'Functional group characteristics', 'IonType'
]

# raw splits
X_train_raw = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
y_train     = train_df[TARGET]
X_test_raw  = test_df .drop(columns=[c for c in drop_cols if c in test_df.columns])
y_test      = test_df [TARGET]

# find categorical columns (object or category dtype)
cat_cols = X_train_raw.select_dtypes(include=['object','category']).columns.tolist()

# one-hot encode them *jointly* so train/test share the same columns
X_all = pd.get_dummies(
    pd.concat([X_train_raw, X_test_raw], axis=0),
    columns=cat_cols,
    prefix_sep='_'
)

# split back
X_train = X_all.iloc[:len(X_train_raw), :].copy()
X_test  = X_all.iloc[len(X_train_raw):, :].copy()

# 5) train model
model = HistGradientBoostingRegressor(max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 6) predict & evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# 7) save model
joblib.dump(model, MODEL_OUT)
print(f"Saved model to {MODEL_OUT}")

# 8) plot predicted vs actual
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.4)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--')
plt.xlabel("Actual log(k)")
plt.ylabel("Predicted log(k)")
plt.title("Predicted vs Actual log(k) — Reps → Others")
plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=300)
print(f"Saved performance plot to {PLOT_OUT}")
