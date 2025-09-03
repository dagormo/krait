#!/usr/bin/env python3
"""
Train log(k) models on PCA‐representative analytes and test on the remaining ions.
Assumes that X_{eluent}_train.csv / X_{eluent}_test.csv and y_{eluent}_*.csv
have already been created by the modified prep script.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ELUENTS       = ['hydroxide']#, 'carbonate', 'msa']
X_TRAIN_TPL   = "X_{eluent}_train_rep.csv"
X_TEST_TPL    = "X_{eluent}_test_rep.csv"
Y_TRAIN_TPL   = "y_{eluent}_train_rep.csv"
Y_TEST_TPL    = "y_{eluent}_test_rep.csv"
MODEL_TPL     = "logk_model_{eluent}_reps.pkl"
PLOT_TPL      = "pred_reps_{eluent}.png"
FI_TPL        = "fi_reps_{eluent}.png"
RANDOM_STATE  = 42
MAX_ITER      = 500
# ────────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(eluent: str):
    # Load features and targets
    X_train = pd.read_csv(X_TRAIN_TPL.format(eluent=eluent))
    X_test  = pd.read_csv(X_TEST_TPL.format(eluent=eluent))
    y_train = pd.read_csv(Y_TRAIN_TPL.format(eluent=eluent)).squeeze()
    y_test  = pd.read_csv(Y_TEST_TPL.format(eluent=eluent)).squeeze()

    print(f"\n--- {eluent.upper()} ---")
    print(f"Training on {len(X_train)} samples; testing on {len(X_test)} samples")

    # Initialize and train model
    model = HistGradientBoostingRegressor(
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, MODEL_TPL.format(eluent=eluent))

    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Scatter plot: actual vs predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
    plt.xlabel("Actual log(k)")
    plt.ylabel("Predicted log(k)")
    plt.title(f"{eluent}: reps → others")
    plt.tight_layout()
    plt.savefig(PLOT_TPL.format(eluent=eluent), dpi=300)
    plt.close()

    # Permutation importance
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring='neg_root_mean_squared_error'
    )
    importances = perm.importances_mean
    idx = np.argsort(importances)[::-1][:20]
    feat_names = X_test.columns[idx]

    plt.figure(figsize=(8,6))
    plt.barh(range(len(idx))[::-1], importances[idx][::-1])
    plt.yticks(range(len(idx))[::-1], feat_names[::-1])
    plt.xlabel("Mean increase in RMSE if permuted")
    plt.title(f"Top 20 Feature Importances ({eluent})")
    plt.tight_layout()
    plt.savefig(FI_TPL.format(eluent=eluent), dpi=300)
    plt.close()

def main():
    for el in ELUENTS:
        train_and_evaluate(el)

if __name__ == "__main__":
    main()
