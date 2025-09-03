import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Load preprocessed data
eluents = ['hydroxide']  # , 'carbonate', 'msa']
for eluent in eluents:
    X_train = pd.read_csv(f"../data/X_{eluent}_train.csv")
    X_test = pd.read_csv(f"../data/X_{eluent}_test.csv")
    y_train = pd.read_csv(f"../data/y_{eluent}_train.csv").squeeze()
    y_test = pd.read_csv(f"../data/y_{eluent}_test.csv").squeeze()

    # drop the identifier column before fitting
    X_train = X_train.drop(columns=["Analyte"], errors="ignore")
    X_test = X_test .drop(columns=["Analyte"], errors="ignore")

    # Initialize model
    model = HistGradientBoostingRegressor(max_iter=500, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE for {eluent}: {rmse:.4f}")
    print(f"R² Score for {eluent}: {r2:.4f}")

    # Save model
    joblib.dump(model, f'../pkl/logk_model_{eluent}.pkl')

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual log(k)')
    plt.ylabel('Predicted log(k)')
    plt.title(f'Predicted vs Actual log(k) - {eluent}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../data/pred_performance_{eluent}.png", dpi=300)
    plt.clf()

    # Load test features and model
    X_test = pd.read_csv(f"../data/X_{eluent}_test.csv")
    y_test = pd.read_csv(f"../data/y_{eluent}_test.csv").squeeze()
    model = joblib.load(f"../pkl/logk_model_{eluent}.pkl")

    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42,
                                    scoring='neg_root_mean_squared_error')

    # Compute feature importances
    importances = result.importances_mean
    indices = np.argsort(importances)[::-1]
    feature_names = X_test.columns
    sorted_idx = np.argsort(importances)[::-1]

    # Plot top 20
    plt.figure(figsize=(10, 6))
    plt.barh(range(20), importances[sorted_idx[:20]][::-1])
    plt.yticks(range(20), [feature_names[i] for i in sorted_idx[:20]][::-1])
    plt.xlabel("Permutation Importance (mean RMSE drop)")
    plt.title(f"Top 20 Features (Permutation Importance) - {eluent}")
    plt.tight_layout()
    plt.savefig(f"../data/feature_importance_{eluent}.png", dpi=300)
    plt.clf()
