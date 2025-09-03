import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Bootstrapping function
def bootstrap_metrics(y_true, y_pred, n_bootstrap=10000):
    rng = np.random.default_rng(42)
    n = len(y_true)
    rmse_vals = []
    r2_vals = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        rmse_vals.append(np.sqrt(mean_squared_error(y_true_sample, y_pred_sample)))
        r2_vals.append(r2_score(y_true_sample, y_pred_sample))

    return np.array(rmse_vals), np.array(r2_vals)

# Setup
eluents = ['hydroxide']
held_out_chem = ['AS10','AS11','AS11-HC','AS15','AS16','AS17','AS18','AS19','AS20','AS24','AS27']

# For violin plot aggregation
rmse_distributions = []
r2_distributions = []
column_names = []

# Main loop
for eluent in eluents:
    for chem in held_out_chem:
        X_train = pd.read_csv(f"X_{eluent}_{chem}_train.csv")
        X_test = pd.read_csv(f"X_{eluent}_{chem}_test.csv")
        y_train = pd.read_csv(f"y_{eluent}_{chem}_train.csv").squeeze()
        y_test = pd.read_csv(f"y_{eluent}_{chem}_test.csv").squeeze()

        # Drop metadata
        X_train = X_train.drop(columns=["Analyte", "Retention time"], errors="ignore")
        X_test = X_test.drop(columns=["Analyte", "Retention time"], errors="ignore")

        # Train model
        model = HistGradientBoostingRegressor(max_iter=500, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate point estimates
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"{eluent}_{chem}: RMSE = {rmse:.4f}, R² = {r2:.4f}")

        # Save model
        joblib.dump(model, f'logk_model_{eluent}_{chem}.pkl')

        # Predicted vs actual plot
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual log(k)')
        plt.ylabel('Predicted log(k)')
        plt.title(f'Predicted vs Actual log(k) - {eluent} - {chem}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"pred_performance_{eluent}_{chem}.png", dpi=300)
        plt.clf()

        # Bootstrapping
        y_test_arr = y_test.to_numpy()
        rmse_vals, r2_vals = bootstrap_metrics(y_test_arr, y_pred)

        # Store for violin plot
        rmse_distributions.append(rmse_vals)
        r2_distributions.append(r2_vals)
        column_names.append(chem)

        # Print 95% CIs
        rmse_ci = np.percentile(rmse_vals, [2.5, 97.5])
        r2_ci = np.percentile(r2_vals, [2.5, 97.5])
        print(f"{chem} - RMSE 95% CI: [{rmse_ci[0]:.3f}, {rmse_ci[1]:.3f}], R² 95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")

# Create long-form DataFrames for violin plots
df_rmse = pd.DataFrame({chem: dist for chem, dist in zip(column_names, rmse_distributions)})
df_r2 = pd.DataFrame({chem: dist for chem, dist in zip(column_names, r2_distributions)})

df_rmse_melt = df_rmse.melt(var_name="Column", value_name="RMSE")
df_r2_melt = df_r2.melt(var_name="Column", value_name="R²")

# Plot RMSE violin
plt.figure(figsize=(10, 5))
sns.violinplot(data=df_rmse_melt, x="Column", y="RMSE", inner="quartile", color="lightgrey", linewidth=1)
plt.title("Bootstrapped RMSE Distribution (95% CI)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bootstrap_rmse_violin.png", dpi=300)
plt.clf()

# Plot R² violin
plt.figure(figsize=(10, 5))
sns.violinplot(data=df_r2_melt, x="Column", y="R²", inner="quartile", color="lightgrey", linewidth=1)
plt.title("Bootstrapped R² Distribution (95% CI)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bootstrap_r2_violin.png", dpi=300)
plt.clf()
