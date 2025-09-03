import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np

# Load model and data
eluents = ['hydroxide', 'carbonate', 'msa']
for eluent in eluents:
    model = joblib.load(f"logk_model_{eluent}.pkl")
    X_test = pd.read_csv(f"X_{eluent}_test.csv")
    y_test = pd.read_csv(f"y_{eluent}_test.csv").squeeze()
    merged = pd.read_csv(f"merged_{eluent}.csv")

    # Predict and compute residuals
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    threshold = 3 * rmse

    # Identify outlier rows
    outlier_mask = np.abs(residuals) > threshold
    merged_test = merged.iloc[X_test.index]  # align with test set

    # Extract relevant columns only
    outliers = merged_test[outlier_mask][["Analyte", "Filename"]].copy()
    outliers["actual_log_k"] = y_test[outlier_mask].values
    outliers["predicted_log_k"] = y_pred[outlier_mask]
    outliers["residual"] = residuals[outlier_mask]

    # Sort by largest error
    outliers = outliers.sort_values(by="residual", key=np.abs, ascending=False)

    # Save to file
    outliers.to_csv(f"logk_{eluent}_outliers_summary.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, label="Inliers")
    plt.scatter(y_test[outlier_mask], y_pred[outlier_mask], color='red', label="Outliers")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label="Ideal (y=x)")
    plt.xlabel("Actual log(k)")
    plt.ylabel("Predicted log(k)")
    plt.title(f"Predicted vs Actual log(k) with Outliers Highlighted - {eluent}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"overlay_predicted_{eluent}.png", dpi=300)

    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(3*rmse, color='red', linestyle='--', label='+3×RMSE')
    plt.axhline(-3*rmse, color='red', linestyle='--', label='–3×RMSE')
    plt.xlabel("Actual log(k)")
    plt.ylabel("Residual (Actual − Predicted)")
    plt.title(f"Residuals vs Actual log(k) - {eluent}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"predicted_residuals_{eluent}.png", dpi=300)

    plt.clf()
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.axvline(3*rmse, color='red', linestyle='--')
    plt.axvline(-3*rmse, color='red', linestyle='--')
    plt.xlabel("Residual (Actual − Predicted)")
    plt.ylabel("Count")
    plt.title(f"Histogram of Residuals - {eluent}")
    plt.tight_layout()
    plt.savefig(f"histogram_residuals_{eluent}.png", dpi=300)

    plt.clf()
    plt.scatter(X_test["PC17"], X_test["PC2"], c=residuals, cmap="coolwarm", alpha=0.5)
    plt.colorbar(label="Residual")
    plt.xlabel("PC17")
    plt.ylabel("PC2")
    plt.title(f"Residuals in PCA Space - {eluent}")
    plt.savefig(f"pca_residuals_{eluent}.png", dpi=300)

    # Align metadata
    merged_test = merged.iloc[X_test.index].copy()
    merged_test["Residual"] = residuals
    merged_test["PC17"] = X_test["PC17"]

    # Define "outlier" threshold for PC17 and residual
    pc17_thresh = 4  # tweak as needed
    resid_thresh = 3 * rmse

    # Filter for analytes with high PC17 and large absolute residual
    outliers = merged_test[(merged_test["PC17"] > pc17_thresh) & (merged_test["Residual"].abs() > resid_thresh)]

    # Sort by PC17 or residual if you like
    outliers = outliers.sort_values(by="PC17", ascending=False)

    # Save or inspect
    outliers.to_csv(f"high_PC17_outliers_{eluent}.csv", index=False)