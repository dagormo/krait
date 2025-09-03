import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

eluents = ['hydroxide']
held_out_chem = ['AS10','AS11','AS11-HC','AS15','AS16','AS17','AS18','AS19','AS20','AS24','AS27']
spearman_rhos = []
spearman_labels = []

for eluent in eluents:
    for chem in held_out_chem:
        X_train = pd.read_csv(f"X_{eluent}_{chem}_train.csv")
        X_test = pd.read_csv(f"X_{eluent}_{chem}_test.csv")
        y_train = pd.read_csv(f"y_{eluent}_{chem}_train.csv").squeeze()
        y_test = pd.read_csv(f"y_{eluent}_{chem}_test.csv").squeeze()

        # drop the identifier column before fitting
        X_train = X_train.drop(columns=["Analyte"], errors="ignore")
        X_test  = X_test.drop(columns=["Analyte"], errors="ignore")
        X_train = X_train.drop(columns=["Retention time"], errors="ignore")
        X_test = X_test.drop(columns=["Retention time"], errors="ignore")

        model = HistGradientBoostingRegressor(max_iter=500, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rho, p_value = spearmanr(y_test, y_pred)

        spearman_rhos.append(rho)
        spearman_labels.append(chem)

plt.figure(figsize=(8,5))
bars = plt.bar(spearman_labels, spearman_rhos, color='lightgrey', edgecolor='black')

# Add labels on top of each bar
for bar, value in zip(bars, spearman_rhos):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{value:.2f}",
             ha='center', va='bottom', fontsize=9)

plt.ylabel("Spearman ρ")
plt.title("Elution Order Agreement by Column")
plt.ylim(0, 1.05)
plt.axhline(0.8, linestyle='--', color='gray')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("Spearman_by_column_labeled.png", dpi=300)
plt.clf()