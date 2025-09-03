import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from scipy.optimize import curve_fit

def neue_kuss(phi, log_kw, S, beta):
    return log_kw - S * phi + beta * (phi ** 2)

# Load preprocessed data
eluents = ['hydroxide']
held_out_chem = ['AS10','AS11','AS11-HC','AS15','AS16','AS17','AS18','AS19','AS20','AS24','AS27']
log_kw_list = []
S_list = []
beta_list = []

for eluent in eluents:
    for chem in held_out_chem:
        X_train = pd.read_csv(f"X_{eluent}_{chem}_train.csv")
        X_test = pd.read_csv(f"X_{eluent}_{chem}_test.csv")
        y_train = pd.read_csv(f"y_{eluent}_{chem}_train.csv").squeeze()
        y_test = pd.read_csv(f"y_{eluent}_{chem}_test.csv").squeeze()

        # drop the identifier column before fitting
        X_train = X_train.drop(columns=["Analyte"], errors="ignore")
        X_test  = X_test.drop(columns=["Analyte"], errors="ignore")
        if "Start Concentration" in X_test.columns and "Gradient slope" in X_test.columns and "Retention time" in X_test.columns:
            retention_time = X_test["Retention time"].values
            start_conc = X_test["Start Concentration"].values
            slope = X_test["Gradient slope"].values
            eluent_conc = start_conc + slope * retention_time
            max_conc = 100.0 # Maximum KOH concentration used in gradient; capped by eluent generator
            phi = eluent_conc / max_conc
        else:
            raise ValueError("Start Concentration and Gradient Slope not found in X_test.")

        X_train = X_train.drop(columns=["Retention time"], errors="ignore")
        X_test = X_test.drop(columns=["Retention time"], errors="ignore")

        model = HistGradientBoostingRegressor(max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        params, _ = curve_fit(neue_kuss, phi, y_pred, p0=[1.0, 5.0, 0.1])
        log_kw, S, beta = params

        print(f"Neue-Kuss parameters for {eluent}_{chem}: log_kw={log_kw:.3f}, S={S:.3f}, β={beta:.3f}")

        log_kw_list.append(log_kw)
        S_list.append(S)
        beta_list.append(beta)

plt.figure(figsize=(6,5))
plt.scatter(S_list, beta_list, c='blue')
for i, label in enumerate(held_out_chem):
    plt.text(S_list[i], beta_list[i], label, fontsize=9)
plt.xlabel("S")
plt.ylabel("β")
plt.title("Neue-Kuss S vs β by Column")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"Neue-Kuss fit.png", dpi=300)
plt.clf()