#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

# ─── SETTINGS ───────────────────────────────────────────────────────────────────
PCA_CSV     = "pca_space_95pct.csv"   # PC1…PCn, Name, SMILES, IonType
MERGED_CSV  = "merged_hydroxide.csv"  # must have columns Name, log(k), plus any others
PC_PREFIX   = "PC"
N_REPS      = 15
INIT_RANDOM = 5
SEED        = 42
# ────────────────────────────────────────────────────────────────────────────────

# 1) Load PCA coords + IonType
pca_df = pd.read_csv(PCA_CSV)
pca_df = pca_df.loc[:, ~pca_df.columns.duplicated()]

# 2) Load original data with the log(k) target (and rename Analyte->Name if needed)
orig = pd.read_csv(MERGED_CSV)
if "Analyte" in orig.columns:
    orig = orig.rename(columns={"Analyte": "Name"})
orig = orig.loc[:, ~orig.columns.duplicated()]

# 3) Merge so we have log(k) alongside the PCs
df = pd.merge(
    pca_df,
    orig[["Name", "log(k)"]],
    on="Name",
    how="inner"
)
df = df.loc[:, ~df.columns.duplicated()]
# now df has columns: PC1…PCn, Name, SMILES, IonType, log(k)

# ─── supervised selection routine ───────────────────────────────────────────────
def supervised_max_error(sub, feat_cols, n_select, init_n, seed):
    X = sub[feat_cols].values
    y = sub["log(k)"].values

    rng = np.random.RandomState(seed)
    all_idx = np.arange(len(sub))
    selected = list(rng.choice(all_idx, size=init_n, replace=False))
    remaining = [i for i in all_idx if i not in selected]

    model = HistGradientBoostingRegressor(max_iter=500, random_state=seed)

    while len(selected) < n_select:
        model.fit(X[selected], y[selected])
        y_pred = model.predict(X[remaining])
        errs = np.abs(y_pred - y[remaining])
        # pick the worst‐predicted compound
        worst = remaining[int(np.argmax(errs))]
        selected.append(worst)
        remaining.remove(worst)

    return selected

# 4) Run for a given IonType
def pick_and_eval(df, ion):
    pcs = [c for c in df.columns if c.startswith(PC_PREFIX)]
    sub = df[df.IonType == ion].reset_index(drop=True)
    if len(sub) < N_REPS:
       raise ValueError(f"Only {len(sub)} {ion}s available, need {N_REPS}")

    idxs = supervised_max_error(sub, pcs, N_REPS, INIT_RANDOM, SEED)
    reps = sub.iloc[idxs][["Name","SMILES"]].copy()
    reps["IonType"] = ion

    # retrain & evaluate
    X_tr = sub.iloc[idxs][pcs].values
    y_tr = sub.iloc[idxs]["log(k)"].values
    mask = np.ones(len(sub), bool)
    mask[idxs] = False
    X_te = sub.iloc[mask][pcs].values
    y_te = sub.iloc[mask]["log(k)"].values

    model = HistGradientBoostingRegressor(max_iter=500, random_state=SEED)
    model.fit(X_tr, y_tr)
    r2 = r2_score(y_te, model.predict(X_te))
    print(f"{ion:>8}: R² = {r2:.4f} with {len(idxs)} reps")

    return reps

# 5) Execute for hydroxide anions and cations
anion_reps  = pick_and_eval(df, "anion")
#cation_reps = pick_and_eval(df, "cation")

anion_reps .to_csv("reps_supervised_anions.csv",  index=False)
#cation_reps.to_csv("reps_supervised_cations.csv", index=False)
print("✅ Saved supervised reps.")
