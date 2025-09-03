#!/usr/bin/env python3
"""
Select representative anions and cations separately via:
 (a) Kennard–Stone (MaxMin distance)
 (b) D-optimal exchange
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# ─── SETTINGS ───────────────────────────────────────────────────────────────────
PCA_CSV    = "pca_space_95pct.csv"
PC_PREFIX  = "PC"
N_ANION    = 15
N_CATION   = 15

OUT_KS_AN  = "reps_ks_anions.csv"
OUT_KS_CAT = "reps_ks_cations.csv"
OUT_DO_AN  = "reps_do_anions.csv"
OUT_DO_CAT = "reps_do_cations.csv"
# ────────────────────────────────────────────────────────────────────────────────

def kennard_stone(df, feat_cols, n_select):
    X = df[feat_cols].values
    # Step 1: pick the two most distant points
    dist_mat = cdist(X, X)
    i, j = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)
    selected = [i, j]
    # Step 2: iteratively pick the point with largest distance to nearest selected
    while len(selected) < n_select:
        mask = np.ones(len(X), bool)
        mask[selected] = False
        rem = np.where(mask)[0]
        d_to_sel = dist_mat[np.ix_(rem, selected)].min(axis=1)
        next_idx = rem[np.argmax(d_to_sel)]
        selected.append(int(next_idx))
    return selected

def d_optimal_exchange(df, feat_cols, n_select, n_iter=500):
    """
    Fedorov exchange algorithm: start with a random subset of size n_select,
    then randomly swap in/out points if they increase det(X^T X).
    """
    X = df[feat_cols].values
    n = X.shape[0]
    # initialize with a random subset
    subset = list(np.random.choice(n, size=n_select, replace=False))
    best_det = np.linalg.det(X[subset].T @ X[subset])
    for _ in range(n_iter):
        out_idx = np.random.choice(subset)
        in_cand = np.random.choice([i for i in range(n) if i not in subset])
        trial = subset.copy()
        trial[trial.index(out_idx)] = in_cand
        det = np.linalg.det(X[trial].T @ X[trial])
        if det > best_det:
            best_det = det
            subset = trial
    return subset

def pick_and_save(df, ion, n_select, out_ks, out_do):
    pcs = [c for c in df.columns if c.startswith(PC_PREFIX)]
    sub = df[df.IonType == ion].reset_index(drop=True)
    if len(sub) < n_select:
        raise ValueError(f"Only {len(sub)} {ion}s but need {n_select}")

    # Kennard–Stone
    ks_idxs = kennard_stone(sub, pcs, n_select)
    ks_reps = sub.iloc[ks_idxs][["Name","SMILES"]].copy()
    ks_reps["Method"] = "Kennard-Stone"
    ks_reps["IonType"] = ion
    ks_reps.to_csv(out_ks, index=False)
    print(f"Wrote {len(ks_reps)} KS {ion} reps → {out_ks}")

    # D-optimal via exchange
    do_idxs = d_optimal_exchange(sub, pcs, n_select, n_iter=1000)
    do_reps = sub.iloc[do_idxs][["Name","SMILES"]].copy()
    do_reps["Method"] = "D-optimal"
    do_reps["IonType"] = ion
    do_reps.to_csv(out_do, index=False)
    print(f"Wrote {len(do_reps)} DO {ion} reps → {out_do}")

def main():
    df = pd.read_csv(PCA_CSV)
    pick_and_save(df, "anion",  N_ANION,  OUT_KS_AN,  OUT_DO_AN)
    pick_and_save(df, "cation", N_CATION, OUT_KS_CAT, OUT_DO_CAT)

if __name__ == "__main__":
    main()
