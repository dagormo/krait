#!/usr/bin/env python3
"""
Select 10 anion reps and 10 cation reps by clustering in PCA space.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# ─── CONFIG ────────────────────────────────────────────────────────────────────
PCA_CSV    = "pca_space_95pct.csv"    # must have columns PC1…PCn, Name, SMILES, IonType
N_REPS     = 15                       # reps per IonType
SEED       = 0                        # for KMeans
PC_PREFIX  = "PC"                     # PCA column prefix
ANION_OUT  = "rep_anions.csv"
CATION_OUT = "rep_cations.csv"
# ────────────────────────────────────────────────────────────────────────────────

def pick_medoid_reps(df_sub: pd.DataFrame, n_reps: int, seed: int):
    """
    Given a subset df_sub with PCA columns PC*, cluster into n_reps groups
    and return the medoid (closest to each centroid) as a DataFrame.
    """
    pcs = [c for c in df_sub.columns if c.startswith(PC_PREFIX)]
    X   = df_sub[pcs].values

    km = KMeans(n_clusters=n_reps, random_state=seed)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    # find medoids
    idxs, dists = pairwise_distances_argmin_min(centers, X)
    reps = df_sub.iloc[idxs][["Name","SMILES"]].copy().reset_index(drop=True)
    reps["distance_to_centroid"] = dists
    return reps

def main():
    df = pd.read_csv(PCA_CSV)

    # Anions
    anions = df[df["IonType"] == "anion"].reset_index(drop=True)
    if len(anions) < N_REPS:
        raise ValueError(f"Only {len(anions)} anions available, need {N_REPS}")
    reps_anion = pick_medoid_reps(anions, N_REPS, SEED)
    reps_anion["IonType"] = "anion"
    reps_anion.to_csv(ANION_OUT, index=False)
    print(f"Wrote {len(reps_anion)} anion reps to {ANION_OUT}")

if __name__ == "__main__":
    main()
