#!/usr/bin/env python3
import re
import random
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

# ─── CONFIG ────────────────────────────────────────────────────────────────────
PCA_CSV    = "pca_space_95pct.csv"
MERGED_CSV = "merged_hydroxide.csv"
ALT_ID     = "Analyte"   # column in merged_hydroxide to rename → Name
ID_COL     = "Name"
TARGET     = "log(k)"
PC_PREFIX  = "PC"

POP_SIZE   = 40
GENS       = 30
K          = 12
TOURN_SIZE = 4
CXPB       = 0.8
MUTPB      = 0.3
SEED       = 2
SEED_PANEL = ['cis-Aconitate', 'Selenate', 'Propanesulfonate', 'Nitrate', 'Fluoroacetate', 'Chloride', 'Benzenesulfonate', 'Caprate', 'Trimetaphosphate', 'Molybdate', 'Mesoxalate', 'Heptanoate']
EARLY_STOP = 10  # Stop if no improvement after this many generations

# ────────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)

# 1b) merged runs file
runs = pd.read_csv(MERGED_CSV)
runs.columns = runs.columns.str.strip()
if ALT_ID in runs.columns:
    runs = runs.rename(columns={ALT_ID: ID_COL})
runs = runs.loc[:, ~runs.columns.duplicated()]

# Now detect PC columns directly in `runs`
pc_pattern = re.compile(r'^PC\d+$', re.IGNORECASE)
pc_cols = [c for c in runs.columns if pc_pattern.match(c)]
if not pc_cols:
    raise RuntimeError(f"No PC columns found in {MERGED_CSV}; got {runs.columns.tolist()}")

df = runs.copy().reset_index(drop=True)
# ─── DIAGNOSTIC: Check seed panel membership ───────────────────────────────
if SEED_PANEL:
    df_names_set = set(df[ID_COL].unique())
    panel_set = set(SEED_PANEL)
    missing = panel_set - df_names_set
    present = panel_set & df_names_set

    print(f"\n✅ {len(present)} analytes found in df[{ID_COL}]: {sorted(present)}")
    if missing:
        print(f"❌ {len(missing)} analytes NOT found in df[{ID_COL}]: {sorted(missing)}")
    else:
        print("🎉 All seed panel analytes are present in the dataset.")

# Build X_all, y_all, and names
# Define same column drop logic
drop_cols = [
    'Name', 'SMILES', TARGET,
    'Filename', 'Retention time', 'Peak Area', 'Asymmetry', 'ESI',
    '1/T', 'Void', 'Plates', 'Column i.d.', 'Column length',
    'Chemistry', 'log_hydrophobicity', 'Eluent',
    'Functional group characteristics', 'IonType'
]

X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])
cat_cols = X_raw.select_dtypes(include=['object','category']).columns.tolist()

X_all = pd.get_dummies(X_raw, columns=cat_cols, prefix_sep='_').values

y_all = df[TARGET].values
names = df[ID_COL].tolist()

# ─── MANUAL BASELINE TEST ─────────────────────────────────────────────
if SEED_PANEL:
    mask = df[ID_COL].isin(SEED_PANEL)
    X_tr, y_tr = X_all[mask], y_all[mask]
    X_te, y_te = X_all[~mask], y_all[~mask]
    print(f"\nManual baseline fit: training on {X_tr.shape[0]} rows, testing on {X_te.shape[0]} rows")

    mdl = HistGradientBoostingRegressor(max_iter=500, random_state=SEED)
    mdl.fit(X_tr, y_tr)
    pred = mdl.predict(X_te)
    r2 = r2_score(y_te, pred)

    print(f"💡 Manual baseline R² = {r2:.4f}")


# 2) FITNESS: train on subset runs, test on the rest

def fitness(subset):
    mask = df[ID_COL].isin(subset)
    X_tr, y_tr = X_all[mask],   y_all[mask]
    X_te, y_te = X_all[~mask],  y_all[~mask]
    mdl = HistGradientBoostingRegressor(max_iter=500, random_state=SEED)
    mdl.fit(X_tr, y_tr)
    return r2_score(y_te, mdl.predict(X_te))

# 3) COMPUTE K-MEANS MEDOID SEED PANEL

unique_ions = df[[ID_COL] + pc_cols] \
                .drop_duplicates(subset=ID_COL) \
                .reset_index(drop=True)
X_uniqu = unique_ions[pc_cols].values

if SEED_PANEL:
    missing = [ion for ion in SEED_PANEL if ion not in unique_ions[ID_COL].values]
    if len(SEED_PANEL) != K:
        raise ValueError(f"Hard-coded seed panel must contain exactly {K} analytes; got {len(SEED_PANEL)}.")
    if missing:
        raise ValueError(f"These analytes are not present in the data: {missing}")
    seed_panel = SEED_PANEL.copy()
    print("Using hard-coded seed panel:", seed_panel)
else:
    km = KMeans(n_clusters=K, random_state=SEED).fit(X_uniqu)
    seed_idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, X_uniqu)
    seed_panel = unique_ions.loc[seed_idx, ID_COL].tolist()
    print("Seed panel (K-Means medoids):", seed_panel)

# 4) RUN GENETIC ALGORITHM

print(f"First few names in df[{ID_COL}]:", df[ID_COL].unique()[:5])
print(f"Example from seed_panel:", seed_panel[0])

no_improve = 0
all_ions = unique_ions[ID_COL].tolist()

# initialize with seed + random panels
population = [seed_panel] + [
    random.sample(all_ions, K)
    for _ in range(POP_SIZE - 1)
]

best_subset = population[0].copy()
best_score  = fitness(best_subset)
print(f"\nGen  0 (seed): best R² = {best_score:.4f}")

for gen in range(1, GENS+1):
    scores = [fitness(ind) for ind in population]

    # update best
    improved = False
    for ind, sc in zip(population, scores):
        if sc > best_score:
            best_score = sc
            best_subset = ind.copy()
            improved = True

    if improved:
        no_improve = 0
    else:
        no_improve += 1

    print(f"Gen {gen:02d}: best R² = {best_score:.4f}")

    if no_improve >= EARLY_STOP:
        print(f"\n🛑 Early stopping: no improvement for {EARLY_STOP} generations.")
        break

    # breed next generation
    new_pop = [best_subset.copy()]  # elitism
    while len(new_pop) < POP_SIZE:
        # tournament selection
        p1 = max(random.sample(list(zip(population, scores)), TOURN_SIZE),
                 key=lambda x: x[1])[0]
        p2 = max(random.sample(list(zip(population, scores)), TOURN_SIZE),
                 key=lambda x: x[1])[0]

        # crossover
        if random.random() < CXPB:
            cut = K//2
            child = p1[:cut] + [a for a in p2 if a not in p1[:cut]]
            child = child[:K]
        else:
            child = p1.copy()

        # mutation
        if random.random() < MUTPB:
            rem = random.choice(child)
            pool = [a for a in all_ions if a not in child]
            child.remove(rem)
            child.append(random.choice(pool))

        new_pop.append(child)

    population = new_pop

# 5) REPORT & SAVE

print(f"\nBest 15-anion panel (R² = {best_score:.4f}):", best_subset)
pd.DataFrame({ID_COL: best_subset}) \
  .to_csv(f"genetic_{K}_reps.csv", index=False)
print(f"Saved genetic_{K}_reps.csv")
