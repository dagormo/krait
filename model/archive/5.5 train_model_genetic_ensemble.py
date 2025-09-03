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

POP_SIZE   = 30
GENS       = 40
K          = 15
TOURN_SIZE = 3
CXPB       = 0.8
MUTPB      = 0.3
SEED       = 25
EARLY_STOP = 12  # Stop if no improvement after this many generations

# Top-3 panels to build union sampling pool
panel_42 = ['Tartrate','Citraconate','Ethanesulfonate','Mesaconate','Sucrose','Chromate',
            'Trifluroacetate','Fluoroacetate','Benzoate','Malonate','Sorbose','Benzenesulfonate',
            'Bromoacetate','Pyruvate']

panel_1  = ['Quinate','Caprylate','Bromide','Succinate','Glucose','Nitrate','Difluoroacetate',
            'Mannosamine','Maltitol','Citrate','Melibiose','p-Chlorobenzenesulfonate',
            'Bromoacetate','Pyrophosphate']

panel_22 = ['Acetate','Oxalate','Trimetaphosphate','Tripolyphosphate','Lactose','Nitrate',
            'Fluoride','Glucosamine','Bromide','Phenylacetate','Sorbitol','Chlorite',
            'Dibromoacetate','Pyrophosphate']

union_pool = sorted(set(panel_42 + panel_1 + panel_22))

print(f"Union panel size: {len(union_pool)}")

# ─── SEED FIXING ────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)

# ─── 1) LOAD & CLEAN ─────────────────────────────────────────────────────────────
runs = pd.read_csv(MERGED_CSV)
runs.columns = runs.columns.str.strip()
if ALT_ID in runs.columns:
    runs = runs.rename(columns={ALT_ID: ID_COL})
runs = runs.loc[:, ~runs.columns.duplicated()]

# detect PC columns
pc_pattern = re.compile(r'^PC\d+$', re.IGNORECASE)
pc_cols = [c for c in runs.columns if pc_pattern.match(c)]
if not pc_cols:
    raise RuntimeError(f"No PC columns found in {MERGED_CSV}; got {runs.columns.tolist()}")

df = runs.copy().reset_index(drop=True)

# ─── 2) BUILD MATRICES ───────────────────────────────────────────────────────────
drop_cols = [
    'Name', 'SMILES', TARGET, 'Filename', 'Retention time', 'Peak Area', 'Asymmetry', 'ESI',
    '1/T', 'Void', 'Plates', 'Column i.d.', 'Column length', 'Chemistry', 'log_hydrophobicity',
    'Eluent', 'Functional group characteristics', 'IonType'
]

X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])
cat_cols = X_raw.select_dtypes(include=['object','category']).columns.tolist()
X_all = pd.get_dummies(X_raw, columns=cat_cols, prefix_sep='_').values
y_all = df[TARGET].values
names = df[ID_COL].tolist()

# ─── 3) DEFINE FITNESS FUNCTION ─────────────────────────────────────────────────
def fitness(subset):
    mask = df[ID_COL].isin(subset)
    X_tr, y_tr = X_all[mask], y_all[mask]
    X_te, y_te = X_all[~mask], y_all[~mask]
    mdl = HistGradientBoostingRegressor(max_iter=500, random_state=SEED)
    mdl.fit(X_tr, y_tr)
    return r2_score(y_te, mdl.predict(X_te))

# ─── 4) PREP SEED PANEL ─────────────────────────────────────────────────────────
unique_ions = df[[ID_COL] + pc_cols].drop_duplicates(subset=ID_COL).reset_index(drop=True)
valid_set = set(unique_ions[ID_COL].values)
all_ions = [a for a in union_pool if a in valid_set]

if len(all_ions) < K:
    raise ValueError(f"Not enough valid analytes in union pool to sample {K}.")

seed_panel = random.sample(all_ions, K)
print(f"Using random seed panel from union pool:\n{seed_panel}")

# ─── 5) RUN GENETIC ALGORITHM ───────────────────────────────────────────────────
print(f"First few names in df[{ID_COL}]:", df[ID_COL].unique()[:5])
print(f"Example from seed_panel:", seed_panel[0])

no_improve = 0
population = [seed_panel] + [
    random.sample(all_ions, K)
    for _ in range(POP_SIZE - 1)
]

best_subset = population[0].copy()
best_score = fitness(best_subset)
print(f"\nGen  0 (seed): best R² = {best_score:.4f}")

for gen in range(1, GENS+1):
    scores = [fitness(ind) for ind in population]

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
        p1 = max(random.sample(list(zip(population, scores)), TOURN_SIZE), key=lambda x: x[1])[0]
        p2 = max(random.sample(list(zip(population, scores)), TOURN_SIZE), key=lambda x: x[1])[0]

        if random.random() < CXPB:
            cut = K // 2
            child = p1[:cut] + [a for a in p2 if a not in p1[:cut]]
            child = child[:K]
        else:
            child = p1.copy()

        if random.random() < MUTPB:
            rem = random.choice(child)
            pool = [a for a in all_ions if a not in child]
            child.remove(rem)
            child.append(random.choice(pool))

        new_pop.append(child)

    population = new_pop

# ─── 6) SAVE BEST PANEL ─────────────────────────────────────────────────────────
print(f"\nBest 15-anion panel (R² = {best_score:.4f}):", best_subset)
pd.DataFrame({ID_COL: best_subset}) \
  .to_csv(f"genetic_{K}_reps.csv", index=False)
print(f"Saved genetic_{K}_reps.csv")
