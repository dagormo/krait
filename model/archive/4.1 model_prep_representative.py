import pandas as pd

# ─── SETTINGS ─────────────────────────────────────────────────────────────────
ALT_ID_COL  = "Analyte"                   # your merged_* files use this name
ID_COL      = "Name"                      # we'll rename Analyte→Name
TARGET_COL  = "log(k)"

DROP_COLS   = [
    'SMILES', 'Filename', 'Retention time', 'Peak Area', 'Asymmetry',
    'ESI', '1/T', 'Void', 'Plates', 'Column i.d.', 'Column length',
    'Chemistry', 'log_hydrophobicity', 'Eluent','Functional group characteristics'
]
ELUENTS     = ['hydroxide', 'carbonate', 'msa']
# ───────────────────────────────────────────────────────────────────────────────

# load reps list

for eluent in ELUENTS:
    if eluent in ['hydroxide','carbonate']:
        reps_df = pd.read_csv('rep_anions.csv')
    else:
        reps_df = pd.read_csv('rep_cations.csv')

    rep_names = set(reps_df[ID_COL])
    # 1) load and unify identifier
    df = pd.read_csv(f"merged_{eluent}.csv")
    if ALT_ID_COL in df.columns:
        df = df.rename(columns={ALT_ID_COL: ID_COL})
    df = df.reset_index(drop=True)

    # 2) drop everything you truly don't want
    to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=to_drop, errors='ignore')

    # 3) split off the target
    y = df[TARGET_COL]
    df = df.drop(columns=[TARGET_COL])
    df = df.loc[:, ~df.columns.duplicated()]

    # 4) split into reps vs non-reps (still carrying the NAME column)
    is_rep    = df[ID_COL].isin(rep_names)
    df_train  = df[is_rep].copy()
    df_test   = df[~is_rep].copy()
    y_train   = y[is_rep].copy()
    y_test    = y[~is_rep].copy()

    print(f"[{eluent}] train on {len(df_train)} reps; test on {len(df_test)} others")

    # 5) drop the ID column from both subsets—now no risk of Analyte_/Name_ dummy
    df_train = df_train.drop(columns=[ID_COL])
    df_test  = df_test .drop(columns=[ID_COL])

    # 6) concatenate for joint one-hot encoding, IGNORING the old index
    df_train['__is_train'] = True
    df_test ['__is_train'] = False
    df_all  = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # 7) one-hot encode any categoricals in the full block
    cat_cols = df_all.select_dtypes(include=['object','category']).columns.tolist()
    if cat_cols:
        df_all = pd.get_dummies(df_all, columns=cat_cols, prefix_sep='_')

    # 7b) drop any duplicate column names (just in case)
    df_all = df_all.loc[:, ~df_all.columns.duplicated()]

    # 8) split back using .loc (no reindexing!)
    train_mask     = df_all['__is_train'] == True
    df_train_enc   = df_all.loc[train_mask ].drop(columns=['__is_train'])
    df_test_enc    = df_all.loc[~train_mask].drop(columns=['__is_train'])

    # 9) save out
    df_train_enc.to_csv(f"X_{eluent}_train_rep.csv", index=False)
    df_test_enc.to_csv(f"X_{eluent}_test_rep.csv",  index=False)
    y_train.to_csv(f"y_{eluent}_train_rep.csv", index=False)
    y_test.to_csv(f"y_{eluent}_test_rep.csv",  index=False)
