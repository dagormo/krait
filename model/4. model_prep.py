import pandas as pd
from sklearn.model_selection import train_test_split


def clean_and_encode_categoricals(df, categorical_columns, rare_thresh=0, prefix_sep='_'):
    df = df.copy()
    for col in categorical_columns:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna('Unknown').astype(str).str.strip()
        if rare_thresh > 0:
            counts = df[col].value_counts()
            rare = counts[counts < rare_thresh].index
            df[col] = df[col].apply(lambda x: 'Other' if x in rare else x)
    df = pd.get_dummies(df, columns=categorical_columns, prefix_sep=prefix_sep)
    return df


# ----- SETTINGS -----
eluents = ['hydroxide']  # , 'carbonate', 'msa']
for eluent in eluents:

    target_column = 'log(k)'
    df = pd.read_csv(f'merged_{eluent}.csv')

    # ----- CLEAN IDENTIFIERS -----
    drop_cols = ['SMILES', 'Name', 'Filename', 'Retention time', 'Peak Area', 'Flow rate', 'Asymmetry', 'ESI',
                 '1/T', 'Void', 'Plates', 'Column i.d.', 'Column length', 'Chemistry', 'log_hydrophobicity',
                 'Eluent', 'Functional group characteristics', 'Analyte']   # drop as needed
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # ----- SPLIT TARGET & FEATURES -----
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ----- CLEAN + ONE-HOT ENCODE CATEGORICALS -----
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_cols:
        print(f"Cleaning and encoding categorical columns: {categorical_cols}")
        X = clean_and_encode_categoricals(X, categorical_cols, rare_thresh=1)

        # ----- SPLIT TRAIN/TEST -----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----- SAVE OUTPUTS -----
    X_train.to_csv(f'../data/X_{eluent}_train.csv', index=False)
    X_test.to_csv(f'../data/X_{eluent}_test.csv', index=False)
    y_train.to_csv(f'../data/y_{eluent}_train.csv', index=False)
    y_test.to_csv(f'../data/y_{eluent}_test.csv', index=False)
