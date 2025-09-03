import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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

# __________________________________________________________________________________________________
held_out_chemistry = 'AS11'
target_column = 'log(k)'
# ___________________________________________________________________________________________________

df = pd.read_csv(f'merged_{eluent}.csv')

drop_cols = ['SMILES', 'Name', 'Filename', 'Retention time', 'Peak Area', 'Asymmetry', 'ESI', '1/T', 'Void', 'Plates', 'Column i.d.', 'Column length', 'log_hydrophobicity','Eluent','Functional group characteristics']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Split data into train (excluding held-out chemistry) and test (held-out chemistry)
train_df = df[df['Chemistry'] != held_out_chemistry]
test_df = df[df['Chemistry'] == held_out_chemistry]

# Subsample 20% of the training data
train_df = train_df.sample(frac=0.2, random_state=42)

# Separate features and targets
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]
X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

# Encode categoricals
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    print(f"[{eluent}] Encoding categoricals: {categorical_cols}")
    X_train = clean_and_encode_categoricals(X_train, categorical_cols, rare_thresh=1)
    X_test = clean_and_encode_categoricals(X_test, categorical_cols, rare_thresh=1)

# Align columns between train/test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Save files
X_train.to_csv(f'X_{eluent}_train.csv', index=False)
X_test.to_csv(f'X_{eluent}_test.csv', index=False)
y_train.to_csv(f'y_{eluent}_train.csv', index=False)
y_test.to_csv(f'y_{eluent}_test.csv', index=False)
