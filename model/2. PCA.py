from rdkit import Chem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib


# ─── helper to classify by formal charge ──────────────────────────────────────
def classify_by_charge(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown"
    net_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if net_charge < 0:
        return "anion"
    elif net_charge > 0:
        return "cation"
    else:
        return "neutral"
# ───────────────────────────────────────────────────────────────────────────────


# Load descriptor matrix
df = pd.read_csv(r"..\data\descriptors_noredundancy.csv")

# Add IonType via SMILES
df["IonType"] = df["SMILES"].apply(classify_by_charge)

# Separate ID + IonType
id_columns = ["SMILES", "Name", "IonType"]
id_df = df[id_columns] if all(col in df.columns for col in id_columns) else pd.DataFrame()

# Build and clean feature matrix
X = df.drop(columns=id_columns, errors="ignore")
X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Scree plot
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.axhline(0.95, color="red", linestyle="--", label="95%")
plt.xlabel("Components")
plt.ylabel("Cumulative var")
plt.legend()
plt.tight_layout()
plt.savefig(r"..\data\pca_scree_plot.png", dpi=300)

# PCA to 95%
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Find index of acetate
idx = df[df["Name"] == "Acetate"].index[0]

# Get corresponding PCA vector
acetate_pca_vector = X_pca[idx]

# Print it
print("\n🎯 acetate projected PCA vector:")
print(np.round(acetate_pca_vector, 4))


# Optional: save loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)
loadings.to_csv(r"..\data\pca_loadings.csv")

# Build PCA-space DataFrame
pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
if not id_df.empty:
    pca_df["SMILES"] = id_df["SMILES"].values
    pca_df["Name"] = id_df["Name"].values
    pca_df["IonType"] = id_df["IonType"].values

pca_df.to_csv(r"..\data\pca_space_95pct.csv", index=False)

pca_package = {
    "scaler": scaler,
    "pca": pca,
    "columns": list(X.columns),  # Save original feature names for alignment
}

joblib.dump(pca_package, "../pkl/pca_model_95pct.pkl")
