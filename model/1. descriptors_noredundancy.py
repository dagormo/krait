import subprocess
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors

# --- Inputs ---
SMI_FILE = r"../data/PaDEL-Descriptor/smi_files/input_OH.smi"
PADEL_JAR = r"../data/PaDEL-Descriptor/PaDEL-Descriptor.jar"
PADEL_OUT = r"../data/padel_output.csv"

# --- Load SMILES ---
smiles, names = [], []
with open(SMI_FILE) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            smiles.append(parts[0])
            names.append(parts[1])

# --- Mordred descriptors ---
calc = Calculator(descriptors, ignore_3D=False)
mordred_data, descriptor_names = [], None

for smi, name in zip(smiles, names):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    result = calc(mol)
    desc_keys = [str(d) for d in result.keys()]
    if descriptor_names is None:
        descriptor_names = desc_keys
    row = [smi, name] + [result[k] for k in desc_keys]
    mordred_data.append(row)

mordred_df = pd.DataFrame(mordred_data, columns=["SMILES", "Name"] + descriptor_names)

# --- PaDEL descriptors ---
subprocess.run([
    "java", "-jar", PADEL_JAR,
    "-dir", r"../data/PaDEL-Descriptor/smi_files",
    "-file", PADEL_OUT,
    "-2d", "-3d", "-removesalt", "-standardizenitro"
], check=True)

padel_df = pd.read_csv(PADEL_OUT)

# --- Merge ---
mordred_df.rename(
    columns={c: f"{c}_mordred" for c in mordred_df.columns if c not in ("SMILES", "Name")},
    inplace=True
)
padel_df.rename(
    columns={c: f"{c}_padel" for c in padel_df.columns if c not in ("SMILES", "Name")},
    inplace=True
)

combined = pd.merge(
    mordred_df, padel_df, on="Name", how="inner"
)
print(f"✅ Merged: {combined.shape}")

# --- Filtering ---
threshold = 0.2
null_ratio = combined.isnull().mean()
good_cols = null_ratio[null_ratio <= threshold].index
filtered = combined[good_cols]

# Ensure SMILES and Name stay
if "SMILES" not in filtered.columns:
    filtered.insert(0, "SMILES", combined["SMILES"])
if "Name" not in filtered.columns:
    filtered.insert(1, "Name", combined["Name"])

filtered.to_csv("../data/combined_descriptors_filtered.csv", index=False)

# --- Remove redundant features ---
df = filtered.copy()
id_cols = ["SMILES", "Name"]
id_df = df[id_cols]
features = df.drop(columns=id_cols).select_dtypes(include=[np.number])

# Remove constant columns
features = features.loc[:, (features != features.iloc[0]).any()]
features = features.loc[:, features.apply(lambda x: not ((x.fillna(0) == 0).all()))]

# Correlation filtering
corr_matrix = features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
dedup = features.drop(columns=to_drop)

# Final merge and save
final = pd.concat([id_df.reset_index(drop=True), dedup.reset_index(drop=True)], axis=1)
final.to_csv(r"../data/descriptors_noredundancy.csv", index=False)
