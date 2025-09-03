import os
import subprocess
import pandas as pd
import numpy as np
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
import importlib
from mordred import Calculator, descriptors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

PADEL_JAR_PATH = os.path.join(DATA_DIR, "PaDEL-Descriptor", "PaDEL-Descriptor.jar")
DESCRIPTOR_TEMPLATE = os.path.join(DATA_DIR, "descriptors_noredundancy.csv")


# Load template for expected PaDEL features
_expected_padel = None

def get_expected_padel():
    global _expected_padel
    if _expected_padel is None:
        template_df = pd.read_csv(DESCRIPTOR_TEMPLATE, nrows=0)
        _expected_padel = [c for c in template_df.columns if c not in ("Name", "SMILES")]
    return _expected_padel


def get_padel_descriptors_from_smi(smiles: str, name: str, padel_jar: str = PADEL_JAR_PATH):
    if not _expected_padel:
        get_expected_padel()
    with tempfile.TemporaryDirectory() as temp_dir:
        smi_path = os.path.join(temp_dir, "input.smi")
        output_csv = os.path.join(temp_dir, "padel_output.csv")

        # Write temporary .smi file
        with open(smi_path, "w") as f:
            f.write(f"{smiles} {name}\n")

        # Run PaDEL-Descriptor
        subprocess.run([
            "java", "-jar", padel_jar,
            "-dir", temp_dir,
            "-file", output_csv,
            "-removesalt", "-standardizenitro",
            "-2d", "-3d", "-fingerprints", "false"
        ], check=True)

        df = pd.read_csv(output_csv)
        raw = df.iloc[0].to_dict()

        clean = {}
        for k, v in raw.items():
            try:
                val = float(v)
            except:
                val = np.nan
            clean[k] = val

    return clean


def get_mordred_descriptors(mol):
    calc = Calculator(descriptors, ignore_3D=False)
    result = calc(mol)
    desc_dict = result.asdict()
    clean = {}
    for k, v in desc_dict.items():
        try:
            if pd.notnull(v) and np.isfinite(v):
                clean[k] = float(v)
        except:
            continue
    return clean


def smiles_to_mol(smiles: str, name: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES.")
    mol = Chem.AddHs(mol)
    mol.SetProp("_Name", name)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    return mol


def ensure_full_vector(d: dict, expected_keys, suffix="_padel", default=np.nan):
    for k in expected_keys:
        d.setdefault(f"{k}{suffix}", default)
    return d


descriptor_modules = [
    'mordred.ABC',
    'mordred.AtomType',
    'mordred.Autocorrelation',
    'mordred.BaryszMatrix',
    'mordred.BondType',
    'mordred.BurdenModifiedEigenvalue',
    'mordred.CarbonTypes',
    'mordred.Chi',
    'mordred.CircularFingerprint',
    'mordred.Constitutional',
    'mordred.DetourMatrix',
    'mordred.DistanceMatrix',
    'mordred.EState',
    'mordred.FunctionalGroup',
    'mordred.HBond',
    'mordred.InformationContent',
    'mordred.Ipc',
    'mordred.KappaShapeIndex',
    'mordred.MOE',
    'mordred.Moran',
    'mordred.MQNs',
    'mordred.PathCount',
    'mordred.PEOE_VSA',
    'mordred.RDF',
    'mordred.RingCount',
    'mordred.RotatableBonds',
    'mordred.Spacial',
    'mordred.SpanningTree',
    'mordred.SpMax',
    'mordred.TopoPSA',
    'mordred.VE3',
    'mordred.VSA_EState',
    'mordred.WHIM',
    'mordred._3D'
]

for module_name in descriptor_modules:
    try:
        importlib.import_module(module_name)
    except ImportError:
        pass  # Silently skip missing ones

calc = Calculator(descriptors, ignore_3D=False)


def classify_ion_type(mol):
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if charge > 0:
        return "cation"
    elif charge < 0:
        return "anion"
    else:
        return "neutral"


def get_combined_descriptors(smiles: str, name: str):
    mol = smiles_to_mol(smiles, name)

    # 1) Pull raw descriptor dicts
    mordred = get_mordred_descriptors(mol)
    padel = get_padel_descriptors_from_smi(smiles, name)

    # 2) Inject ID fields
    mordred["Name"] = name
    mordred["SMILES"] = smiles
    padel["Name"] = name
    padel["SMILES"] = smiles

    # 3) Build one-row DataFrames
    mordred_df = pd.DataFrame([mordred])
    padel_df = pd.DataFrame([padel])

    # 4) Suffix all descriptor columns
    mordred_df.rename(columns={c: f"{c}_mordred" for c in mordred_df.columns if c not in ("Name", "SMILES")},
                      inplace=True)
    padel_df.rename(columns={c: f"{c}_padel" for c in padel_df.columns if c not in ("Name", "SMILES")}, inplace=True)

    # 5) Merge on IDs
    combined_df = pd.merge(mordred_df, padel_df, on=["Name", "SMILES"], how="inner")

    # 6) Flatten to dict and ensure full padel vector
    combined = combined_df.iloc[0].to_dict()
    combined = ensure_full_vector(combined, get_expected_padel(), suffix="_padel", default=np.nan)

    return combined, mol
