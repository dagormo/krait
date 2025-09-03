import matplotlib
matplotlib.use("Agg")
import os
import sys
import subprocess
import pandas as pd
import numpy as np
import tempfile
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
import importlib
import matplotlib.pyplot as plt
from mordred import Calculator, descriptors

# Dynamically import all descriptor modules so Calculator(descriptors) works even when frozen
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(BASE_DIR, "resources")

PADEL_JAR_PATH = os.path.join(RESOURCES, "PaDEL-Descriptor.jar")
PCA_MODEL_PATH = os.path.join(RESOURCES, "pca_model_95pct.pkl")
MODEL_PATH = os.path.join(RESOURCES, "logk_model_hydroxide.pkl")
DESCRIPTOR_TEMPLATE = os.path.join(RESOURCES, "descriptors_noredundancy.csv")

# Load template for expected PaDEL features
template_df = pd.read_csv(DESCRIPTOR_TEMPLATE, nrows=0)
_expected_padel = [c for c in template_df.columns if c not in ("Name", "SMILES")]


def ensure_full_vector(d: dict, expected_keys, suffix="_padel", default=np.nan):
    """
    Ensure that every key in expected_keys with the given suffix exists in d, inserting default values if missing.
    """
    for k in expected_keys:
        d.setdefault(f"{k}{suffix}", default)
    return d


# === Helper: Convert SMILES to 3D molecule ===
def smiles_to_mol(smiles: str, name: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES.")
    mol = Chem.AddHs(mol)
    mol.SetProp("_Name", name)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    return mol


# === Descriptor: Mordred ===
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


# === Descriptor: PaDEL ===
def get_padel_descriptors_from_smi(smiles: str, name: str, padel_jar: str = PADEL_JAR_PATH):
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


# === Combined Descriptor Dictionary ===
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
    mordred_df.rename(columns={c: f"{c}_mordred" for c in mordred_df.columns if c not in ("Name", "SMILES")}, inplace=True)
    padel_df.rename(columns={c: f"{c}_padel" for c in padel_df.columns if c not in ("Name", "SMILES")}, inplace=True)

    # 5) Merge on IDs
    combined_df = pd.merge(mordred_df, padel_df, on=["Name", "SMILES"], how="inner")

    # 6) Flatten to dict and ensure full padel vector
    combined = combined_df.iloc[0].to_dict()
    combined = ensure_full_vector(combined, _expected_padel, suffix="_padel", default=np.nan)

    return combined, mol


# === PCA Projection ===
def apply_pca(desc_dict: dict, pca_model_path: str):
    pca_package = joblib.load(pca_model_path)
    scaler, pca, cols = (pca_package["scaler"], pca_package["pca"], pca_package["columns"])

    # Report and fill missing
    missing = [c for c in cols if c not in desc_dict]
    for c in missing:
        desc_dict[c] = 0.0

    # Build DataFrame for scaler
    xdf = pd.DataFrame([{col: desc_dict[col] for col in cols}])
    scaled = scaler.transform(xdf)
    pcs = pca.transform(scaled)
    return pcs[0], [f"PC{i+1}" for i in range(pcs.shape[1])]


# === Charge classification ===
def classify_ion_type(mol):
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if charge > 0:
        return "cation"
    elif charge < 0:
        return "anion"
    else:
        return "neutral"


# === Helper for flexible input ===
def prompt_float(field: str, helper: str = None):
    prompt = f"{field}"
    if helper:
        prompt += f" ({helper})"
    prompt += ": "
    val = input(prompt).strip()
    if val == "":
        return np.nan
    try:
        return float(val)
    except:
        print("⚠️ Invalid number. Try again.")
        return prompt_float(field, helper)


# === Method-3 gradient utilities ===
def calibrate_velocity(concs, rts, t0, L):
    t_prime = rts - t0
    v_iso = L / t_prime
    return np.dot(concs, v_iso) / np.dot(concs, concs)


def build_conc_function(points):
    times, concs = zip(*points)
    return (lambda t: np.interp(t, times, concs)), np.array(times), np.array(concs)


def predict_rt(conc_at, k, L, t0, dt=0.01, t_max=None):
    if t_max is None:
        t_max = times[-1] + 100
    t, L_acc = 0.0, 0.0
    while L_acc < L and t < t_max:
        L_acc += (k * conc_at(t)) * dt
        t += dt
    return t+t0, t


def plot_gradient(times, concs, rt_adj):
    plt.figure(figsize=(8, 4))
    plt.plot(times, concs, '-', linewidth=2)
    plt.axvline(rt_adj, color='gray', linestyle='--', label=f'Elution at {rt_obs:.2f} min (predicted)')
    plt.xlabel("Time (min)")
    plt.ylabel("Eluent Concentration (mM)")
    plt.title("Multi-Step Gradient Profile")
    plt.legend(loc='best')
    plt.tight_layout()
    fig = plt.gcf()
    plt.close(fig)
    return fig


# === Full Pipeline Entry ===
if __name__ == "__main__":
    smiles = input("Enter SMILES: ").strip()
    name = input("Enter compound name: ").strip()

    try:
        desc, mol = get_combined_descriptors(smiles, name)
        x_pca, pc_names = apply_pca(desc, PCA_MODEL_PATH)

        flow = prompt_float("Flow rate", "mL/min")
        temp = prompt_float("Temperature", "°C")
        dp = prompt_float("Particle diameter", "µm")
        cap = prompt_float("Column capacity", "µeq")
        latex_d = prompt_float("Latex diameter", "nm")
        latex_x = prompt_float("Latex x-linking", "%")
        void_t = prompt_float("Void time", "min")
        hydro = prompt_float("Column hydrophobicity", "")
        Lc = prompt_float("Column length", "mm")

        # === IonType ===
        ion = classify_ion_type(mol)
        iontype_cols = ['IonType_anion', 'IonType_neutral']
        ion_vector = {col: 1.0 if ion in col else 0.0 for col in iontype_cols}

        # === Functional group ===
        fg_options = [
            "Alkanol quaternary ammonium",
            "Alkyl/alkanol quaternary ammonium",
            "Unknown"
        ]
        print("\n📘 Functional group options:")
        for i, fg in enumerate(fg_options, 1):
            print(f"  {i}: {fg}")
        fg_choice = input("Select functional group (1-3): ").strip()
        fg_label = fg_options[int(fg_choice) - 1] if fg_choice in ['1', '2', '3'] else "Unknown"
        fg_cols = [f"Functional group_{x}" for x in fg_options]
        fg_vector = {col: 1.0 if fg_label in col else 0.0 for col in fg_cols}

        # === Resin composition ===
        resin_options = ["Unknown", "microporous", "super macroporous"]
        print("\n📘 Resin composition options:")
        for i, r in enumerate(resin_options, 1):
            print(f"  {i}: {r}")
        r_choice = input("Select resin composition (1-3): ").strip()
        r_label = resin_options[int(r_choice) - 1] if r_choice in ['1', '2', '3'] else "Unknown"
        r_cols = [f"Resin composition_{x}" for x in resin_options]
        r_vector = {col: 1.0 if r_label in col else 0.0 for col in r_cols}

        # === Base feature vector ===
        model = joblib.load(MODEL_PATH)
        feature_names = model.feature_names_in_
        base = dict(zip(pc_names, x_pca))
        base.update({
            'Flow rate': flow, 'Temperature': temp, 'Particle diameter': dp,
            'Column capacity': cap, 'Latex diameter': latex_d, 'Latex x-linking': latex_x,
            'Hydrophobicity': hydro
        })
        base.update(ion_vector)
        base.update(fg_vector)
        base.update(r_vector)

        # Fill missing with 0
        feature_row = [base.get(col, 0.0) for col in feature_names]
        X = pd.DataFrame([feature_row], columns=feature_names)

        # Isocratic calibration
        concs_iso = [5, 10, 30]
        rts_iso = []
        for c in concs_iso:
            feat = base.copy()
            feat['Start Concentration'] = c
            feat['Gradient slope'] = 0.0
            feat['Column length'] = Lc
            feat['Flow rate'] = flow
            row = [feat.get(n, 0.0) for n in feature_names]
            logk = model.predict(pd.DataFrame([row], columns=feature_names))[0]
            rts_iso.append(void_t * (10 ** logk) + void_t)
        rts_iso = np.array(rts_iso)

        # Velocity calibration
        k = calibrate_velocity(concs_iso, rts_iso, void_t, Lc)

        # Gradient profile
        n = int(input('How many gradient points? '))
        pts = [(0, 5), (1, 5), (25, 100), (30, 100), (30, 5)]
        print("Enter each point as: time(min),concentration(mM)")
        for i in range(n):
            t, c = map(float, input(f' Point {i + 1}: ').split(','))
            pts.append((t, c))
        conc_at, times, concs = build_conc_function(pts)
        rt_obs, rt_adj = predict_rt(conc_at, k, Lc, void_t, dt=0.01, t_max=times[-1] + 100)
        print(f'Predicted RT = {rt_obs:.2f} min')
        plot_gradient(times, concs, rt_adj)

    except Exception as e:
        print(f"❌ Error: {e}")

    input("\nPress Enter to exit…")
