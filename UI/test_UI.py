import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import joblib
import os

# --- Load models and template ---
RESOURCES = "resources"
PCA_MODEL_PATH = os.path.join(RESOURCES, "pca_model_95pct.pkl")
MODEL_PATH = os.path.join(RESOURCES, "logk_model_hydroxide.pkl")
DESCRIPTOR_TEMPLATE = os.path.join(RESOURCES, "descriptors_noredundancy.csv")
_expected_padel = pd.read_csv(DESCRIPTOR_TEMPLATE, nrows=0).columns.drop(["Name", "SMILES"])

# --- UI Layout ---
st.set_page_config(page_title="Krait Predictor", layout="wide")
st.title("🔬 Krait Retention Time Predictor")

# --- Input Form ---
with st.form("compound_form"):
    st.header("1. Enter Compound Information")
    smiles = st.text_input("SMILES string", "CC(=O)[O-]")
    name = st.text_input("Compound name", "Acetate")

    st.header("2. Enter Method and Column Conditions")
    flow = st.number_input("Flow rate (mL/min)", value=1.0)
    temp = st.number_input("Temperature (°C)", value=30.0)
    start = st.number_input("Start Concentration (mM)", value=2.0)
    slope = st.number_input("Gradient slope (mM/min)", value=1.0)
    dp = st.number_input("Particle diameter (µm)", value=4.0)
    cap = st.number_input("Column capacity (µeq)", value=290.0)
    latex_d = st.number_input("Latex diameter (nm)", value=150.0)
    latex_x = st.number_input("Latex cross-linking (frac)", value=0.08)
    t0 = st.number_input("Void time (min)", value=2.8)
    hydro = st.number_input("Column hydrophobicity", value=1.6)

    st.header("3. Column Functional Group and Resin")
    fg_label = st.selectbox("Functional group", [
        "Alkanol quaternary ammonium",
        "Alkyl/alkanol quaternary ammonium",
        "Unknown"])
    r_label = st.selectbox("Resin composition", [
        "Unknown", "microporous", "super macroporous"])

    submitted = st.form_submit_button("🔮 Predict Retention Time")

if submitted:
    from main import get_combined_descriptors, apply_pca, classify_ion_type

    try:
        desc, mol = get_combined_descriptors(smiles, name)
        x_pca, pc_names = apply_pca(desc, PCA_MODEL_PATH)

        # Categorical columns
        ion = classify_ion_type(mol)
        ion_vector = {col: 1.0 if ion in col else 0.0 for col in ['IonType_anion', 'IonType_neutral']}
        fg_vector = {f"Functional group_{x}": 1.0 if fg_label == x else 0.0 for x in [
            "Alkanol quaternary ammonium", "Alkyl/alkanol quaternary ammonium", "Unknown"]}
        r_vector = {f"Resin composition_{x}": 1.0 if r_label == x else 0.0 for x in [
            "Unknown", "microporous", "super macroporous"]}

        # Feature vector
        feature_dict = dict(zip(pc_names, x_pca))
        feature_dict.update({
            "Flow rate": flow,
            "Temperature": temp,
            "Start Concentration": start,
            "Gradient slope": slope,
            "Particle diameter": dp,
            "Column capacity": cap,
            "Latex diameter": latex_d,
            "Latex x-linking": latex_x,
            "Hydrophobicity": hydro
        })
        feature_dict.update(ion_vector)
        feature_dict.update(fg_vector)
        feature_dict.update(r_vector)

        model = joblib.load(MODEL_PATH)

        feature_names = model.feature_names_in_
        feature_row = [feature_dict.get(col, 0.0) for col in feature_names]
        X = pd.DataFrame([feature_row], columns=feature_names)

        logk = model.predict(X)[0]
        k = 10 ** logk
        v0 = t0 * flow
        vR = v0 * (1+k)
        tR = vR/flow

        st.success(f"Predicted log(k): {logk:.3f}")
        st.success(f"Estimated retention time: {tR:.2f} min")

        st.image(Draw.MolToImage(mol, size=(300, 300)), caption=name)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
