import streamlit as st
import pandas as pd
import numpy as np

import joblib

from engine import (
    build_conc_function,
    calibrate_velocity,
    predict_rt,
    preprocess_analyte,
    predict_with_preprocessed,
    DESCRIPTOR_TEMPLATE
)
from api import (
    PCA_MODEL_PATH,
    LOGK_MODEL_PATH
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_expected_padel = pd.read_csv(DESCRIPTOR_TEMPLATE, nrows=0).columns.drop(["Name", "SMILES"])

# --- UI Layout ---
st.set_page_config(page_title="Krait Predictor", layout="wide")
st.title("Krait Retention Time Predictor with Multistep Gradient")


def parse_gradient_input(text):
    pts = []
    for line in text.strip().splitlines():
        try:
            t, c = map(float, line.split(','))
            pts.append((t, c))
        except:
            st.warning(f"Invalid line: {line}")
    return pts


# --- Input Form ---
with st.form("compound_form"):
    st.header("1. Enter Compound List")
    smiles_text = st.text_area("Paste compound name and SMILES (one per line, tab separated)",
                               "Myo-inositol-1,4,5-phosphate\t"
                               "C1[C@@H](O)[C@H](O[P](=O)([O-])[O-])[C@@H](O)[C@H](O[P](=O)([O-])[O-])[C@H]1OP(=O)([O-])[O-]\n"
                               "Caffeic acid\t"
                               "OC(=C/C=C/c1ccc(O)c(O)c1)C([O-])=O")
    st.header("2. Column and Method Conditions")
    flow = st.number_input("Flow rate (mL/min)", value=1.0)
    temp = st.number_input("Temperature (°C)", value=30.0)
    dp = st.number_input("Particle diameter (µm)", value=4.0)
    cap = st.number_input("Column capacity (µeq)", value=290.0)
    latex_d = st.number_input("Latex diameter (nm)", value=70.0)
    latex_x = st.number_input("Latex cross-linking (frac)", value=0.08)
    t0 = st.number_input("Void time (min)", value=2.8)
    hydro = st.number_input("Column hydrophobicity", value=1.6)
    column_length = st.number_input("Column length (mm)", value=250.0)

    st.header("3. Column Functional Group and Resin")
    fg_label = st.selectbox("Functional group", [
        "Alkanol quaternary ammonium",
        "Alkyl/alkanol quaternary ammonium",
        "Unknown"])
    r_label = st.selectbox("Resin composition", [
        "super macroporous", "microporous", "Unknown"])

    st.header("4. Multistep Gradient Program")
    st.markdown("Enter gradient steps as time,concentration per line (e.g. `0,5`)")
    gradient_text = st.text_area("Gradient profile", value="0,5\n1,5\n25,100\n30,100\n30,5")

    submitted = st.form_submit_button("🔮 Predict Retention Times")

if submitted:

    name_smiles_pairs = []
    for line in smiles_text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2:
            name, smiles = parts
        elif len(parts) == 1:
            name = smiles = parts[0]  # fallback if no tab
        else:
            st.warning(f"Invalid input line (too many tabs?): {line}")
            continue
        name_smiles_pairs.append((name.strip(), smiles.strip()))

    gradient_pts = parse_gradient_input(gradient_text)
    conc_at, times, concs = build_conc_function(gradient_pts)

    model = joblib.load(LOGK_MODEL_PATH)
    feature_names = model.feature_names_in_

    predictions = []

    for name, smiles in name_smiles_pairs:
        try:
            conditions = {
                "Flow rate": flow,
                "Temperature": temp,
                "Particle diameter": dp,
                "Column capacity": cap,
                "Latex diameter": latex_d,
                "Latex x-linking": latex_x,
                "Hydrophobicity": hydro,
                "Column length": column_length,
                "Functional group": fg_label,
                "Resin composition": r_label,
            }

            base_features, mol = preprocess_analyte(smiles, name, PCA_MODEL_PATH)
            concs_iso = [5.0, 10.0, 30.0]
            rts_iso = []
            for c in concs_iso:
                conds = conditions.copy()
                conds["Start Concentration"] = c
                conds["Gradient slope"] = 0.0
                logk, tR = predict_with_preprocessed(base_features, conds, LOGK_MODEL_PATH, t0)
                rts_iso.append(tR)
            rts_iso = np.array(rts_iso)

            velocity = calibrate_velocity(concs_iso, rts_iso, t0, column_length)
            rt_obs, rt_adj = predict_rt(conc_at, velocity, column_length, t0)

            st.markdown(f"### {name}")
            st.success(f"Estimated retention time: {rt_obs:.2f} min")
            predictions.append((name, rt_obs))

        except Exception as e:
            st.error(f"Error processing {smiles}: {e}")

    # --- Plot simulated chromatogram ---
    if predictions:
        pred_df = pd.DataFrame(predictions, columns=["Compound", "Predicted_RT_min"])
        pred_df.to_csv("predicted_retention_times.csv", index=False)
        st.success("Predicted retention times saved to predicted_retention_times.csv")

        predictions.sort(key=lambda x: x[1])
        times = np.linspace(0, max([rt for _, rt in predictions]) + 2, 2000)
        chromatogram = np.zeros_like(times)
        peak_width = 15 / 60  # 15 seconds in minutes
        sigma = peak_width / 2.355

        for name, rt in predictions:
            peak = np.exp(-0.5 * ((times - rt) / sigma) ** 2)
            chromatogram += peak

        st.download_button(
            label="Download predictions as CSV",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="predicted_retention_times.csv",
            mime="text/csv"
        )

        fig, ax = plt.subplots()
        ax.plot(times, chromatogram, label="Simulated Chromatogram")
        for label, rt in predictions:
            ax.axvline(rt, linestyle=':', color='gray', alpha=0.5)
            ax.text(rt, 1.02, label, rotation=90, fontsize=8, ha='center', va='bottom')
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Signal")
        ax.set_title("Predicted Chromatogram")
        st.pyplot(fig)
        plt.close(fig)
