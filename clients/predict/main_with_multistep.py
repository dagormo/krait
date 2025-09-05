import numpy as np

from src.engine import (
    build_conc_function,
    calibrate_velocity,
    predict_rt,
    preprocess_analyte,
    predict_with_preprocessed
)

from src.api import (
    PCA_MODEL_PATH,
    LOGK_MODEL_PATH
)

import matplotlib
matplotlib.use("Agg")

FIELDS = [
    ("Flow rate", "mL/min"),
    ("Temperature", "°C"),
    ("Particle diameter", "µm"),
    ("Column capacity", "µeq"),
    ("Latex diameter", "nm"),
    ("Latex x-linking", "%"),
    ("Void time", "min"),
    ("Column hydrophobicity", ""),
    ("Column length", "mm"),
]


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


def prompt_choice(field, options):
    print(f"\n📘 {field} options:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}: {opt}")
    choice = input(f"Select {field} (1-{len(options)}): ").strip()
    return options[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(options) else options[-1]


# === Full Pipeline Entry ===
if __name__ == "__main__":
    smiles = input("Enter SMILES: ").strip()
    name = input("Enter compound name: ").strip()

    try:
        fg_label = prompt_choice("Functional group", [
            "Alkanol quaternary ammonium",
            "Alkyl/alkanol quaternary ammonium",
            "Unknown"
        ])
        r_label = prompt_choice("Resin composition", [
            "Unknown", "microporous", "super macroporous"
        ])

        conditions = {}
        for field, helper in FIELDS:
            conditions[field] = prompt_float(field, helper)

        void_t = conditions["Void time"]
        Lc = conditions["Column length"]
        base_features, mol = preprocess_analyte(smiles, name, PCA_MODEL_PATH)

        concs_iso = [5, 10, 30]
        rts_iso = []
        for c in concs_iso:
            conds = conditions.copy()
            conds["Start Concentration"] = c
            conds["Gradient slope"] = 0.0
            _, tR = predict_with_preprocessed(base_features, conds, LOGK_MODEL_PATH, void_t)
            rts_iso.append(tR)
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
        plot_gradient(times, concs, rt_adj, rt_obs)

    except Exception as e:
        print(f"Error: {e}")

    input("\nPress Enter to exit…")
