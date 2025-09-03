import joblib
import pandas as pd


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
