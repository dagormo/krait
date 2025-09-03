import pandas as pd
import joblib
from .pca import apply_pca
from .descriptors import get_combined_descriptors, classify_ion_type


def build_feature_vector(smiles, name, conditions, pca_model_path, model_path):
    """
    Build a feature vector for prediction from SMILES and method conditions.
    conditions = dict with keys like Flow rate, Temperature, etc.
    """
    desc, mol = get_combined_descriptors(smiles, name)
    x_pca, pc_names = apply_pca(desc, pca_model_path)

    # Ion type
    ion = classify_ion_type(mol)
    ion_vector = {col: 1.0 if ion in col else 0.0 for col in ['IonType_anion', 'IonType_neutral']}

    # Combine everything
    feature_dict = dict(zip(pc_names, x_pca))
    feature_dict.update(conditions)
    feature_dict.update(ion_vector)

    # Align with model
    model = joblib.load(model_path)
    feature_names = model.feature_names_in_
    feature_row = [feature_dict.get(col, 0.0) for col in feature_names]
    X = pd.DataFrame([feature_row], columns=feature_names)

    return X, model


def predict_single(smiles, name, conditions, pca_model_path, model_path, t0):
    """
    Predict log(k) and retention time for a single analyte.
    """
    X, model = build_feature_vector(smiles, name, conditions, pca_model_path, model_path)
    logk = model.predict(X)[0]
    k = 10 ** logk
    v0 = t0 * conditions["Flow rate"]
    vR = v0 * (1 + k)
    tR = vR / conditions["Flow rate"]
    return logk, tR


def predict_retention(smiles, name, conditions, pca_model_path, model_path, t0):
    X, model = build_feature_vector(smiles, name, conditions, pca_model_path, model_path)
    logk = model.predict(X)[0]
    k = 10 ** logk
    v0 = t0 * conditions["Flow rate"]
    vR = v0 * (1 + k)
    tR = vR / conditions["Flow rate"]
    return logk, tR