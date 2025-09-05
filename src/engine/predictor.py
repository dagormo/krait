import pandas as pd
import joblib
from .pca import apply_pca
from .descriptors import get_combined_descriptors, classify_ion_type


def build_feature_vector(base_features, conditions, model_path):
    """
    Assemble full feature row for given conditions using preprocessed features.
    """
    feature_dict = dict(base_features)  # copy
    feature_dict.update(conditions)

    model = joblib.load(model_path)
    feature_row = [feature_dict.get(col, 0.0) for col in model.feature_names_in_]
    X = pd.DataFrame([feature_row], columns=model.feature_names_in_)
    return X, model


def predict_with_preprocessed(base_features, conditions, model_path, t0):
    X, model = build_feature_vector(base_features, conditions, model_path)
    logk = model.predict(X)[0]
    k = 10 ** logk
    v0 = t0 * conditions["Flow rate"]
    vR = v0 * (1 + k)
    tR = vR / conditions["Flow rate"]
    return logk, tR