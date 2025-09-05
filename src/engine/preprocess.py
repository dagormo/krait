# engine/preprocess.py
from .descriptors import get_combined_descriptors, classify_ion_type
from .pca import apply_pca


def preprocess_analyte(smiles, name, pca_model_path):
    desc, mol = get_combined_descriptors(smiles, name)
    x_pca, pc_names = apply_pca(desc, pca_model_path)

    # Ion type
    ion = classify_ion_type(mol)
    ion_vector = {col: 1.0 if ion in col else 0.0 for col in ['IonType_anion', 'IonType_neutral']}

    base_features = dict(zip(pc_names, x_pca))
    base_features.update(ion_vector)

    return base_features, mol
