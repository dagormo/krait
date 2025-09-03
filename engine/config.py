import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(BASE_DIR, "resources")
PCA_MODEL_PATH = os.path.join(RESOURCES, "pca_model_95pct.pkl")
MODEL_PATH = os.path.join(RESOURCES, "logk_model_hydroxide.pkl")