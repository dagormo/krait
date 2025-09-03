import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA = os.path.join(PROJECT_ROOT, "data")
PKL = os.path.join(PROJECT_ROOT, "pkl")

PCA_MODEL_PATH = os.path.join(PKL, "pca_model_95pct.pkl")
MODEL_PATH = os.path.join(PKL, "logk_model_hydroxide.pkl")