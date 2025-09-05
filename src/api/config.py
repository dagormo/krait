from pathlib import Path
import os
from dotenv import load_dotenv

# load .env if present
load_dotenv()

# Root-relative defaults (no hardcoded absolute paths)
BASE_DIR = Path(os.getenv("KRAIT_PROJECT_ROOT", "")).resolve()
MODELS_DIR = Path(os.getenv("KRAIT_MODELS_DIR", BASE_DIR / "pkl")).resolve()
DATA_DIR = Path(os.getenv("KRAIT_DATA_DIR", BASE_DIR / "data")).resolve()

# Model artifacts (override in .env if needed)
PCA_MODEL_PATH = Path(os.getenv("KRAIT_PCA_MODEL", MODELS_DIR / "pca_model_95pct.pkl")).resolve()
LOGK_MODEL_PATH = Path(os.getenv("KRAIT_LOGK_MODEL", MODELS_DIR / "logk_model_hydroxide.pkl")).resolve()
PADEL_JAR_PATH = os.path.join(DATA_DIR, "PaDEL-Descriptor", "PaDEL-Descriptor.jar")
DESCRIPTOR_TEMPLATE = os.path.join(DATA_DIR, "descriptors_noredundancy.csv")

#  To support original coding
MODEL_PATH = LOGK_MODEL_PATH


