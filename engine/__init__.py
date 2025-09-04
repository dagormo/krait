# engine/__init__.py

from .descriptors import get_combined_descriptors, classify_ion_type, DESCRIPTOR_TEMPLATE
from .pca import apply_pca
from .simulation import calibrate_velocity, build_conc_function, predict_rt, plot_gradient
from .predictor import build_feature_vector, predict_with_preprocessed
from .config import BASE_DIR, MODEL_PATH, PCA_MODEL_PATH, DATA, PKL
from .preprocess import preprocess_analyte
from .core_models import RetentionModel, fit_ln_k_ln_c, k_from_C, sigma_t_from_tr
from .gradient_tools import (
    build_gradient_profile,
    seed_from_df,
    enforce_slope,
    enforce_nondec_concentration,
    collapse_repeats
)
from .simulate_tools import find_critical_pair
from .optimize_sa import OptConfig, anneal_live
from .optimize_nsga import nsga2_live

__all__ = [
    "get_combined_descriptors",
    "classify_ion_type",
    "apply_pca",
    "calibrate_velocity",
    "build_conc_function",
    "predict_rt",
    "plot_gradient",
    "predict_with_preprocessed",
    "preprocess_analyte",
    "build_feature_vector",
    "BASE_DIR",
    "DATA",
    "PCA_MODEL_PATH",
    "MODEL_PATH",
    "DESCRIPTOR_TEMPLATE",
    "PKL",
    "RetentionModel",
    "fit_ln_k_ln_c",
    "k_from_C",
    "sigma_t_from_tr",
    "build_gradient_profile",
    "seed_from_df",
    "enforce_slope",
    "enforce_nondec_concentration",
    "collapse_repeats",
    "find_critical_pair",
    "OptConfig",
    "anneal_live",
    "nsga2_live"
]
