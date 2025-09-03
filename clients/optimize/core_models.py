from dataclasses import dataclass
import numpy as np
import math

@dataclass
class RetentionModel:
    """Simple retention model: ln k = a − b ln(C)."""
    a: float
    b: float
    t0: float  # hold-up time (min)
    N: float   # theoretical plates

def fit_ln_k_ln_c(iso_concs_mM, iso_tr_min, t0_min, N):
    """Fit ln k = a − b ln C from 3 isocratic points."""
    C = np.asarray(iso_concs_mM, dtype=float)
    tR = np.asarray(iso_tr_min, dtype=float)
    k = np.maximum((tR - t0_min) / max(t0_min, 1e-6), 1e-6)
    lnk = np.log(k)
    lnc = np.log(np.maximum(C, 1e-9))
    X = np.vstack([np.ones_like(lnc), -lnc]).T
    coef, *_ = np.linalg.lstsq(X, lnk, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return RetentionModel(a=a, b=b, t0=float(t0_min), N=float(N))

def k_from_C(model: RetentionModel, C_mM):
    C = np.asarray(C_mM, dtype=float)
    lnk = model.a - model.b * np.log(np.maximum(C, 1e-9))
    k = np.exp(lnk)
    return np.clip(k, 1e-6, 1e6)

def sigma_t_from_tr(model: RetentionModel, tR: float) -> float:
    N = max(float(model.N), 1.0)
    return max(1e-3, float(tR) / math.sqrt(N))
