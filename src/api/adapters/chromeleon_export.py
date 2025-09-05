import numpy as np
import pandas as pd
from engine import round_to

def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def _extract_numeric_series(obj, names=None, key_candidates=("rt","tR","value","val")):
    if obj is None:
        return []
    if isinstance(obj, dict):
        if names is not None:
            return [_safe_float(obj.get(n)) for n in names]
        return [_safe_float(v) for v in obj.values()]
    if isinstance(obj, (list, tuple, np.ndarray, pd.Series)):
        vals = []
        for item in obj:
            if isinstance(item, (int, float, np.floating, np.integer)) or item is None:
                vals.append(_safe_float(item)); continue
            if isinstance(item, dict):
                v = None
                for k in key_candidates:
                    if k in item:
                        v = item[k]; break
                if v is None:
                    for k in ("sigma","std","width"):
                        if k in item:
                            v = item[k]; break
                vals.append(_safe_float(v)); continue
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                vals.append(_safe_float(item[1])); continue
            vals.append(_safe_float(item))
        return vals
    return [_safe_float(obj)]

def build_export_payload(pm_name, injection_name, column_name, bt, bc, res, names_fallback, include_window):
    # quantize/round for export
    times_q = round_to(bt, 0.1)
    concs_q = np.round(bc, 1)
    gradient_list = [{"t": float(t), "c": float(c)} for t, c in zip(times_q, concs_q)]

    # Names
    names = list(getattr(res, "names", names_fallback))

    # Extract retention times & sigmas robustly
    tRs = _extract_numeric_series(getattr(res, "tRs", None), names=names)
    sigmas = _extract_numeric_series(getattr(res, "sigmas", None), names=names) if include_window else None

    def _align(seq, n):
        seq = [] if seq is None else list(seq)
        if len(seq) < n:
            seq = seq + [None] * (n - len(seq))
        return seq[:n]

    tRs = _align(tRs, len(names))
    if include_window:
        sigmas = _align(sigmas, len(names))

    components = []
    for i, name in enumerate(names):
        rt_val = None if tRs[i] is None else round(float(tRs[i]), 3)
        if include_window and sigmas is not None and sigmas[i] is not None:
            window_val = round(4.0 * float(sigmas[i]), 3)  # ~±2σ
        else:
            window_val = None
        components.append({"name": str(name), "rt": rt_val, "window": window_val})

    payload = {
        "schema_version": "1.0",
        "context": {
            "processing_method_name": pm_name,
            "injection_name": injection_name,
            "column": column_name,
            "units": {"rt": "min", "window": "min", "conc": "mM", "time": "min"},
        },
        "gradient": gradient_list,
        "components": components,
    }
    return payload
