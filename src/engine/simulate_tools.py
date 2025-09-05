import numpy as np
import pandas as pd
from .core_models import RetentionModel, k_from_C, sigma_t_from_tr


def predict_tr_gradient(model: RetentionModel, time_grid, conc_grid, t_max):
    k = k_from_C(model, conc_grid)
    invk = 1.0 / np.maximum(k, 1e-9)
    dt = np.diff(time_grid, prepend=time_grid[0])
    g = np.cumsum(invk * dt)
    idx = np.searchsorted(g, model.t0, side="left")
    if idx >= len(time_grid):
        return float(t_max)
    return float(time_grid[idx])


def simulate_chromatogram(models, time_grid, conc_grid):
    y = np.zeros_like(time_grid)
    tRs, sigmas = {}, {}
    t_max = float(time_grid[-1])
    for name, mdl in models.items():
        tR = predict_tr_gradient(mdl, time_grid, conc_grid, t_max)
        tRs[name] = tR
        s = sigma_t_from_tr(mdl, tR)
        sigmas[name] = s
        y += np.exp(-0.5 * ((time_grid - tR) / max(s,1e-9)) ** 2)
    if y.max() > 0:
        y = y / y.max()
    return y, tRs, sigmas


def pairwise_resolution(tR1, s1, tR2, s2):
    w1 = 4.0 * s1
    w2 = 4.0 * s2
    return 1.18 * abs(tR2 - tR1) / (w1 + w2 + 1e-9)


def evaluate_resolution(tRs: dict, sigmas: dict):
    items = sorted(tRs.items(), key=lambda kv: kv[1])
    rows = []
    min_rs = float("inf")
    for i in range(len(items) - 1):
        n1, t1 = items[i]
        n2, t2 = items[i + 1]
        rs = pairwise_resolution(t1, sigmas[n1], t2, sigmas[n2])
        rows.append({"pair": f"{n1} | {n2}", "Rs": rs})
        min_rs = min(min_rs, rs)
    return (min_rs if rows else float("inf")), pd.DataFrame(rows)


def find_critical_pair(tRs: dict, sigmas: dict):
    items = sorted(tRs.items(), key=lambda kv: kv[1])
    worst_pair, worst_rs = None, float("inf")
    for i in range(len(items) - 1):
        n1, t1 = items[i]
        n2, t2 = items[i+1]
        rs = pairwise_resolution(t1, sigmas[n1], t2, sigmas[n2])
        if rs < worst_rs:
            worst_rs = rs
            worst_pair = (n1, n2)
    return worst_pair, worst_rs
