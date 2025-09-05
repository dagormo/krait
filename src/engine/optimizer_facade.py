from __future__ import annotations
from typing import Dict, List, Tuple
from types import SimpleNamespace
import numpy as np

from engine import (
    enforce_nondec_concentration,
    enforce_slope,
    collapse_repeats,
    build_gradient_profile,
    simulate_chromatogram,
    evaluate_resolution,
    OptConfig,
    anneal_live,
    fit_ln_k_ln_c
)


def append_return_and_equilibrate(bt, bc, start_c, eq_min, hard_cap_min):
    """Append a step back to the starting concentration and hold to equilibrate, capping total at hard_cap_min."""
    bt = list(map(float, bt))
    bc = list(map(float, bc))
    t_end = float(bt[-1])

    # instantaneous step back to start concentration (no "dip" segments)
    if abs(bc[-1] - start_c) > 1e-9:
        bt += [t_end, t_end + 0.1]   # small 0.1-min step
        bc += [bc[-1], start_c]
        t_end = bt[-1]

    # hold for equilibration
    hold_end = min(hard_cap_min, t_end + float(eq_min))
    if hold_end > t_end:
        bt.append(hold_end)
        bc.append(start_c)
    return np.array(bt), np.array(bc)


def run_optimize(
    analytes: List[dict],
    triad_concs: Tuple[float, float, float],
    bounds: dict,
    constraints: dict,
    weights: dict,
    objective: str,
    t0_min: float | None,
    plates: int,
    dt: float,
    seed: List[dict] | None,
):
    # --- 1) fit simple ln k = a − b ln C models from 3 iso points ---
    C1, C2, C3 = map(float, triad_concs)
    analyte_names = []
    models: Dict[str, object] = {}
    for a in analytes:
        name = str(a["name"])
        analyte_names.append(name)
        rts = [float(a["rt_at_c1_min"]), float(a["rt_at_c2_min"]), float(a["rt_at_c3_min"])]
        # fallback t0: small fraction of smallest supplied RT
        t0 = float(t0_min) if (t0_min and t0_min > 0) else max(0.01, 0.05 * min(rts))
        models[name] = fit_ln_k_ln_c([C1, C2, C3], rts, t0_min=t0, N=float(plates))

    # --- 2) seed gradient ---
    if seed and len(seed) >= 2:
        seed_t = np.array([float(n["t_min"]) for n in seed], dtype=float)
        seed_c = np.array([float(n["c_mM"]) for n in seed], dtype=float)
    else:
        # simple default: start at C1, climb to C3 within allowed bounds
        tmax = float(bounds["time_max_min"])
        seed_t = np.array([0.0, 0.5 * tmax, 0.8 * tmax], dtype=float)
        seed_c = np.array([C1, 0.5 * (C1 + C3), C3], dtype=float)

    # enforce constraints upfront
    if constraints.get("enforce_nondec", True):
        seed_t, seed_c = enforce_nondec_concentration(seed_t, seed_c)
    slope_limit = constraints.get("slope_limit", None)
    if slope_limit:
        seed_t, seed_c = enforce_slope(seed_t, seed_c, float(slope_limit))

    # --- 3) SA configuration (respect bounds; 60 min & 1..100 mM enforced earlier) ---
    eq_tail = float(constraints.get("equilibration_min", 5.0))
    # keep a little room for the 0.1-min return step + equilibration hold
    analysis_budget = max(1.0, float(bounds["time_max_min"]) - (eq_tail + 0.2))

    cfg = OptConfig(
        iterations=1500,
        init_temp=1.0,
        min_conc=float(bounds["min_conc_mM"]),
        max_conc=float(bounds["max_conc_mM"]),
        max_time=float(analysis_budget),
        enforce_nondec=bool(constraints.get("enforce_nondec", True)),
        target_Rs=float(constraints.get("target_Rs", 1.8)),
        objective_mode=("constraint_then_time" if objective == "constraint_then_time" else "weighted"),
        alpha_time=float(weights.get("alpha_time", 0.2)),
        step_penalty=float(weights.get("step_penalty", 0.02)),
        dt=float(dt),
        slope_limit=float(slope_limit) if slope_limit else None,
    )

    # --- 4) run annealer (single final packet by setting update_every=iterations) ---
    gen = anneal_live(models, seed_t, seed_c, cfg, update_every=cfg.iterations)
    packet = next(gen)                     # final snapshot
    best_t = packet["best_times"]
    best_c = packet["best_concs"]

    # quantize/collapse minor duplicates
    best_k = np.full(len(best_t), 5, dtype=int)
    bt, bc, _ = collapse_repeats(best_t, best_c, best_k)

    # --- 5) append return-to-start + equilibration; never exceed 60 min ---
    start_c = float(bc[0])
    bt, bc = append_return_and_equilibrate(bt, bc, start_c, eq_tail, float(bounds["time_max_min"]))

    # --- 6) preview simulation for diagnostics ---
    tg, Cg = build_gradient_profile(bt, bc, float(bounds["time_max_min"]), float(dt))
    _, tRs, sigmas = simulate_chromatogram(models, tg, Cg)
    min_Rs, _ = evaluate_resolution(tRs, sigmas)
    res = SimpleNamespace(names=analyte_names, tRs=tRs, sigmas=sigmas)

    return dict(
        times=bt, concs=bc, res=res, analyte_names=analyte_names,
        predicted_rt=tRs, min_Rs=float(min_Rs)
    )
