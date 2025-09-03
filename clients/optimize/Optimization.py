
import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================
# Helpers
# =============================

def strength_transform(C, mode):
    C = np.asarray(C, dtype=float)
    if mode == "C (linear)":
        return C
    elif mode == "log10(C)":
        # guard against log of <=0
        return np.log10(np.maximum(C, 1e-9))
    elif mode == "ln(C)":
        return np.log(np.maximum(C, 1e-9))
    else:
        return C

def lss_fit_from_isocratic(concs, tR_list, t0, mode="log10(C)"):
    """Fit generalized LSS-style model per analyte:
       log10(k) = a - b * f(C), where f is chosen by 'mode'.
       Returns (a, b), and residuals for diagnostics.
    """
    k = np.maximum((np.array(tR_list, dtype=float) - t0) / max(t0, 1e-9), 1e-12)
    y = np.log10(k)
    X1 = strength_transform(np.array(concs, dtype=float), mode)
    X = np.vstack([np.ones_like(X1), -X1]).T  # [1, -f(C)]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coef[0], coef[1]
    y_hat = X @ coef
    resid = y - y_hat
    return float(a), float(b), resid

def k_from_conc(a, b, C, mode):
    return 10 ** (a - b * strength_transform(np.asarray(C, dtype=float), mode))

def simulate_retention_time(a, b, times, concs, t0, mode, dt=0.01, max_time=None):
    if max_time is None:
        max_time = float(times[-1]) + 2 * t0 + 30.0
    t = 0.0
    theta = 0.0
    while t <= max_time and theta < 1.0:
        C_t = float(np.interp(t, times, concs))
        k_t = k_from_conc(a, b, C_t, mode)
        theta += dt / (max(t0, 1e-9) * (1.0 + k_t))
        t += dt
    return t if theta >= 1.0 else np.nan

def compute_peak_sigma(tR, N):
    if np.isnan(tR) or N <= 0:
        return np.nan
    return tR / max(np.sqrt(N), 1e-9)

def resolution_adjacent(times_sorted, sigmas_sorted):
    out = []
    for i in range(len(times_sorted) - 1):
        t1, t2 = times_sorted[i], times_sorted[i + 1]
        s1, s2 = sigmas_sorted[i], sigmas_sorted[i + 1]
        if np.any(np.isnan([t1, t2, s1, s2])):
            out.append((i, np.nan))
            continue
        w1h = 2.355 * s1
        w2h = 2.355 * s2
        denom = (w1h + w2h)
        Rs = 1.18 * (t2 - t1) / max(denom, 1e-9)
        out.append((i, float(Rs)))
    return out

def _times_for_interp(times):
    eps = 1e-9
    t = times.astype(float).copy()
    for i in range(1, len(t)):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + eps
    return t

def build_gradient_fn(grad_df):
    g = grad_df.sort_values("time_min").reset_index(drop=True)
    times = g["time_min"].to_numpy(dtype=float)
    concs = g["conc_mM"].to_numpy(dtype=float)
    if len(times) == 0:
        times = np.array([0.0], dtype=float)
        concs = np.array([0.0], dtype=float)
    elif times[0] > 0:
        times = np.insert(times, 0, 0.0)
        concs = np.insert(concs, 0, concs[0])
    return _times_for_interp(times), concs

def truncate_gradient_to_peaks(grad_df, tRs):
    if tRs is None or len(tRs) == 0 or np.all(np.isnan(tRs)):
        return grad_df.copy()
    latest = float(np.nanmax(tRs))
    if not np.isfinite(latest):
        return grad_df.copy()
    t_end = int(math.ceil(latest))
    g = grad_df.sort_values("time_min").reset_index(drop=True)
    if t_end >= float(g["time_min"].max()):
        return g
    times = g["time_min"].to_numpy(dtype=float)
    concs = g["conc_mM"].to_numpy(dtype=float)
    c_end = float(np.interp(t_end, _times_for_interp(times), concs))
    g = g[g["time_min"] <= t_end].copy()
    if len(g) == 0 or g["time_min"].iloc[-1] < t_end:
        g.loc[len(g)] = {"time_min": t_end, "conc_mM": c_end}
    else:
        g.loc[g.index[-1], "conc_mM"] = c_end
    return g

# =============================
# Optimizer utilities
# =============================

def _round_program(g):
    g = g.copy()
    g["time_min"] = np.round(g["time_min"].to_numpy(dtype=float) * 10.0) / 10.0
    g["conc_mM"] = np.rint(g["conc_mM"].to_numpy(dtype=float))
    return g

def enforce_gradient_constraints(g, conc_bounds, t_bounds, min_step, monotonic, slope_max):
    g = g.sort_values("time_min").reset_index(drop=True).copy()
    if len(g) == 0:
        return pd.DataFrame({"time_min":[0.0], "conc_mM":[conc_bounds[0]]})
    g["conc_mM"] = np.clip(g["conc_mM"].to_numpy(dtype=float), conc_bounds[0], conc_bounds[1])
    g.loc[0, "time_min"] = 0.0
    for i in range(1, len(g)):
        t_prev = float(g.loc[i-1, "time_min"])
        t_i    = float(g.loc[i, "time_min"])
        if t_i < t_prev:
            g.loc[i, "time_min"] = t_prev
        elif t_i > t_prev and (t_i - t_prev) < min_step:
            g.loc[i, "time_min"] = t_prev + min_step
    g.loc[len(g)-1, "time_min"] = min(float(g.loc[len(g)-1, "time_min"]), float(t_bounds[1]))
    if monotonic:
        for i in range(1, len(g)):
            g.loc[i, "conc_mM"] = max(float(g.loc[i, "conc_mM"]), float(g.loc[i-1, "conc_mM"]))
    if slope_max is not None and slope_max > 0:
        for i in range(1, len(g)):
            dt = float(g.loc[i, "time_min"] - g.loc[i-1, "time_min"])
            if dt <= 0:
                continue
            max_allowed = float(g.loc[i-1, "conc_mM"]) + slope_max * dt
            g.loc[i, "conc_mM"] = min(float(g.loc[i, "conc_mM"]), max_allowed)
    g = _round_program(g)
    for i in range(1, len(g)):
        t_prev = float(g.loc[i-1, "time_min"])
        t_i    = float(g.loc[i, "time_min"])
        if t_i < t_prev:
            g.loc[i, "time_min"] = t_prev
        elif t_i > t_prev and (t_i - t_prev) < min_step:
            g.loc[i, "time_min"] = t_prev + min_step
    g = _round_program(g)
    return g

def evaluate_gradient(grad_df, models, analyte_names, t0, N, dt, mode, t_guard=30.0):
    times, concs = build_gradient_fn(grad_df)
    max_time = float(times[-1]) + t_guard
    tRs, sigmas = [], []
    for name in analyte_names:
        a, b, m = models[name]
        tR = simulate_retention_time(a, b, times, concs, t0, m, dt=dt, max_time=max_time)
        tRs.append(tR)
        sigmas.append(compute_peak_sigma(tR, N))
    tRs = np.array(tRs, dtype=float)
    sigmas = np.array(sigmas, dtype=float)
    order = np.argsort(tRs)
    t_sorted = tRs[order]
    s_sorted = sigmas[order]
    pair_Rs = resolution_adjacent(t_sorted, s_sorted)
    min_Rs = np.nanmin([r for _, r in pair_Rs]) if len(pair_Rs) else np.nan
    program_end = float(times[-1])
    latest_peak = float(np.nanmax(tRs)) if len(tRs) else program_end
    effective_time = float(int(math.ceil(latest_peak))) if np.isfinite(latest_peak) else program_end
    n_steps = max(0, len(grad_df) - 1)
    return tRs, sigmas, pair_Rs, min_Rs, program_end, effective_time, n_steps

def score_candidate(min_Rs, effective_time, T_ref, mode, alpha, rs_target, n_steps, step_beta):
    step_penalty = step_beta * n_steps
    if mode == "Constraint then time":
        margin = float(min_Rs) - float(rs_target)
        if not np.isfinite(margin):
            return -1e12
        if margin < 0:
            return margin - step_penalty
        return 10000.0 - (effective_time / max(T_ref, 1e-9)) - step_penalty + 1e-3 * margin
    else:
        return float(min_Rs) - alpha * (effective_time / max(T_ref, 1e-9)) - step_penalty

def mutate_gradient(grad_df, conc_bounds, t_bounds, min_step, time_sigma, conc_sigma, monotonic, slope_max, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    g = grad_df.copy()
    n = len(g)
    if n < 2:
        base = pd.DataFrame({"time_min":[0.0, min(5.0, t_bounds[1])], "conc_mM":[conc_bounds[0], conc_bounds[0]]})
        g = base
        n = 2
    op = rng.choice(["jitter", "jitter", "add", "remove", "tailcut"])
    if op == "add" and n < 30:
        i = int(rng.integers(0, n-1))
        t_mid = (g.loc[i, "time_min"] + g.loc[i+1, "time_min"]) / 2.0
        c_mid = (g.loc[i, "conc_mM"] + g.loc[i+1, "conc_mM"]) / 2.0
        g = pd.concat([g.iloc[:i+1], pd.DataFrame({"time_min":[t_mid], "conc_mM":[c_mid]}), g.iloc[i+1:]], ignore_index=True)
    elif op == "remove" and n > 2:
        i = int(rng.integers(1, n-1))
        g = pd.concat([g.iloc[:i], g.iloc[i+1:]], ignore_index=True)
    elif op == "tailcut" and n >= 2:
        shrink = abs(rng.normal(0.0, 1.0))
        g.loc[n-1, "time_min"] = max(g.loc[n-2, "time_min"], g.loc[n-1, "time_min"] - shrink)
        g.loc[n-1, "conc_mM"] = max(conc_bounds[0], g.loc[n-1, "conc_mM"] - abs(rng.normal(0.0, conc_sigma)))
    else:
        idx = int(rng.integers(0, n))
        g.loc[idx, "time_min"] = float(g.loc[idx, "time_min"]) + rng.normal(0, time_sigma)
        g.loc[idx, "conc_mM"] = float(g.loc[idx, "conc_mM"]) + rng.normal(0, conc_sigma)
    g = enforce_gradient_constraints(g, conc_bounds, t_bounds, min_step, monotonic, slope_max)
    return g

def add_post_peak_jump(g, tRs, conc_max, min_step):
    if tRs is None or len(tRs) < 2 or np.all(np.isnan(tRs)):
        return g
    finite = np.array(tRs, dtype=float)[np.isfinite(tRs)]
    if len(finite) < 2:
        return g
    t2 = np.sort(finite)[-2]
    t_jump = round((t2 + 0.1) * 10.0) / 10.0
    g2 = g.copy().sort_values("time_min").reset_index(drop=True)
    g2 = pd.concat([g2, pd.DataFrame({"time_min":[t_jump], "conc_mM":[g2["conc_mM"].max()]})], ignore_index=True)
    g2 = enforce_gradient_constraints(
        g2,
        conc_bounds=(g2["conc_mM"].min(), g2["conc_mM"].max()),
        t_bounds=(0.0, float(g2["time_min"].max()+10.0)),
        min_step=min_step,
        monotonic=False,
        slope_max=None
    )
    return g2

def optimize_gradient(
    init_grad_df, models, analyte_names, t0, N, dt, transform_mode,
    mode="Constraint then time", rs_target=1.8, alpha=0.2,
    step_beta=0.02, iterations=3000, temperature=0.5, conc_bounds=(1.0, 200.0),
    t_bounds=(0.0, 40.0), min_step=0.1, time_sigma=0.3, conc_sigma=1.0,
    monotonic=True, slope_max=None, post_peak_jump=True, rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    current = enforce_gradient_constraints(init_grad_df.copy(), conc_bounds, t_bounds, min_step, monotonic, slope_max)
    current = mutate_gradient(current, conc_bounds, t_bounds, min_step, time_sigma, conc_sigma, monotonic, slope_max, rng=rng)

    tRs, sigmas, pair_Rs, min_Rs, program_end, effective_time, n_steps = evaluate_gradient(current, models, analyte_names, t0, N, dt, transform_mode)
    T_ref = max(effective_time, 1e-9)
    current_score = score_candidate(min_Rs, effective_time, T_ref, mode, alpha, rs_target, n_steps, step_beta)

    best = current.copy()
    best_score = current_score
    best_tuple = (tRs, sigmas, pair_Rs, min_Rs, program_end, effective_time, n_steps)

    for it in range(int(iterations)):
        cand = mutate_gradient(current, conc_bounds, t_bounds, min_step, time_sigma, conc_sigma, monotonic, slope_max, rng=rng)
        tRs_c, sigmas_c, pair_Rs_c, min_Rs_c, program_end_c, effective_time_c, n_steps_c = evaluate_gradient(cand, models, analyte_names, t0, N, dt, transform_mode)

        if post_peak_jump and np.isfinite(min_Rs_c) and min_Rs_c >= rs_target:
            g_jump = add_post_peak_jump(cand, tRs_c, conc_bounds[1], min_step)
            tRs_j, sigmas_j, pair_Rs_j, min_Rs_j, program_end_j, effective_time_j, n_steps_j = evaluate_gradient(g_jump, models, analyte_names, t0, N, dt, transform_mode)
            T_ref_local = max(T_ref, 1e-9)
            score_c = score_candidate(min_Rs_c, effective_time_c, T_ref_local, mode, alpha, rs_target, n_steps_c, step_beta)
            score_j = score_candidate(min_Rs_j, effective_time_j, T_ref_local, mode, alpha, rs_target, n_steps_j, step_beta)
            if score_j > score_c:
                cand = g_jump
                tRs_c, sigmas_c, pair_Rs_c, min_Rs_c, program_end_c, effective_time_c, n_steps_c = (tRs_j, sigmas_j, pair_Rs_j, min_Rs_j, program_end_j, effective_time_j, n_steps_j)
                cand_score = score_j
            else:
                cand_score = score_c
        else:
            cand_score = score_candidate(min_Rs_c, effective_time_c, T_ref, mode, alpha, rs_target, n_steps_c, step_beta)

        if not np.isfinite(cand_score):
            accept = False
        else:
            if cand_score >= current_score:
                accept = True
            else:
                accept_prob = np.exp((cand_score - current_score) / max(temperature, 1e-9))
                accept = (np.random.rand() < accept_prob)

        if accept:
            current = cand
            current_score = cand_score
            if cand_score > best_score:
                best = cand
                best_score = cand_score
                best_tuple = (tRs_c, sigmas_c, pair_Rs_c, min_Rs_c, program_end_c, effective_time_c, n_steps_c)

        if (it + 1) % 100 == 0:
            temperature *= 0.95

    return best, best_tuple, best_score

# =============================
# Default analytes (your list)
# =============================
DEFAULT_ANALYTES = pd.DataFrame({
    "Analyte": [
        "Acetate","Bromide","Carbonate","Chloride","Fluoride","Formate","Glycolate",
        "Iodide","Nitrate","Nitrite","Oxalate","Phosphate","Propionate",
        "Sulfate","Sulfite","Thiocyanate","Thiosulfate"
    ],
    "tR_C1": [2.19, 6.249, 4.952, 3.395, 2.12, 2.341, 2.177, 28.057, 6.609, 3.93, 7.541, 17.809, 2.276, 6.296, 5.795, 55.989, 21.469],
    "tR_C2": [2.07, 4.83, 3.205, 2.883, 2.026, 2.172, 2.062, 19.69, 5.059, 3.254, 4.316, 7.067, 2.134, 3.756, 3.558, 38.687, 10.28],
    "tR_C3": [1.971, 3.699, 2.335, 2.481, 1.943, 2.037, 1.969, 13.023, 3.82, 2.707, 2.742, 3.128, 2.017, 2.521, 2.452, 24.947, 4.876],
})

DEFAULT_GRADIENT = pd.DataFrame({
    "time_min": [0.0, 4.0, 6.0, 6.0, 10.0],
    "conc_mM":  [5.0, 10.0, 15.0, 50.0, 50.0],
})

# =============================
# UI
# =============================
st.set_page_config(page_title="IC Gradient Optimizer — Rs then Time", layout="wide")
st.title("🔬 IC Multi-Step Gradient Optimizer — Rs First, Then Time")

# Sidebar form
with st.sidebar:
    with st.form("settings_form", clear_on_submit=False):
        N = st.number_input("Theoretical plates (N)", 1000, 2_000_000, 50_000, 1000, key="N")
        c1 = st.number_input("Conc 1 (mM)", 0.1, 500.0, 5.0, 0.1, key="c1")
        c2 = st.number_input("Conc 2 (mM)", 0.1, 500.0, 10.0, 0.1, key="c2")
        c3 = st.number_input("Conc 3 (mM)", 0.1, 500.0, 30.0, 0.1, key="c3")
        auto_t0 = st.checkbox("Estimate t0 automatically", value=True, key="auto_t0")
        t0_user = st.number_input("Hold-up time t0 (min)", 0.01, 20.0, 1.0, 0.01, disabled=auto_t0, key="t0_user")
        dt = st.number_input("Integration step dt (min)", 0.001, 0.1, 0.01, 0.001, format="%.3f", key="dt")

        transform_mode = st.selectbox("Eluent strength transform f(C)", ["log10(C)", "C (linear)", "ln(C)"], index=0, key="transform_mode")

        mode = st.radio("Objective", ["Constraint then time", "Weighted tradeoff"], index=0, key="mode")
        rs_target = st.slider("Target minimum resolution (Rs)", 0.5, 3.0, 1.8, 0.1, key="rs_target")
        penalty_alpha = st.slider("Run time penalty (α) — for Weighted mode", 0.0, 1.0, 0.20, 0.01, key="penalty_alpha")
        step_beta = st.number_input("Step penalty β (per step)", 0.0, 1.0, 0.02, 0.01, key="step_beta")
        iterations = st.number_input("Iterations", 100, 20000, 3000, 100, key="iterations")
        temperature = st.number_input("Initial temperature", 0.01, 5.0, 0.5, 0.01, key="temperature")
        conc_min = st.number_input("Min concentration (mM)", 0.0, 500.0, 5.0, 1.0, key="conc_min")
        conc_max = st.number_input("Max concentration (mM)", 0.0, 500.0, 100.0, 1.0, key="conc_max")
        t_max = st.number_input("Max end time (min)", 1.0, 240.0, 40.0, 0.1, key="t_max")

        monotonic = st.checkbox("Enforce non-decreasing concentration", value=True, key="monotonic")
        slope_cap_on = st.checkbox("Limit ramp slope (mM/min)", value=False, key="slope_cap_on")
        slope_max = st.number_input("Max slope (mM/min)", 0.1, 200.0, 10.0, 0.1, key="slope_max", disabled=not slope_cap_on)

        post_peak_jump = st.checkbox("Jump to max conc after penultimate peak (heuristic)", value=True, key="post_peak_jump")

        col_s1, col_s2, col_s3 = st.columns([1,1,1])
        apply_settings = col_s1.form_submit_button("Apply")
        apply_and_sim = col_s2.form_submit_button("Apply & Simulate")
        apply_and_opt = col_s3.form_submit_button("Apply & Optimize")

# Tables
st.markdown("### 1) Analytes and isocratic retention times")
if "df_analytes" not in st.session_state:
    st.session_state.df_analytes = DEFAULT_ANALYTES.copy()
if "df_gradient" not in st.session_state:
    st.session_state.df_gradient = DEFAULT_GRADIENT.copy()

with st.form("tables_form", clear_on_submit=False):
    anal_df = st.data_editor(
        st.session_state.df_analytes, num_rows="dynamic", use_container_width=True,
        key="analyte_editor",
        column_config={
            "Analyte": st.column_config.TextColumn("Analyte"),
            "tR_C1": st.column_config.NumberColumn(f"tR@{st.session_state.get('c1', c1)} mM", step=0.001, format="%.3f"),
            "tR_C2": st.column_config.NumberColumn(f"tR@{st.session_state.get('c2', c2)} mM", step=0.001, format="%.3f"),
            "tR_C3": st.column_config.NumberColumn(f"tR@{st.session_state.get('c3', c3)} mM", step=0.001, format="%.3f"),
        },
    )
    st.markdown("### 2) Multi-step gradient (equal timestamps allowed for steps)")
    grad_df = st.data_editor(
        st.session_state.df_gradient, num_rows="dynamic", use_container_width=True, key="gradient_editor",
        column_config={
            "time_min": st.column_config.NumberColumn("time_min (0.1 min)", step=0.1, format="%.1f"),
            "conc_mM": st.column_config.NumberColumn("conc_mM (1 mM)", step=1, format="%.0f"),
        },
    )

    col_t1, col_t2, col_t3 = st.columns([1,1,1])
    save_tables = col_t1.form_submit_button("💾 Save")
    save_and_sim = col_t2.form_submit_button("Save & Simulate")
    save_and_opt = col_t3.form_submit_button("Save & Optimize")

# Action logic
simulate_requested = False
optimize_requested = False

if save_tables or save_and_sim or save_and_opt:
    grad_df_save = enforce_gradient_constraints(
        grad_df.copy(),
        conc_bounds=(float(st.session_state.get("conc_min", 5.0)), float(st.session_state.get("conc_max", 100.0))),
        t_bounds=(0.0, float(st.session_state.get("t_max", 40.0))),
        min_step=0.1,
        monotonic=bool(st.session_state.get("monotonic", True)),
        slope_max=float(st.session_state.get("slope_max", 10.0)) if st.session_state.get("slope_cap_on", False) else None,
    )
    st.session_state.df_analytes = anal_df.copy()
    st.session_state.df_gradient = grad_df_save
    st.success("Tables saved.")
    simulate_requested = save_and_sim
    optimize_requested = save_and_opt

if apply_settings or apply_and_sim or apply_and_opt:
    simulate_requested = simulate_requested or apply_and_sim
    optimize_requested = optimize_requested or apply_and_opt

# Build models from current tables
anal_df_clean = st.session_state.df_analytes.dropna(subset=["Analyte"]).copy()
anal_df_clean = anal_df_clean[anal_df_clean["Analyte"].astype(str).str.strip() != ""]
if st.session_state.get("auto_t0", True):
    concs = np.array([st.session_state.get("c1", 5.0), st.session_state.get("c2", 10.0), st.session_state.get("c3", 30.0)], dtype=float)
    max_idx = int(np.argmax(concs))
    tR_cols = ["tR_C1", "tR_C2", "tR_C3"]
    col_fast = tR_cols[max_idx]
    min_fast = float(anal_df_clean[col_fast].min()) if len(anal_df_clean) else 1.0
    t0 = max(0.1, 0.5 * min_fast)
else:
    t0 = float(st.session_state.get("t0_user", 1.0))

transform_mode = st.session_state.get("transform_mode", "log10(C)")

models = {}
analyte_names = []
fit_rows = []
isoc_concs = np.array([st.session_state.get("c1", 5.0), st.session_state.get("c2", 10.0), st.session_state.get("c3", 30.0)], dtype=float)
for _, row in anal_df_clean.iterrows():
    name = str(row["Analyte"]).strip()
    if not name:
        continue
    tR_list = [float(row["tR_C1"]), float(row["tR_C2"]), float(row["tR_C3"])]
    a, b, resid = lss_fit_from_isocratic(isoc_concs, tR_list, t0, mode=transform_mode)
    models[name] = (a, b, transform_mode)
    analyte_names.append(name)
    # diagnostics (RMS residual in log10k)
    rms = float(np.sqrt(np.mean(resid**2)))
    fit_rows.append({"Analyte": name, "b_slope": float(b), "RMS_resid_log10k": rms})

fit_diag = pd.DataFrame(fit_rows).sort_values("b_slope")

# Evaluate/Optimize
if "best_grad" not in st.session_state:
    st.session_state.best_grad = None
    st.session_state.best_tuple = None
if "current_eval" not in st.session_state:
    st.session_state.current_eval = None

grad_df_work = st.session_state.df_gradient.copy().sort_values("time_min").reset_index(drop=True)

def run_simulation(gdf):
    return evaluate_gradient(
        gdf, models, analyte_names, t0, st.session_state.get("N", 50000), st.session_state.get("dt", 0.01), transform_mode
    )

if simulate_requested and len(analyte_names) > 0:
    st.session_state.current_eval = run_simulation(grad_df_work)

if optimize_requested and len(analyte_names) > 0:
    best_grad, best_tuple, best_score = optimize_gradient(
        grad_df_work, models, analyte_names, t0,
        st.session_state.get("N", 50000), st.session_state.get("dt", 0.01), transform_mode,
        mode=st.session_state.get("mode", "Constraint then time"),
        rs_target=float(st.session_state.get("rs_target", 1.8)),
        alpha=float(st.session_state.get("penalty_alpha", 0.2)),
        step_beta=float(st.session_state.get("step_beta", 0.02)),
        iterations=int(st.session_state.get("iterations", 3000)),
        temperature=float(st.session_state.get("temperature", 0.5)),
        conc_bounds=(float(st.session_state.get("conc_min", 5.0)), float(st.session_state.get("conc_max", 100.0))),
        t_bounds=(0.0, float(st.session_state.get("t_max", 40.0))),
        min_step=0.1, time_sigma=0.3, conc_sigma=1.0,
        monotonic=bool(st.session_state.get("monotonic", True)),
        slope_max=float(st.session_state.get("slope_max", 10.0)) if st.session_state.get("slope_cap_on", False) else None,
        post_peak_jump=True,
    )
    st.session_state.best_grad = truncate_gradient_to_peaks(best_grad, best_tuple[0]).copy()
    st.session_state.best_tuple = best_tuple
    st.success("optimize complete. See the plots and tables below.")

# Display
use_best = st.toggle("Display optimized result (if available)", value=(st.session_state.best_grad is not None))

if use_best and st.session_state.best_grad is not None and st.session_state.best_tuple is not None:
    disp_grad = st.session_state.best_grad.copy()
    tRs, sigmas, pair_Rs, min_Rs, program_end, effective_time, n_steps = st.session_state.best_tuple
    title_tag = "Optimized"
elif st.session_state.current_eval is not None:
    disp_grad = grad_df_work.copy()
    tRs, sigmas, pair_Rs, min_Rs, program_end, effective_time, n_steps = st.session_state.current_eval
    title_tag = "Current"
else:
    disp_grad = grad_df_work.copy()
    tRs, sigmas, pair_Rs, min_Rs, program_end, effective_time, n_steps = run_simulation(disp_grad)
    title_tag = "Current"

disp_grad_trunc = truncate_gradient_to_peaks(disp_grad, tRs)

col_plot, col_tables = st.columns([1.2, 1.0], gap="large")

with col_plot:
    st.subheader(f"Simulated Chromatogram — {title_tag} Gradient")
    times_show, concs_show = build_gradient_fn(disp_grad_trunc)
    t_max_plot = max(times_show[-1], np.nanmax(tRs) if len(tRs) else 0) + 2.0
    grid = np.linspace(0, t_max_plot, 4000)

    y = np.zeros_like(grid)
    for i, name in enumerate(analyte_names):
        tR = tRs[i]
        s = sigmas[i]
        if np.isnan(tR) or np.isnan(s):
            continue
        y += np.exp(-0.5 * ((grid - tR) / max(s, 1e-9)) ** 2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid, y, label="Simulated chromatogram")

    for i, name in enumerate(analyte_names):
        tR = tRs[i]
        if np.isnan(tR):
            continue
        ax.axvline(tR, linestyle=":", alpha=0.5)
        ax.text(tR, ax.get_ylim()[1] * 0.95, name, rotation=90, va="top", ha="center", fontsize=8)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Signal (a.u.)")
    ax.set_title("Simulated Chromatogram with Gradient Overlay")

    ax2 = ax.twinx()
    ax2.plot(times_show, concs_show, alpha=0.9, linewidth=1.5, label="Eluent conc (mM)")
    ax2.set_ylabel("Eluent concentration (mM)")

    lines_labels = [ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    if labels:
        ax.legend(lines, labels, loc="upper left")

    st.pyplot(fig)
    plt.close(fig)

with col_tables:
    st.subheader("Gradient program (timestamped, truncated)")
    disp_grad_trunc = disp_grad_trunc.sort_values("time_min").reset_index(drop=True)
    disp_grad_trunc["time_min"] = np.round(disp_grad_trunc["time_min"] * 10.0) / 10.0
    disp_grad_trunc["conc_mM"] = np.rint(disp_grad_trunc["conc_mM"])
    st.dataframe(disp_grad_trunc, use_container_width=True)
    csv_grad = disp_grad_trunc.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download gradient CSV", csv_grad, file_name="gradient_program.csv", mime="text/csv")

    st.subheader("Predicted retention metrics")
    out_df = pd.DataFrame({
        "Analyte": analyte_names,
        "tR_min": tRs,
        "sigma_t_min": sigmas,
    }).sort_values("tR_min")
    st.dataframe(out_df, use_container_width=True)

    st.subheader("Model fit diagnostics")
    st.caption("Slope b and RMS residual of the linear fit in log10(k) vs chosen f(C). Lower RMS is better.")
    st.dataframe(fit_diag, use_container_width=True)

    if len(analyte_names) >= 2:
        order = np.argsort(out_df["tR_min"].to_numpy())
        t_sorted = out_df["tR_min"].to_numpy()[order]
        s_sorted = out_df["sigma_t_min"].to_numpy()[order]
        pair_Rs = resolution_adjacent(t_sorted, s_sorted)
        rs_vals = [r for _, r in pair_Rs]
        feas = "✅ meets" if float(np.nanmin(rs_vals)) >= float(st.session_state.get("rs_target", 1.8)) else "⚠️ below"
        st.caption(f"Min Rs = {np.nanmin(rs_vals):.2f} ({feas} target {st.session_state.get('rs_target', 1.8):.2f}) | Effective run time = {int(math.ceil(np.nanmax(tRs)) if len(tRs) else 0)} min | Steps = {n_steps}")

st.markdown(
    """
---
**Modeling note**: For anion-exchange (e.g., hydroxide gradients), **log10(k) often varies ~linearly with log\_10([eluent])**.
If your coelutions looked suspicious (e.g., bromide/sulfite/nitrate/sulfate), switch **Eluent strength transform** to **log10(C)** (default) or **ln(C)** and re-optimize.

**Objective**: Meet the target min **Rs**, then minimize **effective runtime** with a slight **per-step penalty**.  
**Steps**: Equal timestamps encode true steps; rounding is **0.1 min** and **1 mM** throughout.
    """
)
