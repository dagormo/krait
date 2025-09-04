import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from export_utils import build_export_payload
from engine.core_models import fit_ln_k_ln_c
from engine.gradient_tools import (
    build_gradient_profile,
    seed_from_df,
    enforce_slope,
    enforce_nondec_concentration,
    collapse_repeats
)
from engine.simulate_tools import find_critical_pair
from engine.optimize_sa import OptConfig, anneal_live
from engine.config import GAConfig
from ui_defaults import default_analytes_table, default_gradient_table
from engine.optimize_nsga import nsga2_live
from plot_tools import plot_chromatogram_and_gradient


st.set_page_config(page_title="Krait – IC Gradient Optimizer (modular)", layout="wide")
st.title("🔬 Krait – IC Multi-Step Gradient Optimizer (modular)")


# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Settings")

    # Global
    t0_auto = st.checkbox("Estimate t0 automatically", value=True)
    t0_user = st.number_input(
        "Hold-up time t0 (min)", min_value=0.01, max_value=20.0, value=1.87, step=0.01, disabled=t0_auto
    )
    dt = st.number_input("Integration step dt (min)", 0.001, 0.1, 0.01, 0.001)
    plates = st.number_input("Theoretical plates (N)", 1000, 2_000_000, 50_000, 1000)

    st.markdown("---")
    st.subheader("Isocratic scouting")
    c1 = st.number_input("Conc 1 (mM)", 0.1, 500.0, 8.0, 0.1)
    c2 = st.number_input("Conc 2 (mM)", 0.1, 500.0, 20.0, 0.1)
    c3 = st.number_input("Conc 3 (mM)", 0.1, 500.0, 60.0, 0.1)

    st.markdown("---")
    st.subheader("Bounds & objective")
    minC = st.number_input("Min conc (mM)", 0.0, 500.0, 1.0, 1.0)
    maxC = st.number_input("Max conc (mM)", 0.0, 500.0, 100.0, 1.0)
    max_end_time = st.number_input("Max end time (min)", 5.0, 240.0, 30.0, 1.0)

    mode = st.radio("Objective", ["Constraint then time", "Weighted tradeoff"], index=0)
    target_Rs = st.slider("Target min Rs", 0.5, 3.0, 1.8, 0.05)
    alpha = st.slider("Run time penalty α (weighted mode)", 0.0, 1.0, 0.2, 0.01)
    step_beta = st.slider("Step penalty β", 0.0, 0.2, 0.02, 0.01)
    enforce_nondec = st.checkbox("Enforce non-decreasing concentration", True)
    slope_limit_on = st.checkbox("Limit ramp slope", False)
    slope_limit = st.number_input(
        "Max slope (mM/min)", 0.5, 200.0, 10.0, 0.5, disabled=not slope_limit_on
    )

    st.markdown("---")
    st.subheader("Optimizer")
    iterations = st.number_input("Iterations", 100, 50_000, 8_000, 100)
    init_temp = st.number_input("Initial temperature", 0.01, 5.0, 0.6, 0.01)
    update_every = st.number_input("Update every N iters", 10, 2_000, 100, 10)

    st.markdown("---")
    st.subheader("Export context (for JSON)")
    pm_name = st.text_input("Processing method name", "")
    injection_name = st.text_input("Injection name", "")
    column_name = st.text_input("Column", "")
    include_window = st.checkbox("Include peak window (±2σ)", False)

# ---------------------------
# Editable tables
# ---------------------------
st.subheader("1) Analytes & isocratic retention times")
if "analyte_df" not in st.session_state:
    st.session_state["analyte_df"] = default_analytes_table()
analyte_df = st.data_editor(
    st.session_state["analyte_df"], num_rows="dynamic", use_container_width=True, key="analyte_editor"
)
st.session_state["analyte_df"] = analyte_df.copy()

st.subheader("2) Multi-step gradient (repeated times ⇒ instantaneous steps)")
if "grad_df" not in st.session_state:
    st.session_state["grad_df"] = default_gradient_table()
grad_df = st.data_editor(
    st.session_state["grad_df"], num_rows="dynamic", use_container_width=True, key="gradient_editor"
)
st.session_state["grad_df"] = grad_df.copy()


# ---------------------------
# Build retention models
# ---------------------------
c1c2c3 = [c1, c2, c3]
if t0_auto:
    flat_min = float(np.nanmin(analyte_df[["RT@Conc1 (min)", "RT@Conc2 (min)", "RT@Conc3 (min)"]].values))
    t0 = max(0.2, round(0.2 * flat_min, 2))
else:
    t0 = float(t0_user)

models = {}
for _, row in analyte_df.iterrows():
    name = str(row["Analyte"]).strip()
    rts = [
        float(row["RT@Conc1 (min)"]),
        float(row["RT@Conc2 (min)"]),
        float(row["RT@Conc3 (min)"]),
    ]
    models[name] = fit_ln_k_ln_c(c1c2c3, rts, t0_min=t0, N=float(plates))

# Preserve analyte order for export fallback
analyte_order = list(models.keys())

# ---------------------------
# Buttons
# ---------------------------
simulate_btn = st.button("Simulate", type="primary")
run_opt_btn = st.button("Run live optimization")
run_nsga_live_btn = st.button("Run NSGA-II (live + SA polish)")


# ---------------------------
# SIMULATE
# ---------------------------
if simulate_btn:
    times, concs = seed_from_df(st.session_state["grad_df"])
    curves = list(st.session_state["grad_df"]["curve"]) if "curve" in st.session_state["grad_df"].columns else None
    if enforce_nondec:
        times, concs = enforce_nondec_concentration(times, concs)

    # NEW: quantize & collapse before plotting
    times, concs, curves = collapse_repeats(times, concs, curves)

    fig, res = plot_chromatogram_and_gradient(
        times, concs, max_end_time, dt, models, curves=curves,
        title="Simulated Chromatogram — Current Gradient"
    )
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------
# LIVE OPTIMIZATION
# ---------------------------
if run_opt_btn:
    # Seed
    times, concs = seed_from_df(st.session_state["grad_df"])
    curves = list(st.session_state["grad_df"]["curve"]) if "curve" in st.session_state["grad_df"].columns else None
    if enforce_nondec:
        times, concs = enforce_nondec_concentration(times, concs)
    if slope_limit_on:
        times, concs = enforce_slope(times, concs, float(slope_limit))

    times, concs, curves = collapse_repeats(times, concs, curves)

    # Config
    cfg = OptConfig(
        iterations=int(iterations),
        init_temp=float(init_temp),
        min_conc=float(minC),
        max_conc=float(maxC),
        max_time=float(max_end_time),
        enforce_nondec=bool(enforce_nondec),
        target_Rs=float(target_Rs),
        objective_mode="constraint_then_time" if mode.startswith("Constraint") else "weighted",
        alpha_time=float(alpha),
        step_penalty=float(step_beta),
        dt=float(dt),
        slope_limit=float(slope_limit) if slope_limit_on else None,
    )

    # Placeholders
    ph_plot = st.empty()
    tr_col1, tr_col2, tr_col3 = st.columns(3)
    ph_tr1, ph_tr2, ph_tr3 = tr_col1.empty(), tr_col2.empty(), tr_col3.empty()

    # Run
    gen = anneal_live(models, times, concs, cfg, update_every=int(update_every))
    best = None

    for packet in gen:
        cur_t, cur_c = packet["current_times"], packet["current_concs"]
        best_t, best_c = packet["best_times"], packet["best_concs"]
        best = (best_t, best_c)  # remember the latest best
        tr = packet["traces"]

        # --- chromatogram for CURRENT candidate ---
        fig, res = plot_chromatogram_and_gradient(cur_t, cur_c, max_end_time, dt, models, curves=curves, title="Best-so-far Chromatogram — Live")

        # OPTIONAL overlay of best-so-far gradient (faint line on right axis)
        tg_b, Cg_b = build_gradient_profile(best_t, best_c, max_end_time, dt, curves=curves)
        if len(fig.axes) < 2:
            fig.axes[0].twinx()
        ax2 = fig.axes[1]
        ax2.plot(tg_b, Cg_b, alpha=0.35, linewidth=1.0)

        ph_plot.pyplot(fig)
        plt.close(fig)

        # --- traces (each replaces the previous) ---
        # 1) Score trace
        fig1, ax1 = plt.subplots(figsize=(8, 2.8))
        ax1.plot(tr["iter"], tr["best"], label="best score")
        ax1.plot(tr["iter"], tr["current"], alpha=0.6, label="current score")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Score (lower is better)")
        ax1.legend()
        ph_tr1.pyplot(fig1)
        plt.close(fig1)

        # 2) Min-Rs trace
        fig2, ax2p = plt.subplots(figsize=(8, 2.8))
        ax2p.plot(tr["iter"], tr["minrs"], label="candidate min-Rs")
        ax2p.axhline(float(target_Rs), linestyle="--", label="target Rs")
        ax2p.set_xlabel("Iteration")
        ax2p.set_ylabel("Min Rs")
        ax2p.legend()
        ph_tr2.pyplot(fig2)
        plt.close(fig2)

        # 3) Runtime & steps trace
        fig3, ax3 = plt.subplots(figsize=(8, 2.8))
        ax3.plot(tr["iter"], tr["runtime"], label="candidate runtime (min)")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Runtime (min)")
        ax3b = ax3.twinx()

        steps_y = np.asarray(tr["steps"], dtype=float)
        finite_steps = steps_y[np.isfinite(steps_y)]
        y_max = 10 if finite_steps.size == 0 else max(8, float(finite_steps.max()) + 2)

        ax3b.plot(tr["iter"], steps_y, color="tab:orange", label="steps")
        ax3b.set_ylabel("Steps")
        ax3b.set_ylim(0, y_max)

        ph_tr3.pyplot(fig3)
        plt.close(fig3)

    # Final best snapshot
    if best is not None:
        st.success("optimize finished. Best result below.")
        best_times, best_concs = best
        best_curves = np.full(len(best_times), 5, dtype=int)  # SA didn't track curves

        # NEW: quantize & collapse before plotting/export
        bt, bc, bk = collapse_repeats(best_times, best_concs, best_curves)

        fig, res = plot_chromatogram_and_gradient(
            bt, bc, max_end_time, dt, models, curves=bk,
            title="Simulated Chromatogram — Optimized"
        )
        st.pyplot(fig);
        plt.close(fig)

        worst_pair, worst_rs = find_critical_pair(res.tRs, res.sigmas)
        if worst_pair:
            st.info(f"Critical pair: {worst_pair[0]} vs {worst_pair[1]} — Rs = {worst_rs:.2f}")

        # Export JSON
        payload = build_export_payload(pm_name, injection_name, column_name, bt, bc, res, analyte_order, include_window)
        st.subheader("Export — JSON")
        st.code(json.dumps(payload, indent=2), language="json")
        st.download_button(
            "⬇️ Download JSON",
            json.dumps(payload, indent=2).encode("utf-8"),
            file_name="optimized_gradient.json",
            mime="application/json",
        )

# ---------------------------
# NSGA-II + SA polish
# ---------------------------
if run_nsga_live_btn:
    times, concs = seed_from_df(st.session_state["grad_df"])
    curves = list(st.session_state["grad_df"]["curve"]) if "curve" in st.session_state["grad_df"].columns else None
    if enforce_nondec:
        times, concs = enforce_nondec_concentration(times, concs)
    if slope_limit_on:
        times, concs = enforce_slope(times, concs, float(slope_limit))

    times, concs, curves = collapse_repeats(times, concs, curves)

    ga_cfg = GAConfig(
        pop_size=64,
        generations=GAConfig.generations,
        max_points=GAConfig.max_points, min_points=GAConfig.min_points,
        min_conc=float(minC), max_conc=float(maxC), max_time=float(max_end_time),
        dt=float(dt), enforce_nondec=bool(enforce_nondec),
        slope_limit=float(slope_limit) if slope_limit_on else None
    )

    ph_scatter = st.empty()
    ph_trace = st.empty()
    final_packet = None

    # --- NSGA loop (Pareto + traces only) ---
    for packet in nsga2_live(models, times, concs, ga_cfg):
        final_packet = packet
        traces = packet["traces"]

        # Pareto scatter
        pareto = [{"runtime": info["end_time"], "min_Rs": info["min_Rs"], "steps": info["steps"]}
                  for info in packet["front_infos"]]
        pdf = pd.DataFrame(pareto)
        fig_sc, ax_sc = plt.subplots()
        if not pdf.empty:
            ax_sc.scatter(pdf["runtime"], pdf["min_Rs"], s=40+10*pdf["steps"], alpha=0.7)
            ax_sc.axhline(float(target_Rs), linestyle="--", color="gray")
        ax_sc.set_xlabel("Runtime (min)"); ax_sc.set_ylabel("Min Rs")
        ax_sc.set_title(f"Pareto gen {packet['gen']+1}")
        ph_scatter.pyplot(fig_sc); plt.close(fig_sc)

        # trace plot
        fig_tr, ax_tr = plt.subplots()
        ax_tr.plot(traces["gen"], traces["best_Rs"], label="best Rs")
        ax_tr.plot(traces["gen"], traces["median_Rs"], label="median Rs")
        ax_tr.set_ylabel("Resolution"); ax_tr.set_xlabel("Generation")
        ax_tr.legend()
        ph_trace.pyplot(fig_tr); plt.close(fig_tr)

    # --- after GA ends, SA polish once ---
    if final_packet:
        front_infos = final_packet["front_infos"]

        # pick candidates closest to target Rs
        feasible = [info for info in front_infos if info["min_Rs"] >= 0.5]
        if feasible:
            feasible_sorted = sorted(feasible, key=lambda x: abs(x["min_Rs"] - target_Rs))
            cands = feasible_sorted[:3]
        else:
            cands = [front_infos[0]]

        sa_cfg = OptConfig(
            iterations=2000,
            init_temp=float(init_temp),
            min_conc=float(minC),
            max_conc=float(maxC),
            max_time=float(max_end_time),
            enforce_nondec=bool(enforce_nondec),
            target_Rs=float(target_Rs),
            objective_mode="constraint_then_time",   # force Rs floor
            alpha_time=float(alpha),
            step_penalty=float(step_beta),
            dt=float(dt),
            slope_limit=float(slope_limit) if slope_limit_on else None,
        )

        best_times, best_concs, best_curves = None, None, None
        for cand in cands:
            sel_t, sel_c = cand["times"], cand["concs"]
            for pkt in anneal_live(models, sel_t, sel_c, sa_cfg, update_every=sa_cfg.iterations):
                best_times, best_concs = pkt["best_times"], pkt["best_concs"]
                best_curves = pkt.get("best_curves", np.full(len(best_times), 5, dtype=int))

        st.success("NSGA-II finished and polished with SA.")

        # Ensure arrays exist
        best_times = np.asarray(best_times, float)
        best_concs = np.asarray(best_concs, float)
        best_curves = np.asarray(best_curves if best_curves is not None else np.full(len(best_times), 5, int))

        # Quantize & collapse before plotting/export
        bt, bc, bk = collapse_repeats(best_times, best_concs, best_curves)

        fig_final, res = plot_chromatogram_and_gradient(
            bt, bc, max_end_time, dt, models, curves=bk, title="NSGA-II (live) + SA polish"
        )
        st.pyplot(fig_final);
        plt.close(fig_final)

        worst_pair, worst_rs = find_critical_pair(res.tRs, res.sigmas)
        if worst_pair:
            st.info(f"Critical pair: {worst_pair[0]} vs {worst_pair[1]} — Rs = {worst_rs:.2f}")

        # Export JSON
        payload = build_export_payload(pm_name, injection_name, column_name, bt, bc, res, analyte_order, include_window)
        st.subheader("Export — JSON (NSGA+SA)")
        st.code(json.dumps(payload, indent=2), language="json")
        st.download_button(
            "⬇️ Download JSON (NSGA+SA)",
            json.dumps(payload, indent=2).encode("utf-8"),
            file_name="optimized_gradient_nsga_sa.json", mime="application/json"
        )