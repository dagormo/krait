from engine.gradient_tools import build_gradient_profile
from engine.simulate_tools import simulate_chromatogram, evaluate_resolution
import matplotlib.pyplot as plt
from collections import namedtuple

SimResult = namedtuple("SimResult", "tg Cg y tRs sigmas min_Rs")


def plot_chromatogram_and_gradient(times, concs, max_end_time, dt, models, curves=None, title="Simulated Chromatogram"):
    tg, Cg = build_gradient_profile(times, concs, max_end_time, dt, curves=curves)
    y, tRs, sigmas = simulate_chromatogram(models, tg, Cg)
    min_Rs, _ = evaluate_resolution(tRs, sigmas)

    fig, ax1 = plt.subplots(figsize=(9.5, 4))
    ax1.plot(tg, y, label="Simulated chromatogram")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Signal (a.u.)")

    for name, tR in tRs.items():
        ax1.axvline(tR, color="C0", linestyle=":", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(tg, Cg, label="Eluent conc (mM)")
    ax2.set_ylabel("Eluent concentration (mM)")
    ax1.set_title(title)

    return fig, SimResult(tg, Cg, y, tRs, sigmas, min_Rs)