import numpy as np
from gradient_tools import build_gradient_profile
from simulate_tools import simulate_chromatogram
from sa_nsga_tools import (OptConfig, _worst_pair_info, _constraints,_objective_timeseries,
                           _mutate_random, _late_jump_after_penultimate)


def anneal_live(models, seed_times, seed_concs, cfg: OptConfig, update_every=100):
    np.random.seed(42)

    seed_curves = np.full(len(seed_times), 5, dtype=int)
    times, concs, curves = _constraints(seed_times, seed_concs, seed_curves, cfg)
    cur_score, cur_info = _objective_timeseries(models, times, concs, curves, cfg)

    best_times, best_concs, best_curves = times.copy(), concs.copy(), curves.copy()
    best_score = cur_score
    reached_feasible = (cur_info["min_Rs"] >= cfg.target_Rs)

    traces = {"iter": [], "current": [], "best": [], "minrs": [], "runtime": [], "steps": []}
    T0 = cfg.init_temp

    def _accept(cand, cur, T):
        return (cand < cur) or (np.random.rand() < np.exp(-(cand - cur) / max(T, 1e-9)))

    for it in range(cfg.iterations):
        T = max(1e-6, T0 * (1.0 - it / cfg.iterations))

        if not reached_feasible and np.random.rand() < cfg.guided_prob:
            tg, Cg = build_gradient_profile(times, concs, cfg.max_time, cfg.dt, curves=curves)
            _, tRs, sigmas = simulate_chromatogram(models, tg, Cg)
            _, mid_t, _ = _worst_pair_info(tRs, sigmas)
            cand_t, cand_c, cand_cv = _mutate_random(times, concs, curves, cfg)

        elif reached_feasible and np.random.rand() < cfg.late_jump_prob:
            cand_t, cand_c, cand_cv = _late_jump_after_penultimate(cur_info["tRs"], times, concs, curves, cfg)
        else:
            cand_t, cand_c, cand_cv = _mutate_random(times, concs, curves, cfg)

        cand_score, cand_info = _objective_timeseries(models, cand_t, cand_c, cand_cv, cfg)

        traces["iter"].append(it)
        traces["current"].append(cand_score)
        traces["best"].append(best_score)
        traces["minrs"].append(cand_info["min_Rs"])
        traces["runtime"].append(cand_info["end_time"])
        traces["steps"].append(len(cand_t) - 1)

        if _accept(cand_score, cur_score, T):
            times, concs, curves = cand_t, cand_c, cand_cv
            cur_score, cur_info = cand_score, cand_info
            if cand_score < best_score:
                best_score = cand_score
                best_times, best_concs, best_curves = cand_t, cand_c, cand_cv

        if not reached_feasible and cur_info["min_Rs"] >= cfg.target_Rs:
            reached_feasible = True
            T0 = max(T0, 0.8)

        if (it + 1) % update_every == 0:
            yield {
                "current_times": times.copy(),
                "current_concs": concs.copy(),
                "current_curves": curves.copy(),
                "best_times": best_times.copy(),
                "best_concs": best_concs.copy(),
                "best_curves": best_curves.copy(),
                "traces": {k: np.asarray(v) for k, v in traces.items()},
            }

    yield {
        "current_times": times.copy(),
        "current_concs": concs.copy(),
        "current_curves": curves.copy(),
        "best_times": best_times.copy(),
        "best_concs": best_concs.copy(),
        "best_curves": best_curves.copy(),
        "traces": {k: np.asarray(v) for k, v in traces.items()},
    }
