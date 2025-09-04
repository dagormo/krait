from gradient_tools import (
    build_gradient_profile,
    round_to,
    enforce_slope,
    enforce_nondec_concentration,
)
import numpy as np
import math
from simulate_tools import simulate_chromatogram, evaluate_resolution, pairwise_resolution
from typing import Dict, Tuple
from config import GAConfig, OptConfig


def _crowding_distance(objs: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if len(idx) == 0:
        return np.array([])
    M = objs.shape[1]
    d = np.zeros(len(idx), dtype=float)
    F = objs[idx]
    for m in range(M):
        order = np.argsort(F[:, m])
        d[order[0]] = d[order[-1]] = np.inf
        min_m, max_m = F[order[0], m], F[order[-1], m]
        span = max(1e-12, max_m - min_m)
        for i in range(1, len(idx) - 1):
            d[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / span
    return d


def _tournament_select(ranks, crowd, k: int):
    cand = np.random.choice(len(ranks), size=k, replace=False)
    best = cand[0]
    for i in cand[1:]:
        if ranks[i] < ranks[best] or (ranks[i] == ranks[best] and crowd[i] > crowd[best]):
            best = i
    return best


def _constraints(times, concs, curves, cfg: GAConfig):
    times = round_to(times, 0.1).astype(float)
    concs = np.round(concs).astype(float)
    concs = np.clip(concs, cfg.min_conc, cfg.max_conc)
    curves = np.array(curves, int)
    curves = np.clip(curves, 1, 9)

    # times non-decreasing
    times = np.maximum.accumulate(times)
    times[0] = 0.0

    if cfg.slope_limit:
        times, concs = enforce_slope(times, concs, float(cfg.slope_limit))

    if cfg.enforce_nondec:
        times, concs = enforce_nondec_concentration(times, concs)

    # trim or pad
    if len(times) > cfg.max_points:
        keep = np.linspace(0, len(times) - 1, cfg.max_points).round().astype(int)
        times, concs, curves = times[keep], concs[keep], curves[keep]
    if len(times) < cfg.min_points:
        k = cfg.min_points
        tpad = np.linspace(0, max(times[-1] if len(times) else 1.0, 1.0), k)
        cpad = np.linspace(concs[0] if len(concs) else cfg.min_conc,
                           concs[-1] if len(concs) else cfg.max_conc, k)
        cvpad = np.full(k, 5, dtype=int)
        times, concs, curves = tpad, cpad, cvpad

    return round_to(times, 0.1), np.round(concs), curves


def _evaluate(models: Dict[str, object], times, concs, curves, cfg: GAConfig):
    tg, Cg = build_gradient_profile(times, concs, cfg.max_time, cfg.dt, curves=curves)
    _, tRs, sigmas = simulate_chromatogram(models, tg, Cg)
    min_Rs, _ = evaluate_resolution(tRs, sigmas)
    last_peak = max(tRs.values()) if tRs else 0.0
    end_time = min(cfg.max_time, math.ceil((last_peak + 0.5) * 10) / 10.0)
    steps = max(0, len(times) - 1)
    objs = np.array([-float(min_Rs), float(end_time), float(steps)], dtype=float)
    info = dict(min_Rs=float(min_Rs), end_time=float(end_time), steps=int(steps),
                times=times, concs=concs, curves=curves, tg=tg, Cg=Cg)
    return objs, info


def _rand_program(cfg: GAConfig):
    k = np.random.randint(cfg.min_points, cfg.max_points + 1)
    t = np.sort(np.random.uniform(0.0, cfg.max_time * 0.9, size=k))
    t[0] = 0.0
    c = np.sort(np.random.uniform(cfg.min_conc, cfg.max_conc, size=k))
    cv = np.full(k, 5, dtype=int)
    return _constraints(t, c, cv, cfg)


def _seed_variations(seed_t, seed_c, seed_cv, cfg: GAConfig):
    t = np.asarray(seed_t, float).copy()
    c = np.asarray(seed_c, float).copy()
    cv = np.asarray(seed_cv, int).copy()
    t += np.random.normal(0, cfg.time_jitter_sd, size=t.shape)
    c += np.random.normal(0, cfg.conc_jitter_sd, size=c.shape)
    if len(t) < cfg.max_points and np.random.rand() < 0.4:
        i = np.random.randint(0, len(t)-1)
        t_mid = np.random.uniform(min(t[i], t[i+1]), max(t[i], t[i+1]))
        c_mid = np.random.uniform(min(c[i], c[i+1]), max(c[i], c[i+1]))
        t = np.insert(t, i+1, t_mid); c = np.insert(c, i+1, c_mid); cv = np.insert(cv, i+1, 5)
    if len(t) > cfg.min_points and np.random.rand() < 0.3:
        i = np.random.randint(1, len(t)-1)
        t = np.delete(t, i); c = np.delete(c, i); cv = np.delete(cv, i)
    return _constraints(t, c, cv, cfg)


def _merge_crossover(p1_t, p1_c, p1_cv, p2_t, p2_c, p2_cv, cfg: GAConfig):
    ts = np.unique(np.round(np.concatenate([p1_t, p2_t]), 1))
    ts = np.sort(ts)
    c1 = np.interp(ts, p1_t, p1_c)
    c2 = np.interp(ts, p2_t, p2_c)
    cs = 0.5 * (c1 + c2)
    # curves: pick randomly from parent1 or parent2
    cv = np.array([np.random.choice([cv1, cv2]) for cv1, cv2 in zip(
        np.interp(ts, p1_t, p1_cv), np.interp(ts, p2_t, p2_cv))], dtype=int)
    return _constraints(ts, cs, cv, cfg)


def _mutate(times, concs, curves, cfg: GAConfig):
    t = np.asarray(times, float).copy()
    c = np.asarray(concs, float).copy()
    cv = np.asarray(curves, int).copy()
    n = len(t)
    op = np.random.choice(["time", "conc", "add", "del", "curve"], p=[0.25, 0.25, 0.2, 0.1, 0.2])

    if op == "time" and n > 1:
        i = np.random.randint(1, n)
        t[i] = np.clip(t[i] + np.random.normal(0, 0.6), 0, cfg.max_time)
    elif op == "conc":
        i = np.random.randint(0, n)
        c[i] += np.random.normal(0, 2.0)
    elif op == "add" and n < cfg.max_points:
        i = np.random.randint(0, n-1)
        t_mid = np.random.uniform(min(t[i], t[i+1]), max(t[i], t[i+1]))
        c_mid = np.random.uniform(min(c[i], c[i+1]), max(c[i], c[i+1]))
        t = np.insert(t, i+1, t_mid); c = np.insert(c, i+1, c_mid); cv = np.insert(cv, i+1, 5)
    elif op == "del" and n > cfg.min_points:
        i = np.random.randint(1, n-1)
        t = np.delete(t, i); c = np.delete(c, i); cv = np.delete(cv, i)
    elif op == "curve":
        i = np.random.randint(0, len(cv))
        step = np.random.choice([-1, 1])
        cv[i] = int(np.clip(cv[i] + step, 1, 9))

    return _constraints(t, c, cv, cfg)


def _constraints(times: np.ndarray, concs: np.ndarray, curves: np.ndarray, cfg: OptConfig):
    times = round_to(times, 0.1).astype(float)
    concs = np.round(concs).astype(float)
    concs = np.clip(concs, cfg.min_conc, cfg.max_conc)
    curves = np.array(curves, int)
    curves = np.clip(curves, 1, 9)

    times = np.clip(times, 0, cfg.max_time)
    # no np.maximum.accumulate, just sort
    order = np.argsort(times, kind="stable")
    times, concs, curves = times[order], concs[order], curves[order]

    times[0] = 0.0

    if cfg.slope_limit:
        times, concs = enforce_slope(times, concs, float(cfg.slope_limit))

    if cfg.enforce_nondec:
        times, concs = enforce_nondec_concentration(times, concs)

    if len(times) > cfg.max_points:
        keep = np.linspace(0, len(times) - 1, cfg.max_points).round().astype(int)
        times, concs, curves = times[keep], concs[keep], curves[keep]

    return round_to(times, 0.1), np.round(concs), curves


def _objective_timeseries(models: Dict[str, object], times, concs, curves, cfg: OptConfig) -> Tuple[float, dict]:
    tg, Cg = build_gradient_profile(times, concs, cfg.max_time, cfg.dt, curves=curves)
    _, tRs, sigmas = simulate_chromatogram(models, tg, Cg)
    min_Rs, _ = evaluate_resolution(tRs, sigmas)
    last_peak = max(tRs.values()) if tRs else 0.0
    end_time = min(cfg.max_time, math.ceil((last_peak + 0.5) * 10) / 10.0)
    steps_penalty = cfg.step_penalty * max(0, len(times) - 2)

    if cfg.objective_mode == "constraint_then_time":
        if min_Rs < cfg.target_Rs:
            score = 1e6 + (cfg.target_Rs - min_Rs) * 1e4 + end_time + steps_penalty
        else:
            score = end_time + steps_penalty
    else:
        score = (
            cfg.alpha_time * end_time
            + (1.0 - cfg.alpha_time) * (1.0 / max(min_Rs, 1e-6))
            + steps_penalty
        )

    return float(score), {
        "min_Rs": float(min_Rs),
        "end_time": float(end_time),
        "tRs": tRs,
        "tg": tg,
        "Cg": Cg,
        "curves": curves,
    }


# before anneal_live
def _worst_pair_info(tRs: Dict[str, float], sigmas: Dict[str, float]):
    items = sorted(tRs.items(), key=lambda kv: kv[1])
    worst_pair, worst_rs, mid_t = None, float("inf"), None
    for i in range(len(items) - 1):
        n1, t1 = items[i]
        n2, t2 = items[i + 1]
        rs = pairwise_resolution(t1, sigmas[n1], t2, sigmas[n2])
        if rs < worst_rs:
            worst_rs = rs
            worst_pair = (n1, n2)
            mid_t = 0.5 * (t1 + t2)
    return worst_pair, mid_t or 0.0, worst_rs


# ---------- Guided moves ----------
def _mutate_random(times, concs, curves, cfg: OptConfig):
    times = np.asarray(times, float).copy()
    concs = np.asarray(concs, float).copy()
    curves = np.asarray(curves, int).copy()
    n = len(times)

    move = np.random.choice(["time", "conc", "add", "del", "curve"], p=[0.3, 0.3, 0.2, 0.1, 0.1])

    if move == "time" and n > 1:
        i = np.random.randint(1, n)
        times[i] = np.clip(times[i] + np.random.normal(0, 0.6), 0.0, cfg.max_time)

    elif move == "conc":
        i = np.random.randint(0, n)
        concs[i] = np.clip(concs[i] + np.random.normal(0, 2.0), cfg.min_conc, cfg.max_conc)

    elif move == "add" and n < cfg.max_points:
        i = np.random.randint(0, n - 1)
        t_mid = np.random.uniform(times[i], times[i+1])
        c_mid = np.random.uniform(min(concs[i], concs[i+1]), max(concs[i], concs[i+1]))
        times = np.insert(times, i+1, t_mid)
        concs = np.insert(concs, i+1, c_mid)
        curves = np.insert(curves, i+1, 5)

    elif move == "del" and n > 3:
        i = np.random.randint(1, n - 1)
        times = np.delete(times, i)
        concs = np.delete(concs, i)
        curves = np.delete(curves, i)

    elif move == "curve":
        i = np.random.randint(0, len(curves))
        step = np.random.choice([-1, 1])
        curves[i] = int(np.clip(curves[i] + step, 1, 9))

    return _constraints(times, concs, curves, cfg)


def _late_jump_after_penultimate(tRs: Dict[str, float], times, concs, curves, cfg: OptConfig):
    if len(tRs) < 2:
        return times, concs, curves
    sorted_tr = sorted(tRs.values())
    t2 = sorted_tr[-2]
    jump_t = round(max(0.0, t2 + 0.1), 1)
    times = np.append(times, [jump_t, min(cfg.max_time, math.ceil(jump_t + 1.0))])
    concs = np.append(concs, [cfg.max_conc, cfg.max_conc])
    curves = np.append(curves, [5, 5])
    return _constraints(times, concs, curves, cfg)
