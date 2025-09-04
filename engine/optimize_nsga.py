from __future__ import annotations
from typing import Dict, List
import numpy as np
from sa_nsga_tools import (_tournament_select, _crowding_distance, _evaluate, _mutate, _rand_program, _merge_crossover,
                           _seed_variations, _constraints, GAConfig)


# =========================
# NSGA-II core
# =========================
def _fast_non_dominated_sort(objs: np.ndarray) -> List[np.ndarray]:
    N = len(objs)
    S = [[] for _ in range(N)]
    n = np.zeros(N, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if np.all(objs[p] <= objs[q]) and np.any(objs[p] < objs[q]):
                S[p].append(q)
            elif np.all(objs[q] <= objs[p]) and np.any(objs[q] < objs[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    return [np.array(f, dtype=int) for f in fronts[:-1]]


# =========================
# Public API
# =========================
def nsga2_live(models: Dict[str, object], seed_times, seed_concs, cfg: GAConfig):
    """Generator version of NSGA-II: yields after each generation with traces and current Pareto front."""
    # init population
    pop_t, pop_c, pop_cv = [], [], []
    n_seed = max(1, int(cfg.pop_size * cfg.seed_frac))

    seed_cv = np.full(len(seed_times), 5, dtype=int)
    seed_times, seed_concs, seed_cv = _constraints(seed_times, seed_concs, seed_cv, cfg)
    pop_t.append(seed_times); pop_c.append(seed_concs); pop_cv.append(seed_cv)
    for _ in range(n_seed - 1):
        t, c, cv = _seed_variations(seed_times, seed_concs, seed_cv, cfg)
        pop_t.append(t); pop_c.append(c); pop_cv.append(cv)
    while len(pop_t) < cfg.pop_size:
        t, c, cv = _rand_program(cfg)
        pop_t.append(t); pop_c.append(c); pop_cv.append(cv)

    # evaluate
    objs, infos = [], []
    for t, c, cv in zip(pop_t, pop_c, pop_cv):
        f, info = _evaluate(models, t, c, cv, cfg)
        objs.append(f); infos.append(info)
    objs = np.vstack(objs)

    traces = {"gen": [], "best_Rs": [], "median_Rs": [],
              "best_runtime": [], "median_runtime": [], "feasible_count": []}

    for gen in range(cfg.generations):
        # sort
        fronts = _fast_non_dominated_sort(objs)
        ranks = np.empty(len(objs), dtype=int); ranks.fill(len(fronts) + 1)
        for r, idx in enumerate(fronts):
            ranks[idx] = r
        crowd = np.zeros(len(objs))
        for idx in fronts:
            cd = _crowding_distance(objs, idx)
            crowd[idx] = cd

        # offspring
        off_t, off_c, off_cv = [], [], []
        while len(off_t) < cfg.pop_size:
            i = _tournament_select(ranks, crowd, cfg.tournament_k)
            j = _tournament_select(ranks, crowd, cfg.tournament_k)
            t1, c1, cv1 = pop_t[i], pop_c[i], pop_cv[i]
            t2, c2, cv2 = pop_t[j], pop_c[j], pop_cv[j]

            if np.random.rand() < cfg.crossover_prob:
                ct, cc, ccv = _merge_crossover(t1, c1, cv1, t2, c2, cv2, cfg)
            else:
                if np.random.rand() < 0.5:
                    ct, cc, ccv = t1.copy(), c1.copy(), cv1.copy()
                else:
                    ct, cc, ccv = t2.copy(), c2.copy(), cv2.copy()

            if np.random.rand() < cfg.mutation_prob:
                ct, cc, ccv = _mutate(ct, cc, ccv, cfg)

            off_t.append(ct); off_c.append(cc); off_cv.append(ccv)

        # evaluate combined
        cand_t = pop_t + off_t
        cand_c = pop_c + off_c
        cand_cv = pop_cv + off_cv
        cand_objs, cand_infos = [], []
        for t, c, cv in zip(cand_t, cand_c, cand_cv):
            f, info = _evaluate(models, t, c, cv, cfg)
            cand_objs.append(f); cand_infos.append(info)
        cand_objs = np.vstack(cand_objs)

        # environmental selection
        fronts = _fast_non_dominated_sort(cand_objs)
        new_t, new_c, new_cv, new_objs, new_infos = [], [], [], [], []
        for fr in fronts:
            if len(new_t) + len(fr) <= cfg.pop_size:
                for i in fr:
                    new_t.append(cand_t[i]); new_c.append(cand_c[i]); new_cv.append(cand_cv[i])
                    new_objs.append(cand_objs[i]); new_infos.append(cand_infos[i])
            else:
                cd = _crowding_distance(cand_objs, fr)
                order = fr[np.argsort(-cd)]
                for i in order[: cfg.pop_size - len(new_t)]:
                    new_t.append(cand_t[i]); new_c.append(cand_c[i]); new_cv.append(cand_cv[i])
                    new_objs.append(cand_objs[i]); new_infos.append(cand_infos[i])
                break

        pop_t, pop_c, pop_cv = new_t, new_c, new_cv
        objs, infos = np.vstack(new_objs), new_infos

        # log
        min_Rs_vals = [-o[0] for o in objs]
        runtimes = [o[1] for o in objs]
        feasibles = sum(r >= 0.0 for r in min_Rs_vals)

        traces["gen"].append(gen)
        traces["best_Rs"].append(max(min_Rs_vals))
        traces["median_Rs"].append(np.median(min_Rs_vals))
        traces["best_runtime"].append(min(runtimes))
        traces["median_runtime"].append(np.median(runtimes))
        traces["feasible_count"].append(feasibles)

        # yield current front
        fronts = _fast_non_dominated_sort(objs)
        front0 = fronts[0] if len(fronts) > 0 else np.array([], dtype=int)
        front_infos = [infos[i] for i in front0]

        yield {
            "gen": gen,
            "front_infos": front_infos,
            "infos": infos.copy(),
            "traces": {k: np.array(v) for k,v in traces.items()}
        }
