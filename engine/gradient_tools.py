import numpy as np

__all__ = [
    "round_to", "clamp", "build_gradient_profile", "enforce_slope",
    "seed_from_df", "collapse_repeats", "enforce_nondec_concentration",
    "seed_from_df"
]


def round_to(x, step):
    return np.round(np.asarray(x, dtype=float) / step) * step


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def build_gradient_profile(times, concs, t_end, dt):
    """Piecewise-linear ramps; equal timestamps = instantaneous steps."""
    times = np.asarray(times, dtype=float)
    concs = np.asarray(concs, dtype=float)
    t_end = max(float(t_end), float(np.max(times)))
    grid = np.arange(0.0, t_end + dt, dt)
    C = np.zeros_like(grid)

    # stable sort by time (keeps order of equal timestamps)
    order = np.lexsort((np.arange(len(times)), times))
    times = times[order]; concs = concs[order]

    # linear ramps between increasing times
    for i in range(len(times) - 1):
        t0, c0 = times[i], concs[i]
        t1, c1 = times[i + 1], concs[i + 1]
        if t1 > t0:
            m = (grid >= t0) & (grid < t1)
            if np.any(m):
                frac = (grid[m] - t0) / (t1 - t0)
                C[m] = c0 + frac * (c1 - c0)

    # hold after last point; pre-fill before first
    last_t, last_c = times[-1], concs[-1]
    C[grid >= last_t] = last_c
    first_t, first_c = times[0], concs[0]
    C[grid < first_t] = first_c

    # forward-fill any untouched interior nodes
    last_val = first_c
    for i, t in enumerate(grid):
        if (t >= first_t) and (t < last_t) and (C[i] == 0):
            C[i] = last_val
        else:
            last_val = C[i]
    return grid, C


def enforce_slope(times, concs, max_slope):
    """Clip each segment’s slope to |slope| <= max_slope."""
    times = np.asarray(times, float).copy()
    concs = np.asarray(concs, float).copy()
    for i in range(len(times) - 1):
        dt = max(1e-6, times[i+1] - times[i])
        slope = (concs[i+1] - concs[i]) / dt
        if abs(slope) > max_slope:
            concs[i+1] = concs[i] + np.sign(slope) * max_slope * dt
    return times, concs


def enforce_nondec_concentration(times, concs):
    """Force concentration to be non-decreasing over time (steps allowed)."""
    times = np.asarray(times, float)
    concs = np.asarray(concs, float)
    order = np.lexsort((np.arange(len(times)), times))  # stable by time
    times = times[order]
    concs = concs[order]
    concs = np.maximum.accumulate(concs)  # monotone non-decreasing
    return times, concs


def seed_from_df(df, min_step=0.1):
    g = df.dropna().copy()
    times = round_to(g["time_min"].to_numpy(float), 0.1)
    concs = np.round(g["conc_mM"].to_numpy(float))
    order = np.lexsort((np.arange(len(times)), times))
    times, concs = times[order], concs[order]
    # ensure start at 0 and minimum spacing
    times[0] = 0.0
    for i in range(1, len(times)):
        if times[i] < times[i-1] + min_step and times[i] > times[i-1]:
            times[i] = times[i-1] + min_step
    return times, concs


def apply_curve_segment(Vr, Vt, Tr, Tt, curve=5, n_points=50):
    times = np.linspace(Tr, Tt, n_points)
    dt = Tt - Tr

    if curve == 5 or dt <= 0:
        # linear
        concs = np.linspace(Vr, Vt, n_points)
        return times, concs

    if curve in [1, 2, 3, 4]:
        k_values = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75}
        k = k_values[curve]
        frac = (times - Tr) / dt
        term = (1 - 2**(-10 * frac)) / (1 - 2**(-10))
        concs = Vr + (1 - k) * (Vt - Vr) * term + k * (Vt - Vr) * frac
    elif curve in [6, 7, 8, 9]:
        k_values = {6: 0.75, 7: 0.5, 8: 0.25, 9: 0.0}
        k = k_values[curve]
        frac = (Tt - times) / dt
        term = (1 - 2**(-10 * frac)) / (1 - 2**(-10))
        concs = Vt - (1 - k) * (Vt - Vr) * term - k * (Vt - Vr) * frac
    else:
        raise ValueError(f"Unsupported curve number: {curve}")

    return times, concs


def collapse_repeats(times, concs, curves=None, time_step=0.1, conc_step=1.0, conc_tol=1e-6):
    # to arrays
    t = np.asarray(times, dtype=float)
    c = np.asarray(concs, dtype=float)
    k = np.asarray(curves, dtype=float) if curves is not None else None

    # sort stably by time (preserve order of same-time "instant steps")
    order = np.lexsort((np.arange(len(t)), t))
    t, c = t[order], c[order]
    if k is not None:
        k = k[order]

    # quantize (robust to jitter)
    if time_step is not None:
        t = np.round(t / time_step) * time_step
    if conc_step is not None:
        c = np.round(c / conc_step) * conc_step

    # remove exact duplicate rows
    keep = [0]
    for i in range(1, len(t)):
        same_t = (t[i] == t[i-1])
        same_c = (abs(c[i] - c[i-1]) < conc_tol)
        same_k = (k is not None and k[i] == k[i-1])
        if not (same_t and same_c and (same_k if k is not None else True)):
            keep.append(i)
    t, c = t[keep], c[keep]
    if k is not None:
        k = k[keep]

    n = len(t)
    if n <= 2:
        return t.tolist(), c.tolist(), (k.tolist() if k is not None else None)

    # collapse plateaus → keep first and last where conc is constant
    out_t, out_c, out_k = [], [], []
    for i in range(n):
        if i == 0 or i == n-1:
            keep_i = True
        else:
            prev_same = abs(c[i] - c[i-1]) < conc_tol
            next_same = abs(c[i] - c[i+1]) < conc_tol
            # skip only if this point is interior of a plateau (same as both neighbors)
            keep_i = not (prev_same and next_same)

        if keep_i:
            out_t.append(t[i])
            out_c.append(c[i])
            if k is not None:
                out_k.append(k[i])

    return out_t, out_c, (out_k if k is not None else None)