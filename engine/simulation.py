import numpy as np


def calibrate_velocity(concs, rts, t0, L):
    """Calibrate velocity constant for Method-3 gradient model."""
    t_prime = rts - t0
    v_iso = L / t_prime
    return np.dot(concs, v_iso) / np.dot(concs, concs)


def build_conc_function(points):
    """Builds a time → concentration interpolation function from gradient points."""
    times, concs = zip(*points)
    return (lambda t: np.interp(t, times, concs)), np.array(times), np.array(concs)


def predict_rt(conc_at, k, L, t0, dt=0.01, t_max=None):
    """Numerically integrate gradient profile to predict retention time."""
    if t_max is None:
        t_max = 100
    t, L_acc = 0.0, 0.0
    while L_acc < L and t < t_max:
        L_acc += (k * conc_at(t)) * dt
        t += dt
    return t + t0, t
