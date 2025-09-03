import numpy as np
import matplotlib.pyplot as plt

def calibrate_velocity(conc_iso, rt_iso, t0, L):
    t_prime = rt_iso - t0
    v_iso    = L / t_prime
    k = np.dot(conc_iso, v_iso) / np.dot(conc_iso, conc_iso)
    return k

def build_conc_function(gradient_points):
    times, concs = zip(*gradient_points)
    times = np.array(times)
    concs = np.array(concs)
    def conc_at(t):
        return np.interp(t, times, concs)
    return conc_at, times, concs

def predict_rt(conc_at, k, L, t0, dt=0.01, t_max=None):
    if t_max is None:
        t_max = 100

    t = 0.0
    L_acc = 0.0
    while L_acc < L and t < t_max:
        c = conc_at(t)
        v = k * c
        L_acc += v * dt
        t += dt

    return t + t0, t  # observed RT, adjusted time

def plot_gradient(times, concs, rt_adj):
    plt.figure(figsize=(8,4))
    plt.plot(times, concs, '-', linewidth=2)
    plt.axvline(rt_adj, color='gray', linestyle='--', label=f'Elution at {rt_adj:.2f} min (adjusted)')
    plt.xlabel("Time (min)")
    plt.ylabel("Eluent Concentration (mM)")
    plt.title("Multi-Step Gradient Profile")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- 1. Read isocratic calibration ---
    conc_iso = np.array(
        list(map(float,
            input("Enter isocratic concentrations (mM), comma‐separated: ")
            .split(','))))
    rt_iso = np.array(
        list(map(float,
            input("Enter corresponding retention times (min), comma‐separated: ")
            .split(','))))
    t0 = float(input("Enter void time t0 (min): "))
    L  = float(input("Enter column length L (cm): "))

    # --- 2. Read gradient profile ---
    n = int(input("How many gradient points? "))
    gradient_points = []
    print("Enter each point as: time(min),concentration(mM)")
    for i in range(n):
        t, c = map(float, input(f" Point {i+1}: ").split(','))
        gradient_points.append((t, c))

    # --- 3. Calibrate velocity and build gradient function ---
    k = calibrate_velocity(conc_iso, rt_iso, t0, L)
    print(f"\nCalibrated velocity constant k = {k:.4f} cm·min⁻¹·mM⁻¹")
    conc_at, times, concs = build_conc_function(gradient_points)

    # --- 4. Predict retention time ---
    rt_obs, rt_adj = predict_rt(conc_at, k, L, t0, dt=0.01,
                                t_max=times[-1]+100)
    print(f"Predicted RT = {rt_obs:.2f} min")

    # --- 5. Plot gradient and elution time ---
    plot_gradient(times, concs, rt_adj)