from dataclasses import dataclass
import pandas as pd


@dataclass
class GAConfig:
    pop_size: int = 32
    generations: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.35

    # structure & constraints
    max_points: int = 18
    min_points: int = 4
    min_conc: float = 1.0
    max_conc: float = 100.0
    max_time: float = 30.0
    dt: float = 0.01
    enforce_nondec: bool = True
    slope_limit: float | None = None

    # seeding
    seed_frac: float = 0.35
    time_jitter_sd: float = 0.5
    conc_jitter_sd: float = 2.0

    # selection
    tournament_k: int = 4


@dataclass
class OptConfig:
    iterations: int = 8000
    init_temp: float = 1.2
    min_conc: float = 1.0
    max_conc: float = 100.0
    max_time: float = 60.0
    enforce_nondec: bool = True
    target_Rs: float = 1.8
    objective_mode: str = "constraint_then_time"  # or "weighted"
    alpha_time: float = 0.2
    step_penalty: float = 0.01
    dt: float = 0.01
    slope_limit: float | None = None
    max_points: int = 18

    # search behavior
    phase1_frac: float = 0.35
    guided_prob: float = 0.35
    late_jump_prob: float = 0.20


def default_analytes_table():
    rows = [
        ("Bromide", 6.249, 4.83, 3.699),
        ("Carbonate", 4.952, 3.205, 2.335),
        ("Chloride", 3.395, 2.883, 2.481),
        ("Fluoride", 2.12, 2.026, 1.943),
        ("Iodide", 28.057, 19.69, 13.023),
        ("Nitrate", 6.609, 5.059, 3.82),
        ("Nitrite", 3.93, 3.254, 2.707),
        ("Phosphate", 17.809, 7.067, 3.128),
        ("Sulfate", 6.296, 3.756, 2.521),
    ]
    return pd.DataFrame(rows, columns=["Analyte","RT@Conc1 (min)","RT@Conc2 (min)","RT@Conc3 (min)"])

def default_gradient_table():
    return pd.DataFrame({
        "time_min": [0.0, 10.0, 20.0],
        "conc_mM": [5, 50, 100],
        "curve":   [9, 1, 5]
    })

