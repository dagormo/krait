import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA = os.path.join(PROJECT_ROOT, "data")
PKL = os.path.join(PROJECT_ROOT, "pkl")

PCA_MODEL_PATH = os.path.join(PKL, "pca_model_95pct.pkl")
MODEL_PATH = os.path.join(PKL, "logk_model_hydroxide.pkl")

from dataclasses import dataclass


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
