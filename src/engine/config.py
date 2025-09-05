from dataclasses import dataclass, field
from typing import Optional


# ---------- Canonical bounds & constraints (engine-wide) ----------
@dataclass(frozen=True)
class Bounds:
    time_max_min: float = 60.0          # total runtime cap (incl. equilibration)
    conc_min_mM: float = 1.0
    conc_max_mM: float = 100.0


@dataclass(frozen=True)
class Constraints:
    monotone_analysis: bool = True      # no dips during analysis segment
    return_to_start: bool = True        # append tail to return to start conc
    # Equilibration policy: either column volumes or fixed time
    eq_mode: str = "column_volumes"     # "column_volumes" | "time_min"
    eq_volumes: float = 3.0             # if eq_mode == column_volumes
    eq_time_min: float = 5.0            # fallback or if eq_mode == time_min


DEFAULT_BOUNDS = Bounds()
DEFAULT_CONSTRAINTS = Constraints()

@dataclass
class GAConfig:
    pop_size: int = 32
    generations: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.35

    # structure & constraints (use canonical bounds)
    max_points: int = 18
    min_points: int = 4
    min_conc: float = DEFAULT_BOUNDS.conc_min_mM
    max_conc: float = DEFAULT_BOUNDS.conc_max_mM
    max_time: float = DEFAULT_BOUNDS.time_max_min     # <-- was 30; now 60
    dt: float = 0.01
    enforce_nondec: bool = True
    slope_limit: Optional[float] = None

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
    min_conc: float = DEFAULT_BOUNDS.conc_min_mM
    max_conc: float = DEFAULT_BOUNDS.conc_max_mM
    max_time: float = DEFAULT_BOUNDS.time_max_min
    enforce_nondec: bool = True
    target_Rs: float = 1.8
    objective_mode: str = "constraint_then_time"  # or "weighted"
    alpha_time: float = 0.2
    step_penalty: float = 0.01
    dt: float = 0.01
    slope_limit: Optional[float] = None
    max_points: int = 18

    # search behavior
    phase1_frac: float = 0.35
    guided_prob: float = 0.35
    late_jump_prob: float = 0.20


# ---------- Resolution helpers (request/env -> runtime settings) ----------
@dataclass
class RuntimeSettings:
    bounds: Bounds = field(default_factory=lambda: DEFAULT_BOUNDS)
    constraints: Constraints = field(default_factory=lambda: DEFAULT_CONSTRAINTS)
    ga: GAConfig = field(default_factory=GAConfig)
    sa: OptConfig = field(default_factory=OptConfig)


def resolve_runtime_settings(
    req_bounds: dict | None = None,
    req_constraints: dict | None = None,
    ga_overrides: dict | None = None,
    sa_overrides: dict | None = None,
) -> RuntimeSettings:
    """Merge request overrides onto engine defaults. Request > defaults."""
    b = DEFAULT_BOUNDS if not req_bounds else Bounds(**{**DEFAULT_BOUNDS.__dict__, **req_bounds})
    c = DEFAULT_CONSTRAINTS if not req_constraints else Constraints(**{**DEFAULT_CONSTRAINTS.__dict__, **req_constraints})

    ga = GAConfig(**({**GAConfig().__dict__, **(ga_overrides or {})}))
    sa = OptConfig(**({**OptConfig().__dict__, **(sa_overrides or {})}))

    # Keep GA/SA aligned with canonical bounds regardless of overrides
    ga.min_conc = b.conc_min_mM
    ga.max_conc = b.conc_max_mM
    ga.max_time = b.time_max_min
    sa.min_conc = b.conc_min_mM
    sa.max_conc = b.conc_max_mM
    sa.max_time = b.time_max_min

    return RuntimeSettings(bounds=b, constraints=c, ga=ga, sa=sa)
