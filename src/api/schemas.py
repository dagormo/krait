from __future__ import annotations
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, model_validator


# ---- Request ----
class CMContext(BaseModel):
    processing_method_name: str
    injection_name: str
    column: str


class Analyte(BaseModel):
    name: str
    rt_at_c1_min: float
    rt_at_c2_min: float
    rt_at_c3_min: float


class Scouting(BaseModel):
    conc1_mM: float = Field(..., ge=0.01)
    conc2_mM: float = Field(..., ge=0.01)
    conc3_mM: float = Field(..., ge=0.01)


class Bounds(BaseModel):
    time_max_min: float = Field(60.0, gt=0)     # hard cap 60
    min_conc_mM: float = Field(1.0, ge=0.0)
    max_conc_mM: float = Field(100.0, gt=0.0)


    @model_validator(mode="after")
    def clamp(self) -> "Bounds":
        self.time_max_min = min(60.0, float(self.time_max_min))
        self.min_conc_mM = max(1.0, float(self.min_conc_mM))
        self.max_conc_mM = min(100.0, float(self.max_conc_mM))
        return self


class Constraints(BaseModel):
    enforce_nondec: bool = True
    slope_limit: Optional[float] = None         # mM/min
    target_Rs: float = 1.8
    equilibration_min: float = 5.0              # hold after return-to-start


class Weights(BaseModel):
    alpha_time: float = 0.2                     # only used if objective="weighted"
    step_penalty: float = 0.02


class SeedNode(BaseModel):
    t_min: float
    c_mM: float


class CMOptimizeRequest(BaseModel):
    context: CMContext
    analytes: List[Analyte]
    scouting: Scouting
    bounds: Bounds = Bounds()
    constraints: Constraints = Constraints()
    weights: Weights = Weights()
    objective: str = Field("constraint_then_time", pattern="^(constraint_then_time|weighted)$")
    t0_min: Optional[float] = None
    plates: int = 50_000
    dt: float = 0.01
    seed: Optional[List[SeedNode]] = None
    include_window: bool = True


# ---- Response ----
class Preview(BaseModel):
    times_min: List[float]
    concs_mM: List[float]
    predicted_rt_min: Dict[str, float]
    min_Rs: float


class CMOptimizeResponse(BaseModel):
    export: dict
    preview: Preview
