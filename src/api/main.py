from fastapi import FastAPI, HTTPException
from api.schemas import CMOptimizeRequest, CMOptimizeResponse, Preview
from api.adapters.chromeleon_export import build_export_payload
from engine.optimizer_facade import run_optimize

app = FastAPI(title="Krait API")

@app.get("/")
def root():
    return {"service": "krait", "ok": True}

@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/chromeleon/optimize", response_model=CMOptimizeResponse)
def chromeleon_optimize(req: CMOptimizeRequest):
    try:
        result = run_optimize(
            analytes=[a.dict() for a in req.analytes],
            triad_concs=(req.scouting.conc1_mM, req.scouting.conc2_mM, req.scouting.conc3_mM),
            bounds=req.bounds.dict(), constraints=req.constraints.dict(),
            weights=req.weights.dict(), objective=req.objective,
            t0_min=req.t0_min, plates=req.plates, dt=req.dt,
            seed=[s.dict() for s in req.seed] if req.seed else None,
        )
        bt, bc, res = result["times"], result["concs"], result["res"]
        payload = build_export_payload(
            pm_name=req.context.processing_method_name,
            injection_name=req.context.injection_name,
            column_name=req.context.column,
            bt=bt, bc=bc, res=res,
            names_fallback=result["analyte_names"],
            include_window=req.include_window,
        )
        preview = Preview(
            times_min=[float(x) for x in bt],
            concs_mM=[float(x) for x in bc],
            predicted_rt_min={k: float(v) for k, v in result["predicted_rt"].items()},
            min_Rs=float(result["min_Rs"]),
        )
        return CMOptimizeResponse(export=payload, preview=preview)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"optimize failed: {e}")
