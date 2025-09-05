from fastapi import FastAPI

app = FastAPI(title="Krait API")

@app.get("/ping")
def ping():
    return {"ok": True}
