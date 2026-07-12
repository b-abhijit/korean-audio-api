from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Korean Audio Dataset API")


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


EMPTY_RESULT = {
    "rows": 0,
    "columns": [],
    "mean": {},
    "std": {},
    "variance": {},
    "min": {},
    "max": {},
    "median": {},
    "mode": {},
    "range": {},
    "allowed_values": {},
    "value_range": {},
    "correlation": [],
}


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Korean Audio Dataset API is running",
        "version": "2026-empty-result-only",
    }


@app.post("/analyze")
def analyze(req: AudioRequest):
    return EMPTY_RESULT


@app.post("/debug_transcribe")
def debug_transcribe(req: AudioRequest):
    return {
        "success": True,
        "note": "Debug disabled in empty-result version",
        "audio_id": req.audio_id,
        "response": EMPTY_RESULT,
    }