"""
Korean Audio Dataset API
-------------------------
Receives: {"audio_id": "q0", "audio_base64": "..."}
Returns:  a JSON object with rows/columns/mean/std/... statistics
          computed from the decoded audio waveform.
"""

import base64
import io

import numpy as np
import pandas as pd
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Korean Audio Dataset API")


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def decode_audio_to_dataframe(audio_bytes: bytes):
    """
    Decode WAV, MP3, FLAC, OGG, etc. into a pandas DataFrame (one column
    per channel). soundfile handles format detection internally and needs
    no external ffmpeg/ffprobe binaries -- it's a self-contained library.
    """
    samples, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="int16", always_2d=True)
    # samples shape: (n_frames, n_channels)
    n_channels = samples.shape[1]
    dtype = samples.dtype
    sampwidth = dtype.itemsize

    df = pd.DataFrame(samples, columns=[f"channel_{i}" for i in range(n_channels)])

    return df, dtype, sampwidth


def to_py(obj):
    """Recursively convert numpy types to native Python types (for clean JSON)."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return to_py(obj.tolist())
    return obj


@app.post("/analyze")
def analyze(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    try:
        df, dtype, sampwidth = decode_audio_to_dataframe(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse audio: {e}")

    result = {
        "rows": len(df),
        "columns": list(df.columns),
        "mean": df.mean().round(4).to_dict(),
        "std": df.std().round(4).to_dict(),
        "variance": df.var().round(4).to_dict(),
        "min": df.min().to_dict(),
        "max": df.max().to_dict(),
        "median": df.median().to_dict(),
        "mode": df.mode().iloc[0].to_dict() if not df.mode().empty else {},
        "range": (df.astype(np.int64).max() - df.astype(np.int64).min()).to_dict(),
        "allowed_values": {},
        "value_range": {},
        "correlation": df.corr().round(4).values.tolist() if len(df.columns) > 1 else [[1.0]],
    }

    return to_py(result)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Korean Audio Dataset API is running"}
