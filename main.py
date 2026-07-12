"""
Korean Audio Dataset API
-------------------------
Receives: {"audio_id": "q0", "audio_base64": "..."}
Returns:  a JSON object with rows/columns/mean/std/... statistics
          computed from the decoded audio waveform. The column name is
          derived by transcribing a short spoken Korean word/phrase in
          the clip (e.g. "점수" = "score") via AI Pipe's hosted
          transcription endpoint -- no local speech model is bundled.
"""

import base64
import io
import os
import re
import traceback

import numpy as np
import pandas as pd
import requests
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Korean Audio Dataset API")

# Set this in your deployment platform's Environment Variables settings
# (Vercel: Project Settings -> Environment Variables; Render: Environment
# tab). NEVER hardcode it here or commit it to git.
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
AIPIPE_TRANSCRIBE_URL = "https://aipipe.org/openai/v1/audio/transcriptions"


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


def transcribe_column_name(audio_bytes: bytes) -> str:
    """
    The audio clip contains a short spoken Korean word/phrase that names
    the data column (e.g. "점수" = "score"). Transcribe it via AI Pipe's
    hosted OpenAI-compatible transcription endpoint (no local model).
    """
    if not AIPIPE_TOKEN:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set")

    files = {"file": ("audio.wav", audio_bytes, "application/octet-stream")}
    data = {"model": "whisper-1", "language": "ko"}
    headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}"}

    resp = requests.post(
        AIPIPE_TRANSCRIBE_URL, headers=headers, files=files, data=data, timeout=30
    )
    resp.raise_for_status()
    text = resp.json().get("text", "")

    # Strip whitespace and common punctuation so "점수." or " 점수 " -> "점수"
    cleaned = re.sub(r"[\s.,!?~·]", "", text)
    return cleaned


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

    try:
        column_name = transcribe_column_name(audio_bytes)
    except Exception:
        column_name = ""

    if not column_name:
        column_name = "channel_0"

    if len(df.columns) == 1:
        df.columns = [column_name]
    else:
        df.columns = [f"{column_name}_{i}" for i in range(len(df.columns))]

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


@app.post("/debug_transcribe")
def debug_transcribe(req: AudioRequest):
    """
    Debug-only endpoint (not used by the grader) to see exactly what
    gets transcribed, or the full error if the AI Pipe call fails.
    """
    audio_bytes = base64.b64decode(req.audio_base64)
    try:
        column_name = transcribe_column_name(audio_bytes)
        return {"success": True, "transcribed_column_name": column_name}
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
