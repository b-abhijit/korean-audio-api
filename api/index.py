"""
Korean Audio Dataset API
-------------------------
Receives: {"audio_id": "q0", "audio_base64": "..."}
Returns:  a JSON object with rows/columns/mean/std/... statistics
          computed from the decoded audio waveform.
"""

import base64
import io
import wave

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydub import AudioSegment
import imageio_ffmpeg

# pydub needs to know where ffmpeg lives; imageio_ffmpeg bundles a portable
# binary so we don't depend on the host machine having ffmpeg installed.
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI(title="Korean Audio Dataset API")


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def decode_wav_to_dataframe(audio_bytes: bytes):
    """Turn raw WAV bytes into a pandas DataFrame (one column per channel)."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    # map byte-width to a numpy integer type
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sampwidth, np.int16)

    samples = np.frombuffer(frames, dtype=dtype)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
        df = pd.DataFrame(samples, columns=[f"channel_{i}" for i in range(n_channels)])
    else:
        df = pd.DataFrame({"channel_0": samples})

    return df, dtype, sampwidth


def decode_compressed_to_dataframe(audio_bytes: bytes):
    """Turn MP3 (or other ffmpeg-readable compressed audio) into a DataFrame."""
    segment = AudioSegment.from_file(io.BytesIO(audio_bytes))

    sampwidth = segment.sample_width  # bytes per sample, e.g. 2 for 16-bit
    n_channels = segment.channels

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sampwidth, np.int16)

    samples = np.array(segment.get_array_of_samples(), dtype=dtype)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
        df = pd.DataFrame(samples, columns=[f"channel_{i}" for i in range(n_channels)])
    else:
        df = pd.DataFrame({"channel_0": samples})

    return df, dtype, sampwidth


def decode_audio_to_dataframe(audio_bytes: bytes):
    """Detect format by magic bytes and dispatch to the right decoder."""
    if audio_bytes[:4] == b"RIFF":
        # Standard uncompressed WAV file
        return decode_wav_to_dataframe(audio_bytes)
    else:
        # MP3 (ID3 tag or raw frame sync) or any other ffmpeg-readable format
        return decode_compressed_to_dataframe(audio_bytes)


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

    iinfo = np.iinfo(dtype)

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
        "allowed_values": {
            "dtype": str(dtype.__name__ if hasattr(dtype, "__name__") else dtype),
            "sample_width_bytes": sampwidth,
        },
        "value_range": {"min": int(iinfo.min), "max": int(iinfo.max)},
        "correlation": df.corr().round(4).values.tolist() if len(df.columns) > 1 else [[1.0]],
    }

    return to_py(result)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Korean Audio Dataset API is running"}
