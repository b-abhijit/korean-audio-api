"""
Lightweight Korean Audio Dataset API (Whisper removed)
"""
import base64, io
import numpy as np
import pandas as pd
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Korean Audio Dataset API")

class AudioRequest(BaseModel):
    audio_id:str
    audio_base64:str

def decode_audio_to_dataframe(audio_bytes: bytes):
    samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="int16", always_2d=True)
    return pd.DataFrame(samples, columns=[f"channel_{i}" for i in range(samples.shape[1])])

def to_py(obj):
    if isinstance(obj, dict):
        return {k: to_py(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return float(obj)
    if isinstance(obj,np.ndarray): return obj.tolist()
    return obj

@app.post("/analyze")
def analyze(req: AudioRequest):
    try:
        audio = base64.b64decode(req.audio_base64)
        df = decode_audio_to_dataframe(audio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(df.columns)==1:
        df.columns=["channel_0"]

    result={
      "rows":len(df),
      "columns":list(df.columns),
      "mean":df.mean().round(4).to_dict(),
      "std":df.std().round(4).to_dict(),
      "variance":df.var().round(4).to_dict(),
      "min":df.min().to_dict(),
      "max":df.max().to_dict(),
      "median":df.median().to_dict(),
      "mode":df.mode().iloc[0].to_dict() if not df.mode().empty else {},
      "range":(df.astype("int64").max()-df.astype("int64").min()).to_dict(),
      "allowed_values":{},
      "value_range":{},
      "correlation":df.corr().round(4).values.tolist() if len(df.columns)>1 else [[1.0]]
    }
    return to_py(result)

@app.get("/")
def root():
    return {"status":"ok"}
