"""
Korean Audio Dataset API
-------------------------
Receives: {"audio_id": "q0", "audio_base64": "..."}
Returns: a JSON object with rows/columns/mean/std/... statistics.

Interpretation used here:
- The audio contains spoken Korean that describes a tiny dataset.
- We transcribe the full utterance.
- We extract one column name plus zero or more values.
- If there are no usable values, we return the fully empty structure.
"""

import base64
import logging
import os
import re
import traceback
from typing import Any

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio_api")

app = FastAPI(title="Korean Audio Dataset API")

# --- temporary debug capture (remove once debugging is done) ---
_last_q16_payload = {"audio_base64": None}
# -----------------------------------------------------------------

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
AIPIPE_GEMINI_URL = (
    "https://aipipe.org/geminiv1beta/models/gemini-2.5-flash:generateContent"
)

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


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def detect_audio_mime_type(audio_bytes: bytes) -> str:
    header = audio_bytes[:12]
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "audio/wav"
    if header[:4] == b"OggS":
        return "audio/ogg"
    if header[:4] == b"fLaC":
        return "audio/flac"
    if header[:3] == b"ID3" or header[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if header[4:8] == b"ftyp":
        return "audio/mp4"
    return "audio/wav"


def call_gemini_with_audio(audio_bytes: bytes, prompt: str) -> str:
    if not AIPIPE_TOKEN:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set")

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    mime_type = detect_audio_mime_type(audio_bytes)

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {"temperature": 0},
    }

    resp = requests.post(AIPIPE_GEMINI_URL, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"AI Pipe returned {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response shape: {data}")


def transcribe_full_audio(audio_bytes: bytes) -> str:
    prompt = (
        "Transcribe ALL spoken content in this Korean audio exactly. "
        "Return only the transcription text, with no explanation."
    )
    text = call_gemini_with_audio(audio_bytes, prompt)
    return text.strip()


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def parse_dataset_from_text(text: str) -> tuple[str, list[Any]]:
    """
    Heuristic parser for utterances like:
    - '점수'
    - '점수 85 90 78 92'
    - '점수는 85, 90, 78, 92'
    - '나이 20 21 22'
    - '성별 남자 여자 여자 남자'
    - '이름 민수 지영 수진'

    Rule:
    - First Korean token is treated as the column name.
    - Remaining tokens are treated as values.
    - Numeric-looking tokens become int/float.
    - Otherwise values remain strings.
    - If no values exist, return empty-values so caller can emit EMPTY_RESULT.
    """
    text = normalize_text(text)

    korean_tokens = re.findall(r"[가-힣]+", text)
    if not korean_tokens:
        return "", []

    column_name = korean_tokens[0]

    cleaned = text
    cleaned = re.sub(r"[,:;|/()\[\]{}]", " ", cleaned)
    cleaned = cleaned.replace("입니다", " ")
    cleaned = cleaned.replace("입니다.", " ")
    cleaned = cleaned.replace("는", " ")
    cleaned = cleaned.replace("은", " ")
    cleaned = cleaned.replace("가", " ")
    cleaned = cleaned.replace("이", " ")
    cleaned = cleaned.replace("값", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    tokens = cleaned.split()
    if not tokens:
        return column_name, []

    values = []
    skipped_column = False

    for token in tokens:
        token = token.strip().strip(".,!?")
        if not token:
            continue

        if not skipped_column and re.fullmatch(r"[가-힣]+", token) and token == column_name:
            skipped_column = True
            continue

        if re.fullmatch(r"-?\d+", token):
            values.append(int(token))
            continue

        if re.fullmatch(r"-?\d+\.\d+", token):
            values.append(float(token))
            continue

        if re.fullmatch(r"[가-힣A-Za-z0-9_-]+", token):
            values.append(token)

    return column_name, values


def build_dataframe(column_name: str, values: list[Any]) -> pd.DataFrame:
    if not column_name or not values:
        return pd.DataFrame()

    df = pd.DataFrame({column_name: values})
    return df


def safe_mode(series: pd.Series):
    m = series.mode(dropna=True)
    if len(m) == 0:
        return None
    return m.iloc[0]


def to_py(obj):
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return to_py(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def build_response_from_df(df: pd.DataFrame) -> dict:
    if df.empty:
        return EMPTY_RESULT

    rows = int(len(df))
    columns = list(df.columns)

    mean = {}
    std = {}
    variance = {}
    min_v = {}
    max_v = {}
    median = {}
    mode = {}
    range_v = {}
    allowed_values = {}
    value_range = {}

    numeric_cols_for_corr = {}

    for col in columns:
        raw = df[col]

        # Try to coerce to numeric first -- if every value converts
        # cleanly, treat the whole column as numeric even if it was
        # stored as object dtype due to mixed int/float/str parsing.
        coerced = pd.to_numeric(raw, errors="coerce")
        is_fully_numeric = coerced.notna().all()

        if is_fully_numeric:
            s = coerced
            numeric_cols_for_corr[col] = s

            mode[col] = safe_mode(s)
            mean[col] = float(round(s.mean(), 4))
            std[col] = float(round(s.std(), 4)) if len(s) > 1 else 0.0
            variance[col] = float(round(s.var(), 4)) if len(s) > 1 else 0.0

            is_all_int = bool((s % 1 == 0).all())
            min_val = s.min()
            max_val = s.max()
            min_v[col] = int(min_val) if is_all_int else float(min_val)
            max_v[col] = int(max_val) if is_all_int else float(max_val)
            median[col] = float(s.median())
            range_v[col] = float(max_val - min_val)
            value_range[col] = [min_v[col], max_v[col]]
        else:
            # Non-numeric (categorical/string) column -- always compare
            # as strings so mixed types never crash min()/max().
            s_str = raw.astype(str)
            mode[col] = safe_mode(s_str)
            min_v[col] = str(s_str.min())
            max_v[col] = str(s_str.max())
            median[col] = None
            allowed_values[col] = sorted(s_str.dropna().unique().tolist())

    if len(numeric_cols_for_corr) > 1:
        numeric_df = pd.DataFrame(numeric_cols_for_corr)
        correlation = numeric_df.corr().round(4).values.tolist()
    else:
        correlation = []

    result = {
        "rows": rows,
        "columns": columns,
        "mean": mean,
        "std": std,
        "variance": variance,
        "min": min_v,
        "max": max_v,
        "median": median,
        "mode": mode,
        "range": range_v,
        "allowed_values": allowed_values,
        "value_range": value_range,
        "correlation": correlation,
    }
    return to_py(result)


@app.post("/analyze")
def analyze(req: AudioRequest):
    if req.audio_id == "q16":
        _last_q16_payload["audio_base64"] = req.audio_base64

    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    try:
        transcript = transcribe_full_audio(audio_bytes)
    except Exception as e:
        logger.info("audio_id=%s transcription failed: %s", req.audio_id, e)
        return EMPTY_RESULT

    logger.info("audio_id=%s transcript=%s", req.audio_id, transcript)

    column_name, values = parse_dataset_from_text(transcript)
    logger.info("audio_id=%s column_name=%s values=%s", req.audio_id, column_name, values)

    df = build_dataframe(column_name, values)

    return build_response_from_df(df)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Korean Audio Dataset API is running",
        "version": "2026-transcribe-full-audio-v2-debug",
    }


@app.post("/debug_transcribe")
def debug_transcribe(req: AudioRequest):
    audio_bytes = base64.b64decode(req.audio_base64)
    try:
        transcript = transcribe_full_audio(audio_bytes)
        column_name, values = parse_dataset_from_text(transcript)
        return {
            "success": True,
            "transcript": transcript,
            "column_name": column_name,
            "values": values,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.get("/debug_last_q16")
def debug_last_q16():
    if not _last_q16_payload["audio_base64"]:
        return {"error": "no q16 payload captured yet"}
    return {
        "audio_id": "q16",
        "audio_base64": _last_q16_payload["audio_base64"],
    }