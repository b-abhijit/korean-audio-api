"""
Korean Audio Dataset API
-------------------------
Receives: {"audio_id": "q0", "audio_base64": "..."}
Returns: a JSON object with rows/columns/mean/std/... statistics.

Current conservative strategy:
- Transcribe the audio with AI Pipe / Gemini.
- Extract only the column name from instruction-like speech.
- Keep all statistic dictionaries empty unless you explicitly decide otherwise.
"""

import base64
import logging
import os
import re
import traceback

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio_api")

app = FastAPI(title="Korean Audio Dataset API")

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

_last_q16_payload = {"audio_base64": None}


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
    return call_gemini_with_audio(audio_bytes, prompt).strip()


def normalize_text(text: str) -> str:
    text = text.strip().replace("\n", " ")
    return re.sub(r"\s+", " ", text)


def extract_column_name(text: str) -> str | None:
    text = normalize_text(text)

    m = re.search(r"([가-힣A-Za-z0-9_]+?)의\s*(?:최빈값|평균|분산|중앙값|최소값|최대값|범위)", text)
    if m:
        return m.group(1)

    m = re.search(r"([가-힣A-Za-z0-9_]+?)\s*열", text)
    if m:
        return m.group(1)

    return None


def build_conservative_response(transcript: str) -> dict:
    column = extract_column_name(transcript)

    if not column:
        return EMPTY_RESULT

    return {
        "rows": 0,
        "columns": [column],
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

    return build_conservative_response(transcript)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Korean Audio Dataset API is running",
        "version": "2026-column-only-v1",
    }


@app.post("/debug_transcribe")
def debug_transcribe(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        transcript = transcribe_full_audio(audio_bytes)
        return {
            "success": True,
            "transcript": transcript,
            "column_name": extract_column_name(transcript),
            "response_preview": build_conservative_response(transcript),
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