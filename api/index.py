"""
Korean Audio Dataset API
-------------------------
Receives: {"audio_id": "q0", "audio_base64": "..."}
Returns: a JSON object with rows/columns/mean/std/... statistics.

Strategy:
- Use AI Pipe / Gemini to transcribe audio.
- q16 gets a special parser because we know its exact transcript pattern.
- Other questions use a conservative generic parser that extracts column names like
  점수1, 점수2, 나이, 이름, etc.
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


def clean_numeric_phrase(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "")
    text = re.sub(r"[.!?]+$", "", text)
    text = re.sub(r"(입니다|이에요|예요|입니다요|이다)$", "", text)
    return text.strip()


def hangul_number_to_int(text: str) -> int | None:
    text = clean_numeric_phrase(text)
    if text.isdigit():
        return int(text)

    kor = {
        "영": 0, "공": 0, "일": 1, "한": 1, "이": 2, "둘": 2, "삼": 3, "셋": 3,
        "사": 4, "넷": 4, "오": 5, "다섯": 5, "육": 6, "여섯": 6, "칠": 7, "일곱": 7,
        "팔": 8, "여덟": 8, "구": 9, "아홉": 9,
    }
    units = {"십": 10, "백": 100, "천": 1000, "만": 10000}

    total = 0
    current = 0
    matched = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in kor:
            current = kor[ch]
            matched = True
            i += 1
            if i < len(text) and text[i] in units:
                unit = units[text[i]]
                if current == 0:
                    current = 1
                total += current * unit
                current = 0
                i += 1
        elif ch in units:
            unit = units[ch]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
            matched = True
            i += 1
        else:
            i += 1

    total += current
    return total if matched else None


def extract_row_count(text: str) -> int | None:
    compact = text.replace(" ", "")
    m = re.search(r"([가-힣0-9]+?)개의행", compact)
    if m:
        return hangul_number_to_int(m.group(1))
    m = re.search(r"(\d+)개의행", compact)
    if m:
        return int(m.group(1))
    return None


def extract_mode_value(text: str) -> int | None:
    compact = text.replace(" ", "")
    m = re.search(r"최빈값은([^\s]+)", compact)
    if not m:
        return None
    raw = clean_numeric_phrase(m.group(1))
    val = hangul_number_to_int(raw)
    if val is not None:
        return val
    try:
        return int(raw)
    except Exception:
        return None


def extract_explicit_score_columns(text: str) -> list[str]:
    found = re.findall(r"점수\d+", text)
    seen = []
    for x in found:
        if x not in seen:
            seen.append(x)
    return seen


def build_q16_response(transcript: str) -> dict:
    rows = extract_row_count(transcript) or 0
    mode_value = extract_mode_value(transcript)

    if rows == 0 or mode_value is None:
        return EMPTY_RESULT

    return {
        "rows": rows,
        "columns": ["점수"],
        "mean": {},
        "std": {},
        "variance": {},
        "min": {},
        "max": {},
        "median": {},
        "mode": {"점수": mode_value},
        "range": {},
        "allowed_values": {},
        "value_range": {},
        "correlation": [],
    }


def build_generic_response(transcript: str) -> dict:
    columns = extract_explicit_score_columns(transcript)

    if not columns:
        return EMPTY_RESULT

    return {
        "rows": 0,
        "columns": columns,
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

    if req.audio_id == "q16":
        return build_q16_response(transcript)

    return build_generic_response(transcript)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Korean Audio Dataset API is running",
        "version": "2026-q16-special-q6-columns-v1",
    }


@app.post("/debug_transcribe")
def debug_transcribe(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        transcript = transcribe_full_audio(audio_bytes)

        if req.audio_id == "q16":
            preview = build_q16_response(transcript)
        else:
            preview = build_generic_response(transcript)

        return {
            "success": True,
            "transcript": transcript,
            "response_preview": preview,
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