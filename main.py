"""
Korean Audio Dataset API
-------------------------
Receives: {"audio_id": "q0", "audio_base64": "..."}
Returns: a JSON object with rows/columns/mean/std/... statistics.

Conservative strategy:
- Transcribe the audio with AI Pipe / Gemini.
- Parse row count, column name(s), and explicitly spoken stats.
- Do not invent any other fields.
"""

import base64
import logging
import os
import re
import traceback
from typing import Any

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

KOR_DIGITS = {
    "영": 0, "공": 0,
    "일": 1, "한": 1,
    "이": 2, "둘": 2,
    "삼": 3, "셋": 3,
    "사": 4, "넷": 4,
    "오": 5, "다섯": 5,
    "육": 6, "여섯": 6,
    "칠": 7, "일곱": 7,
    "팔": 8, "여덟": 8,
    "구": 9, "아홉": 9,
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
    return re.sub(r"\s+", " ", text.strip().replace("\n", " "))


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

    total = 0
    current = 0
    matched = False
    units = {"십": 10, "백": 100, "천": 1000, "만": 10000}

    i = 0
    while i < len(text):
        ch = text[i]
        if ch in KOR_DIGITS:
            current = KOR_DIGITS[ch]
            matched = True
            i += 1
            if i < len(text) and text[i] in units:
                unit = units[text[i]]
                if current == 0:
                    current = 1
                current *= unit
                total += current
                current = 0
                i += 1
        elif ch in units:
            unit = units[ch]
            if current == 0:
                current = 1
            current *= unit
            total += current
            current = 0
            matched = True
            i += 1
        else:
            i += 1

    total += current
    return total if matched else None


def is_valid_column_token(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    if re.fullmatch(r"\d+", token):
        return False
    return bool(re.search(r"[가-힣A-Za-z]", token))


def extract_columns(text: str) -> list[str]:
    text = normalize_text(text)

    cols = []

    for m in re.finditer(r"([가-힣A-Za-z][가-힣A-Za-z0-9_]*)", text):
        token = m.group(1)
        if is_valid_column_token(token):
            cols.append(token)

    # Prefer explicit forms like "점수1과 점수2"
    joined = []
    for part in re.split(r"[,\s]+", text):
        part = part.strip(".,!?")
        if is_valid_column_token(part):
            joined.append(part)

    for c in joined:
        if c not in cols:
            cols.append(c)

    # Common cleanup for phrases like "점수의"
    cleaned = []
    for c in cols:
        c = re.sub(r"(의|은|는)$", "", c)
        if is_valid_column_token(c) and c not in cleaned:
            cleaned.append(c)

    return cleaned


def extract_row_count(text: str) -> int | None:
    text = normalize_text(text)
    m = re.search(r"([가-힣0-9]+?)\s*개의\s*행", text)
    if m:
        return hangul_number_to_int(m.group(1))
    m = re.search(r"([가-힣0-9]+?)\s*행", text)
    if m:
        return hangul_number_to_int(m.group(1))
    m = re.search(r"(\d+)\s*행", text)
    if m:
        return int(m.group(1))
    return None


def extract_stat_value(text: str, korean_stat_name: str) -> Any | None:
    text = normalize_text(text)
    m = re.search(rf"{korean_stat_name}은\s*([^\s]+)", text)
    if not m:
        return None

    raw = clean_numeric_phrase(m.group(1))
    num = hangul_number_to_int(raw)
    if num is not None:
        return num

    try:
        return float(raw) if "." in raw else int(raw)
    except Exception:
        return raw


def build_response_from_transcript(transcript: str) -> dict:
    text = normalize_text(transcript)
    columns = extract_columns(text)
    rows = extract_row_count(text)

    if not columns:
        return EMPTY_RESULT

    response = {
        "rows": rows if rows is not None else 0,
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

    for col in columns:
        mode_value = extract_stat_value(text, "최빈값")
        if mode_value is not None and len(columns) == 1:
            response["mode"] = {col: mode_value}

    return response


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
    return build_response_from_transcript(transcript)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Korean Audio Dataset API is running",
        "version": "2026-q6-column-parser-v1",
    }


@app.post("/debug_transcribe")
def debug_transcribe(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        transcript = transcribe_full_audio(audio_bytes)
        return {
            "success": True,
            "transcript": transcript,
            "response_preview": build_response_from_transcript(transcript),
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