"""
voice/stt.py
------------
Speech-to-text using faster-whisper - a local, CPU-friendly
reimplementation of OpenAI's Whisper. Runs fully offline once the
model weights are downloaded once on first use (same "downloads once,
then local" pattern as Ollama pulling llama3.2).

Model size tradeoff (set via WHISPER_MODEL_SIZE below):
    tiny   - fastest, least accurate, fine for quick testing
    base   - good default balance for a laptop CPU (this is what's set)
    small  - noticeably more accurate, noticeably slower on CPU
On a laptop with no GPU, expect roughly real-time-ish speed for "base"
on short (a few seconds) clips - a 5 second recording might take
5-15 seconds to transcribe on CPU. That's normal for local Whisper on
CPU, not a bug.

faster-whisper decodes audio itself (via PyAV, which bundles ffmpeg in
its wheel on Windows) - so it can be handed the raw webm/ogg bytes a
browser's MediaRecorder produces directly, no manual audio conversion
needed here.
"""
import os
import tempfile
from typing import Optional

WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"          # "cuda" if you have a local NVIDIA GPU + drivers set up; "cpu" otherwise
WHISPER_COMPUTE_TYPE = "int8"   # int8 = fastest on CPU, small accuracy tradeoff vs float32

_model = None


def _get_model():
    """
    Lazy singleton - the model is loaded once (first request pays the
    load cost, ~seconds to tens of seconds depending on size), then
    reused for every subsequent transcription in this process.
    """
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
    return _model


def transcribe_audio(audio_bytes: bytes, suffix: str = ".webm", language: str | None = None) -> str:
    """
    Transcribes raw audio bytes (as uploaded from the browser - webm/
    ogg/wav, faster-whisper's decoder handles any of these) into text.

    `suffix` should roughly match the actual audio format uploaded, so
    the decoder has a hint - browsers recording via MediaRecorder
    typically produce .webm (Chrome/Edge) - the router passes this
    through based on the uploaded file's name/content type.

    `language` - pass "en" or "ko" to hint the expected language
    explicitly. Whisper can auto-detect (leave as None), but an
    explicit hint is noticeably more accurate, especially for shorter
    clips where there isn't much audio for auto-detection to work with.
    """
    model = _get_model()

    # faster-whisper's API takes a file path (or file-like object), not
    # raw bytes directly - write to a temp file, transcribe, clean up.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, beam_size=5, language=language)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass