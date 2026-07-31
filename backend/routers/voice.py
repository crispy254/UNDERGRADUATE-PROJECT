"""
routers/voice.py
-----------------
Voice endpoints: speak to Nova, get speech back.

Deliberately calls chatbot_engine.handle_message() directly rather
than going through services/chatbot_service.py (which reads/writes
ChatHistory + EmotionLog to the database). I wasn't sure whether your
project is currently on the anonymous session_id version or the
original auth/user_id version - chatbot_engine.py is stateless either
way, so this endpoint works standalone regardless, without guessing at
which persistence layer is currently wired up.

Practical effect: a voice conversation right now does NOT get saved to
ChatHistory/EmotionLog the way a typed message does, and doesn't use
conversation history for context (each voice message is independent).
That's fine for a working demo. Once you confirm which auth/session
setup is currently live in app.py, this can be upgraded to persist and
use history the same way routers/chatbot.py does - flag it if you want
that next.
"""
import base64

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import chatbot_engine
from voice.stt import transcribe_audio
from voice.tts import synthesize_speech, get_voice_index_for_language, TTSError

router = APIRouter(
    prefix="/voice",
    tags=["Voice"]
)


class SpeakRequest(BaseModel):
    text: str


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    STT only - upload an audio file (webm/ogg/wav), get back the
    transcribed text. Useful for testing the mic/upload path on its
    own before wiring up the full round trip.
    """
    audio_bytes = await audio.read()
    suffix = "." + audio.filename.rsplit(".", 1)[-1] if audio.filename and "." in audio.filename else ".webm"
    text = transcribe_audio(audio_bytes, suffix=suffix)
    return {"text": text}


@router.post("/speak")
def speak(request: SpeakRequest):
    """
    TTS only - given text, returns WAV audio bytes directly (not JSON)
    so it can be played straight from a browser <audio> element or
    fetch() response. Useful for testing voice output on its own.

    Returns a plain Response (real Content-Length header) rather than
    a chunked StreamingResponse - browsers/Swagger's audio player
    handle a known-length WAV more reliably than a chunked one.
    """
    try:
        audio_bytes = synthesize_speech(request.text)
    except TTSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=audio_bytes, media_type="audio/wav")


@router.post("/chat")
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: str = Form(default="anonymous"),
    language: str = Form(default="en"),
):
    """
    Full round trip: upload audio -> transcribe -> run through Nova ->
    speak the reply back.

    `language` - "en" or "ko". Hints Whisper's transcription, tells
    Nova which language to reply in, and picks a matching TTS voice IF
    one is installed on this machine (see get_voice_index_for_language
    in voice/tts.py - Korean playback quality depends entirely on
    whether a Korean voice is installed in Windows; falls back to the
    default English voice mispronouncing Korean text if not, and flags
    that via `tts_language_warning` below rather than failing silently).

    Returns JSON (not raw audio) so the frontend gets both the text
    (to display in the chat bubbles, same as typed messages) and the
    audio (base64-encoded, so it fits in the same JSON response rather
    than needing a second request):
        {
            "transcript": "what the student said",
            "reply": "Nova's text reply",
            "intent": "stress_academic" | "gpa" | ...,
            "audio_base64": "...",   # decode + play as audio/wav
            "tts_language_warning": "..." | null
        }

    `session_id` is accepted but not yet used to persist history (see
    module docstring) - passed through so the frontend can start
    sending it now, ready for when persistence is wired in.
    """
    voice_index, voice_found = get_voice_index_for_language(language)
    tts_language_warning = None
    if not voice_found:
        tts_language_warning = (
            "No Korean voice is installed on this machine, so replies "
            "are spoken with the default English voice - text is still "
            "accurate Korean, just mispronounced. Install a Korean "
            "voice via Windows Settings > Time & Language > Speech to fix this."
        )

    audio_bytes = await audio.read()
    suffix = "." + audio.filename.rsplit(".", 1)[-1] if audio.filename and "." in audio.filename else ".webm"
    whisper_lang = language if language in ("en", "ko") else None
    transcript = transcribe_audio(audio_bytes, suffix=suffix, language=whisper_lang)

    if not transcript.strip():
        fallback_text = "다시 한 번 말씀해 주시겠어요?" if language == "ko" else "I didn't catch that - could you try again?"
        try:
            fallback_audio = base64.b64encode(
                synthesize_speech(fallback_text, voice_index=voice_index)
            ).decode("ascii")
        except TTSError:
            fallback_audio = None
        return {
            "transcript": "",
            "reply": fallback_text,
            "intent": None,
            "emotion": None,
            "stress_type": None,
            "risk_flag": False,
            "audio_base64": fallback_audio,
            "tts_language_warning": tts_language_warning,
        }

    result = chatbot_engine.handle_message(transcript, language=language)
    reply_text = result["response"] or "Sorry, I couldn't process that right now."
    emotion_info = result.get("emotion") or {}

    reply_audio = None
    audio_error = None
    try:
        reply_audio = base64.b64encode(
            synthesize_speech(reply_text, voice_index=voice_index)
        ).decode("ascii")
    except TTSError as e:
        # Text reply still succeeds even if voice output fails - but
        # surface WHY, instead of silently degrading to text-only with
        # no explanation (that silence is what made this hard to debug
        # last time).
        audio_error = str(e)

    return {
        "transcript": transcript,
        "reply": reply_text,
        "intent": result["intent"],
        "emotion": emotion_info.get("emotion"),
        "stress_type": emotion_info.get("stress_type"),
        "risk_flag": bool(emotion_info.get("risk")),
        "audio_base64": reply_audio,
        "audio_error": audio_error,
        "tts_language_warning": tts_language_warning,
    }