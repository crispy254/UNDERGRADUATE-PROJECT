"""
voice/tts.py
------------
Text-to-speech using pyttsx3, which drives your OS's built-in speech
engine - on Windows, that's SAPI5, already installed, no model
download needed. This is the "works immediately, zero setup" choice -
voice quality is the standard Windows system voice (functional, not
especially natural-sounding).

Upgrade path later, if the built-in voice sounds too robotic for your
demo: swap this file's implementation for Piper (a small local neural
TTS engine) without changing the function signature below - nothing
else in the app needs to know which engine is behind synthesize_speech().
That's a same-day swap when you have time, not urgent now.
"""
import os
import tempfile
import time


class TTSError(Exception):
    """Raised when speech synthesis produced no usable audio."""
    pass


def get_voice_index_for_language(language: str) -> tuple[int, bool]:
    """
    Picks a voice_index matching the requested language, searching
    installed Windows voices by name/id for language hints (e.g.
    "Korean", "KO-KR"). Returns (voice_index, found) - found=False
    means no matching voice was located, and index 0 (the default
    voice) is returned as a fallback - it will still "speak" Korean
    text, but likely mispronounce it using English phonetics, since
    it's not actually a Korean voice engine.

    Whether a Korean voice is available depends entirely on what's
    installed on this specific Windows machine (Settings > Time & Language
    > Speech, or installing a Korean language pack) - this isn't
    something the code can install for you.
    """
    if language != "ko":
        return 0, True

    voices = list_available_voices()
    for v in voices:
        name_and_id = f"{v['name']} {v['id']}".lower()
        if "korean" in name_and_id or "ko-kr" in name_and_id or "ko_kr" in name_and_id:
            return v["index"], True
    return 0, False


def synthesize_speech(text: str, voice_index: int = 0, rate: int = 175) -> bytes:
    """
    Converts text to speech, returns WAV audio as raw bytes.

    `voice_index` picks among whatever voices Windows has installed
    (0 is usually the default). `rate` is words-per-minute, pyttsx3's
    default (~200) can sound rushed for a supportive-tone bot - 175 is
    a slightly calmer default.

    A new engine instance is created per call rather than reused,
    since pyttsx3's engine isn't safely reusable across concurrent
    requests (it runs its own event loop internally) - FastAPI can
    serve requests concurrently, so this avoids two simultaneous voice
    requests corrupting each other's audio.

    Windows-specific timing quirk: pyttsx3's SAPI5 driver can return
    from runAndWait() before the WAV file is actually fully flushed to
    disk. Reading immediately afterward sometimes gets a 0-byte or
    truncated file (silent audio, no error). This retries with a short
    wait if the file looks too small, and raises TTSError instead of
    silently returning near-empty audio if it never materializes.
    """
    import pyttsx3

    engine = pyttsx3.init()
    tmp_path = None
    try:
        voices = engine.getProperty("voices")
        if voices and 0 <= voice_index < len(voices):
            engine.setProperty("voice", voices[voice_index].id)
        engine.setProperty("rate", rate)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        # A valid WAV header alone is 44 bytes; anything smaller than a
        # few hundred bytes for real speech means the file didn't
        # finish writing yet. Poll rather than trusting runAndWait()'s
        # return timing on Windows. Longer replies take longer to
        # render AND longer to flush to disk, so this window scales
        # with text length instead of using one fixed short timeout
        # that only worked for short test phrases.
        max_wait_seconds = max(3.0, len(text.split()) * 0.15)
        elapsed = 0.0
        interval = 0.15
        while elapsed < max_wait_seconds:
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 200:
                break
            time.sleep(interval)
            elapsed += interval

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) <= 200:
            raise TTSError(
                f"pyttsx3 produced no usable audio for the given text "
                f"(file size: {os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0} bytes, "
                f"waited {max_wait_seconds:.1f}s for {len(text.split())} words). "
                "This is a known Windows SAPI5 timing issue, not a code bug in the request itself - "
                "try running voice/tts.py's list_available_voices() to confirm a voice is actually "
                "installed and selectable."
            )

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        return audio_bytes
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        engine.stop()


def list_available_voices() -> list[dict]:
    """
    Utility for picking a voice_index - prints/returns whatever voices
    are installed on this machine (varies by OS/language packs). Run
    this once locally to see your options:
        python -c "from voice.tts import list_available_voices; print(list_available_voices())"
    """
    import pyttsx3

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    result = [{"index": i, "id": v.id, "name": v.name} for i, v in enumerate(voices)]
    engine.stop()
    return result