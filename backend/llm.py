"""
llm.py
------
Thin wrapper around a locally running Ollama server hosting Llama 3.2.

Setup (one-time, on whichever machine runs the backend):
    1. Install Ollama: https://ollama.com/download
    2. Pull the model:      ollama pull nova-counseling
    3. Start the server:    ollama serve   (usually auto-starts after install)

Ollama exposes a local REST API at http://localhost:11434 by default —
no API key needed. This file is the ONLY place that talks to Ollama.
"""

import json

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "nova-counseling"        
                               
DEFAULT_TIMEOUT = 90           
KEEP_ALIVE = "30m"             
                                 

class LLMConnectionError(Exception):
    """Raised when Ollama isn't reachable (not installed / not running)."""
    pass


def generate_response(messages: list[dict], temperature: float = 0.4,
                       max_tokens: int = 250) -> str:
    """
    Send a chat-style message list to Llama 3.2 via Ollama and return the
    model's text reply.

    Args:
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        temperature: lower = more focused/deterministic (good for academic answers)
        max_tokens: hard cap on response length (num_predict). Acts as a
            backstop against rambling even if the prompt is loosely
            followed — most intents should finish well under this.

    Returns:
        The assistant's reply as a plain string.

    Raises:
        LLMConnectionError: if Ollama is not running / not reachable.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise LLMConnectionError(
            "Could not reach Ollama at localhost:11434. "
            "Make sure Ollama is installed and running (`ollama serve`), "
            "and that you've pulled the model (`ollama pull llama3.2`)."
        )
    except requests.exceptions.Timeout:
        raise LLMConnectionError(
            "Ollama took too long to respond. The model may still be "
            "loading into memory on first request — try again in a moment."
        )
    except requests.exceptions.HTTPError as e:
        raise LLMConnectionError(f"Ollama returned an error: {e}")

    data = response.json()
    # Ollama's /api/chat returns: {"message": {"role": "assistant", "content": "..."}, ...}
    return data.get("message", {}).get("content", "").strip()


def stream_response(messages: list[dict], temperature: float = 0.4,
                     max_tokens: int = 350):
    """
    Same as generate_response, but yields text chunks as they're
    generated instead of waiting for the full reply. This is what
    makes the UI feel fast — the student sees words appear immediately
    instead of staring at "Thinking..." for several seconds.

    Yields:
        str chunks of the response, in order. Concatenate them to get
        the full text.

    Raises:
        LLMConnectionError: if Ollama is not running / not reachable.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=DEFAULT_TIMEOUT,
                                  stream=True)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise LLMConnectionError(
            "Could not reach Ollama at localhost:11434. "
            "Make sure Ollama is installed and running (`ollama serve`), "
            "and that you've pulled the model (`ollama pull llama3.2`)."
        )
    except requests.exceptions.Timeout:
        raise LLMConnectionError(
            "Ollama took too long to respond. The model may still be "
            "loading into memory on first request — try again in a moment."
        )
    except requests.exceptions.HTTPError as e:
        raise LLMConnectionError(f"Ollama returned an error: {e}")

    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        content = data.get("message", {}).get("content", "")
        if content:
            yield content

        if data.get("done"):
            break


def check_ollama_status() -> bool:
    """Quick health check — useful for a /health endpoint or startup check."""
    try:
        r = requests.get("http://localhost:11434", timeout=5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


if __name__ == "__main__":
    # Quick manual test: python llm.py
    print("Checking Ollama status...", "OK" if check_ollama_status() else "NOT RUNNING")
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ]
    try:
        print(generate_response(test_messages))
    except LLMConnectionError as e:
        print(f"Error: {e}")