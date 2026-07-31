"""
live_demo.py
------------
Runs the REAL chatbot_engine.handle_message() pipeline against a set
of example messages, so you can see routing, RAG retrieval, emotion
inference, and crisis detection actually working end-to-end.

llm.generate_response is mocked (returns a placeholder string) because
this sandbox has no network access to reach a real Ollama server.
Everything else — intent detection, crisis short-circuiting, RAG
context retrieval (mocked chunks), emotion inference, prompt
construction — is the real code path, not simulated.

Run for real on your machine (with Ollama running) to see actual
generated replies:
    python live_demo.py
"""
from unittest.mock import patch

import chatbot_engine


EXAMPLES = [
    "explain recursion",
    "make me a study plan for finals, 3 days",
    "I'm so overwhelmed with my classes this week",
    "I'm anxious about my internship interview tomorrow",
    "I feel really lonely and isolated lately",
    "죽고 싶어",  # crisis-risk Korean message
]


def run():
    with patch("chatbot_engine.llm.generate_response", return_value="[MOCKED LLM REPLY — swap in real Ollama on your machine to see actual text]"), \
         patch("chatbot_engine.rag.retrieve", return_value=["(mocked KB chunk about coping strategies)"]):

        for msg in EXAMPLES:
            result = chatbot_engine.handle_message(msg)
            print("=" * 70)
            print(f"USER:     {msg}")
            print(f"INTENT:   {result['intent']}")
            if result["emotion"]:
                print(f"EMOTION:  {result['emotion']}")
            print(f"REPLY:    {result['response']}")
            if result["error"]:
                print(f"ERROR:    {result['error']}")
        print("=" * 70)


if __name__ == "__main__":
    run()