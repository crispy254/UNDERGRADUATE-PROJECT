"""
============================================================
AI Academic Success Companion - Chatbot Engine
============================================================

Purpose
-------
This module is the main controller for the AI chatbot.

Responsibilities
----------------
1. Check for crisis-risk language and short-circuit to a fixed
   resource response if found (never LLM-generated).
2. Detect the user's intent.
3. For counseling intents: infer emotional state, retrieve relevant
   knowledge-base context via RAG, and build an empathetic,
   context-grounded prompt.
4. For academic intents (gpa/study_plan/quiz/explanation/general):
   build the appropriate prompt as before.
5. Send the prompt to the LLM (Llama 3.2 via Ollama).
6. Return a structured, state-aware response.

This module is stateless. Sessions, persistence, and the database are
handled by other layers (see services/chatbot_service.py).

Author: Member 1 - AI & Chatbot
"""

import re
from typing import Optional, List, Dict

import llm
import prompt_builder
import inference
import rag


# Intent Detection

INTENT_PATTERNS = {
    # --- Counseling intents (stress / career / emotional support) ---
    "career_anxiety": [
        r"job", r"career", r"internship", r"resume", r"\bcv\b",
        r"interview", r"진로", r"취업", r"면접", r"what should i do after (i )?graduat"
    ],
    "stress_academic": [
        r"stressed.*(exam|class|grade|assignment|deadline)",
        r"(exam|class|grade|assignment|deadline).*stress",
        r"burnt? out", r"burned out", r"overwhelm", r"can'?t keep up",
        r"학업\s*스트레스", r"시험\s*스트레스"
    ],
    "stress_interpersonal": [
        r"lonely", r"isolated", r"fight with (my )?(friend|roommate|family)",
        r"relationship trouble", r"인간관계", r"외로", r"고립"
    ],
    "emotional_checkin": [
        r"feeling (down|low|anxious|sad|overwhelmed)",
        r"not (feeling|doing) (ok|okay|well|good)",
        r"having a hard time", r"i'?m struggling",
        r"기분이\s*안좋", r"힘들어"
    ],

    # --- Academic intents (existing) ---
    "gpa": [
        r"\bgpa\b",
        r"grade point",
        r"improve.*grade",
        r"raise my grade",
        r"academic performance"
    ],
    "study_plan": [
        r"study plan",
        r"revision plan",
        r"study schedule",
        r"prepare for",
        r"plan.*exam",
        r"how should i study"
    ],
    "quiz": [
        r"\bquiz\b",
        r"practice question",
        r"mock exam",
        r"mock test",
        r"test me",
        r"give me questions"
    ],
    "explanation": [
        r"^explain",
        r"define",
        r"what is",
        r"what are",
        r"how does",
        r"can you explain"
    ]
}

# Intents routed through the RAG + emotion-aware counseling pipeline
# instead of the plain academic prompt builders below.
COUNSELING_INTENTS = {
    "career_anxiety",
    "stress_academic",
    "stress_interpersonal",
    "emotional_checkin",
}


def detect_intent(message: str) -> str:
    """
    Detect the user's intent using lightweight rule-based matching.

    Returns one of the keys in INTENT_PATTERNS, or "general" if
    nothing matches.
    """
    text = message.lower()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return intent

    return "general"


MAX_TOKENS_BY_INTENT = {
    "general": 180,
    "gpa": 160,
    "explanation": 200,
    "study_plan": 300,
    "quiz": 380,
    "career_anxiety": 220,
    "stress_academic": 220,
    "stress_interpersonal": 220,
    "emotional_checkin": 200,
}


PROMPT_BUILDERS = {
    "gpa": prompt_builder.build_gpa_prompt,
    "study_plan": prompt_builder.build_study_plan_prompt,
    "quiz": prompt_builder.build_quiz_prompt,
    "explanation": prompt_builder.build_explanation_prompt,
}


# Main Chatbot Entry Point

def handle_message(
    user_message: str,
    history: Optional[List[Dict]] = None,
    trend_info: Optional[Dict] = None,
    language: str = "en",
) -> Dict:
    """
    Main chatbot entry point.

    Parameters
    ----------
    user_message : str
        Student's message.

    history : list
        Previous conversation turns. Used for the "general" intent and
        all counseling intents. Specialized academic prompts (GPA,
        quiz, etc.) stay single-turn and focused for reliability.

    trend_info : dict, optional
        Multi-session pattern summary from trend_service.get_recent_trend(),
        e.g. {"direction": "worsening", "dominant_stress_type": "academic", ...}.
        Only used for counseling intents; ignored otherwise. This
        module stays stateless -- the caller (chatbot_service.py) is
        responsible for computing this from the DB and passing it in.

    language : str
        "en" or "ko". Controls both the fixed crisis response text and
        an instruction appended to every LLM system prompt to reply in
        that language. Defaults to "en" so existing callers that don't
        pass this keep working unchanged.

    Returns
    -------
    dict
        {
            "intent": "...",
            "response": "...",
            "emotion": {...} | None,   # only populated for counseling intents
            "error": None
        }
    """
    if not user_message.strip():
        return {
            "intent": None,
            "response": "",
            "emotion": None,
            "error": "Message cannot be empty."
        }

    # Crisis check runs before intent detection and before any LLM
    # call. This is intentional - a fixed, human-reviewed response is
    # far safer here than anything model-generated.
    if inference.check_crisis_risk(user_message):
        return {
            "intent": "crisis",
            "response": prompt_builder.get_crisis_response(language),
            "emotion": {"emotion": "crisis", "stress_type": "crisis", "risk": True},
            "error": None
        }

    intent = detect_intent(user_message)
    emotion_info = None

    if intent in COUNSELING_INTENTS:
        emotion_info = inference.infer_emotional_state(user_message, history)
        rag_category = "career_employment" if intent == "career_anxiety" else None
        retrieved_chunks = rag.retrieve(user_message, k=3, category=rag_category)
        messages = prompt_builder.build_counseling_prompt(
            user_message, history, retrieved_chunks, emotion_info, trend_info, language
        )
    elif intent == "general":
        messages = prompt_builder.build_messages(user_message, history, language)
    else:
        messages = PROMPT_BUILDERS[intent](user_message, history, language)

    try:
        token_cap = MAX_TOKENS_BY_INTENT.get(intent, 200)
        response = llm.generate_response(messages, max_tokens=token_cap)

        return {
            "intent": intent,
            "response": response,
            "emotion": emotion_info,
            "error": None
        }

    except llm.LLMConnectionError as e:
        return {
            "intent": intent,
            "response": "",
            "emotion": emotion_info,
            "error": str(e)
        }


# Streaming variant (for web UIs - shows text as it's generated
# instead of waiting for the full reply)

def stream_message(user_message: str, history: Optional[List[Dict]] = None):
    """
    Same behavior as handle_message, but yields events instead of
    returning one dict. Used by app.py's /chat streaming endpoint.

    Yields dicts of one of these shapes:
        {"type": "intent", "intent": "gpa"}
        {"type": "emotion", "emotion": {...}}     (counseling intents only)
        {"type": "chunk", "text": "..."}          (repeated, in order)
        {"type": "error", "error": "..."}

    Crisis-risk messages skip streaming entirely and yield the fixed
    response as a single chunk, since it's not LLM-generated.
    """
    if not user_message.strip():
        yield {"type": "error", "error": "Message cannot be empty."}
        return

    if inference.check_crisis_risk(user_message):
        yield {"type": "intent", "intent": "crisis"}
        yield {"type": "chunk", "text": prompt_builder.CRISIS_RESPONSE}
        return

    intent = detect_intent(user_message)
    yield {"type": "intent", "intent": intent}

    if intent in COUNSELING_INTENTS:
        emotion_info = inference.infer_emotional_state(user_message, history)
        yield {"type": "emotion", "emotion": emotion_info}
        rag_category = "career_employment" if intent == "career_anxiety" else None
        retrieved_chunks = rag.retrieve(user_message, k=3, category=rag_category)
        messages = prompt_builder.build_counseling_prompt(
            user_message, history, retrieved_chunks, emotion_info
        )
    elif intent == "general":
        messages = prompt_builder.build_messages(user_message, history)
    else:
        messages = PROMPT_BUILDERS[intent](user_message, history)

    token_cap = MAX_TOKENS_BY_INTENT.get(intent, 200)

    try:
        for chunk in llm.stream_response(messages, max_tokens=token_cap):
            yield {"type": "chunk", "text": chunk}
    except llm.LLMConnectionError as e:
        yield {"type": "error", "error": str(e)}


# CLI Test Loop

if __name__ == "__main__":
    print("=" * 60)
    print("AI Academic Success Companion (Nova)")
    print("Type 'exit' to quit.")
    print("=" * 60)

    history = []

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ("exit", "quit"):
            print("\nGoodbye!")
            break

        print("\n(thinking... this can take up to a minute on a laptop CPU)")

        result = handle_message(question, history)

        if result["error"]:
            print(f"\nError: {result['error']}")
            continue

        print(f"\nIntent : {result['intent']}")
        if result["emotion"]:
            print(f"Emotion: {result['emotion']}")
        print(f"\nAssistant:\n{result['response']}")

        # Append to CLI history so subsequent calls retain context
        history.append({
            "role": "user",
            "content": question
        })
        history.append({
            "role": "assistant",
            "content": result["response"]
        })