"""
============================================================
AI Academic Success Companion - Chatbot Engine
============================================================

Purpose
-------
Main controller for the chatbot: crisis check, intent detection, RAG +
emotion inference for counseling intents, RL strategy selection for
counseling intents, prompt construction, and LLM generation.

RL integration (strategy_selector.py)
--------------------------------------
For counseling-intent messages only, after inferring emotion/stress
type, the bandit in strategy_selector.py picks ONE of four pre-written
response strategies (see prompt_builder.STRATEGIES / _STRATEGY_RULE_BLOCKS)
based on that context. The chosen strategy is returned in the result
dict as "strategy", so the caller (chatbot_service.py) can log it -
that log is both how the bandit learns (once feedback arrives) and the
raw evaluation data for the research writeup.

The bandit NEVER runs for crisis-risk messages (those short-circuit
before intent detection even happens) and never runs for academic
intents (gpa/study_plan/quiz/explanation) - it is scoped exclusively
to the counseling-support response style, by design.

Intent coverage (why this matters for RL)
-------------------------------------------
The RL bandit only ever sees a (context, action, reward) sample when a
message is classified into a COUNSELING_INTENTS category. If a
student's real phrasing doesn't match any INTENT_PATTERNS regex, the
message falls into "general" - which never touches RAG, emotion
inference, or the bandit at all, and never gets a feedback_id, so no
thumbs-up/down is even possible on that turn. That's not just a UX gap
(missing feedback buttons) - it silently starves the bandit of
training data for however students actually phrase things versus how
the regexes were written.

Two things address this:
  1. INTENT_PATTERNS below is broadened to catch common bare phrasings
     ("am stressed", "i'm stressed") in addition to cause-specific ones.
  2. A generic fallback in detect_intent(): if nothing else matches but
     the message contains clear emotional-distress language, it routes
     to "emotional_checkin" (a COUNSELING_INTENTS member) rather than
     "general". This is deliberately broad/low-precision by design -
     it's fine to occasionally misroute a mildly-worded message into
     counseling, but bad to silently drop a real distress signal into
     the generic chit-chat path with no RAG, no emotion tracking, and
     no RL learning signal.

This module is stateless. Sessions, persistence, and the database are
handled by other layers (see services/chatbot_service.py).
"""

import re
from typing import Optional, List, Dict

import llm
import prompt_builder
import inference
import rag
import strategy_selector


INTENT_PATTERNS = {
    "career_anxiety": [
        r"job", r"career", r"internship", r"resume", r"\bcv\b",
        r"interview", r"진로", r"취업", r"면접", r"what should i do after (i )?graduat"
    ],
    "stress_academic": [
        r"stressed.*(exam|class|grade|assignment|deadline)",
        r"(exam|class|grade|assignment|deadline).*stress",
        r"burnt? out", r"burned out", r"overwhelm", r"can'?t keep up",
        # Broadened: catch bare "I'm stressed" statements with no
        # explicit cause word attached, not just cause-paired phrasings.
        r"\bam stressed\b", r"\bi'?m stressed\b", r"\bso stressed\b",
        r"\breally stressed\b", r"\bfeeling stressed\b", r"\bstressed out\b",
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
    "gpa": [
        r"\bgpa\b", r"grade point", r"improve.*grade", r"raise my grade",
        r"academic performance", r"성적", r"학점", r"평점"
    ],
    "study_plan": [
        r"study plan", r"revision plan", r"study schedule", r"prepare for",
        r"plan.*exam", r"how should i study", r"공부\s*계획", r"학습\s*계획"
    ],
    "quiz": [
        r"\bquiz\b", r"practice question", r"mock exam", r"mock test",
        r"test me", r"give me questions", r"퀴즈", r"문제\s*내", r"연습\s*문제"
    ],
    "explanation": [
        r"^explain", r"define", r"what is", r"what are", r"how does",
        r"can you explain", r"설명해", r"이란\s*무엇", r"뭐야\??$"
    ]
}

COUNSELING_INTENTS = {
    "career_anxiety",
    "stress_academic",
    "stress_interpersonal",
    "emotional_checkin",
}

# Low-precision, deliberately broad fallback signal: if no specific
# intent pattern matched but the message clearly contains distress
# language, route to emotional_checkin rather than losing it to
# "general". False positives here (a mildly-worded message getting
# routed to counseling) are an acceptable trade-off against the
# alternative - a genuine distress signal getting no RAG, no emotion
# tracking, and no RL feedback opportunity at all.
GENERIC_DISTRESS_PATTERNS = [
    r"\bstress(ed)?\b", r"\banxious\b", r"\banxiety\b", r"\bworried\b",
    r"\bexhausted\b", r"\btired of\b", r"\bstruggling\b", r"\boverwhelmed\b",
    r"\bsad\b", r"\bdown\b",
    r"스트레스", r"불안", r"걱정", r"힘들",
]


def detect_intent(message: str) -> str:
    text = message.lower()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return intent

    # Nothing specific matched - check the broad distress fallback
    # before giving up and calling it "general".
    for pattern in GENERIC_DISTRESS_PATTERNS:
        if re.search(pattern, text):
            return "emotional_checkin"

    return "general"


MAX_TOKENS_BY_INTENT = {
    "general": 180, "gpa": 160, "explanation": 200, "study_plan": 300,
    "quiz": 380, "career_anxiety": 180, "stress_academic": 180,
    "stress_interpersonal": 180, "emotional_checkin": 160,
}

PROMPT_BUILDERS = {
    "gpa": prompt_builder.build_gpa_prompt,
    "study_plan": prompt_builder.build_study_plan_prompt,
    "quiz": prompt_builder.build_quiz_prompt,
    "explanation": prompt_builder.build_explanation_prompt,
}


def handle_message(
    user_message: str,
    history: Optional[List[Dict]] = None,
    trend_info: Optional[Dict] = None,
    language: str = "en",
) -> Dict:
    """
    Returns:
        {
            "intent": "...",
            "response": "...",
            "emotion": {...} | None,
            "strategy": "validate_ask" | ... | None,   # only set for counseling intents
            "error": None
        }
    """
    if not user_message.strip():
        return {"intent": None, "response": "", "emotion": None, "strategy": None,
                "error": "Message cannot be empty."}

    if inference.check_crisis_risk(user_message):
        return {
            "intent": "crisis",
            "response": prompt_builder.get_crisis_response(language),
            "emotion": {"emotion": "crisis", "stress_type": "crisis", "risk": True},
            "strategy": None,  # RL never touches crisis handling
            "error": None,
        }

    intent = detect_intent(user_message)
    emotion_info = None
    strategy = None

    if intent in COUNSELING_INTENTS:
        emotion_info = inference.infer_emotional_state(user_message, history)

        # --- RL strategy selection happens here, and only here ---
        strategy, _scores = strategy_selector.select_strategy(
            emotion_info["emotion"], emotion_info["stress_type"]
        )

        rag_category = "career_employment" if intent == "career_anxiety" else None
        retrieved_chunks = rag.retrieve(user_message, k=3, category=rag_category)
        messages = prompt_builder.build_counseling_prompt(
            user_message, history, retrieved_chunks, emotion_info,
            trend_info, language, strategy,
        )
    elif intent == "general":
        messages = prompt_builder.build_messages(user_message, history, language)
    else:
        messages = PROMPT_BUILDERS[intent](user_message, history, language)

    try:
        token_cap = MAX_TOKENS_BY_INTENT.get(intent, 200)
        response = llm.generate_response(messages, max_tokens=token_cap)
        return {
            "intent": intent, "response": response, "emotion": emotion_info,
            "strategy": strategy, "error": None,
        }
    except llm.LLMConnectionError as e:
        return {
            "intent": intent, "response": "", "emotion": emotion_info,
            "strategy": strategy, "error": str(e),
        }