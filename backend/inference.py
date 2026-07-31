"""
inference.py
-------------
Lightweight emotion / stress-type / crisis-risk inference for the
counseling chatbot.

This is a rule-based v1, deliberately kept simple and transparent so
it's easy to audit and extend. It's designed so the *interface*
(infer_emotional_state, check_crisis_risk) won't need to change if you
later swap the internals for a trained sentiment/intent classifier —
callers (chatbot_engine.py) only depend on the function signatures and
the shape of the returned dict.

check_crisis_risk() is intentionally separate from and always checked
before infer_emotional_state() in chatbot_engine.py, so a crisis-risk
message always short-circuits into a fixed, human-reviewed resource
response (see prompt_builder.CRISIS_RESPONSE) instead of going through
the LLM.
"""
import re
from typing import Dict, List, Optional

# Distress phrases that should immediately route to crisis resources
# instead of a normal LLM-generated reply. Keep this list reviewed by
# a real person (e.g. a counseling-center advisor) rather than
# expanded ad hoc — false negatives are worse than false positives
# here, so lean broad.
CRISIS_PATTERNS = [
    r"\bkill myself\b", r"\bend my life\b", r"\bend it all\b",
    r"\bsuicid\w*", r"\bwant to die\b", r"\bdon'?t want to (be alive|live)\b",
    r"\bno reason to live\b", r"\bhurt myself\b", r"\bself[- ]?harm\b",
    r"죽고\s*싶", r"자살", r"자해", r"살기\s*싫", r"사라지고\s*싶",
    r"극단적\s*선택",
]

EMOTION_PATTERNS = {
    "anxious": [r"\banxious\b", r"\banxiety\b", r"\bworried\b", r"\bnervous\b",
                r"\bpanick\w*", r"freak(ing)? out", r"racing (thoughts|mind)",
                r"불안", r"걱정"],
    "overwhelmed": [r"overwhelm\w*", r"too much", r"can'?t cope", r"burnt out",
                     r"burned out", r"drowning in", r"지쳤", r"버겁"],
    "sad": [r"\bsad\b", r"depress\w*", r"down lately", r"crying", r"\blonely\b",
            r"isolat\w*", r"우울", r"슬프", r"외로"],
    "angry": [r"\bangry\b", r"frustrat\w*", r"irritat\w*", r"화가", r"짜증"],
    "hopeful": [r"\bexcited\b", r"looking forward", r"\bhopeful\b", r"기대", r"희망"],
}

STRESS_TYPE_PATTERNS = {
    "academic": [r"\bexam\b", r"\btest\b", r"\bgrade\b", r"\bgpa\b", r"assignment", r"deadline",
                 r"\bclass(es)?\b", r"\bcourse\b", r"\bschool\b", r"\bhomework\b",
                 r"학점", r"시험", r"과제", r"수업"],
    "career": [r"\bjob\b", r"\bcareer\b", r"internship", r"resume", r"\bcv\b",
               r"interview", r"진로", r"취업", r"면접"],
    "interpersonal": [r"\bfriend\b", r"relationship", r"\bfamily\b", r"roommate",
                       r"\blonely\b", r"인간관계", r"외로"],
}


def check_crisis_risk(message: str) -> bool:
    """True if the message contains language suggesting acute risk of
    self-harm/suicide. Always check this before generating any normal
    LLM response."""
    text = message.lower()
    return any(re.search(p, text) for p in CRISIS_PATTERNS)


def _match_category(text: str, patterns: Dict[str, List[str]]) -> Optional[str]:
    for label, pats in patterns.items():
        for p in pats:
            if re.search(p, text):
                return label
    return None


def infer_emotional_state(message: str, history: Optional[List[Dict]] = None) -> Dict:
    """
    Infer emotion + stress-type for a counseling-intent message.

    Returns:
        {
            "emotion": "anxious" | "overwhelmed" | "sad" | "angry" |
                       "hopeful" | "neutral",
            "stress_type": "academic" | "career" | "interpersonal" | "general",
            "risk": bool,   # mirrors check_crisis_risk() for convenience
        }

    `history` is accepted (and reserved) for future use — e.g. trend
    detection across turns — but not currently used by the rule-based
    v1. Keeping it in the signature now avoids a breaking change later.
    """
    text = message.lower()
    emotion = _match_category(text, EMOTION_PATTERNS) or "neutral"
    stress_type = _match_category(text, STRESS_TYPE_PATTERNS) or "general"

    return {
        "emotion": emotion,
        "stress_type": stress_type,
        "risk": check_crisis_risk(message),
    }