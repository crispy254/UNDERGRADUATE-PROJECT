"""
prompt_builder.py
------------------
All prompt-engineering lives here. chatbot_engine.py decides WHAT the
user wants (intent); this file decides HOW to phrase that request to
Nova (the model).

Every build_*_prompt function accepts an optional `history` argument
for a consistent signature, even though only the general/casual and
counseling intents actually use it — GPA/study-plan/quiz/explanation
intentionally stay single-turn and focused rather than dragging old
context into a tightly-formatted answer.
"""


BASE_PERSONA = (
    "You are Nova, a warm, casual academic mentor for undergrad students. "
    "Never robotic, never a list of services, never corporate-sounding."
)

COUNSELING_PERSONA = (
    "You are Nova, a warm, empathetic peer-support companion for "
    "undergrad students dealing with stress and career worries. You are "
    "not a licensed therapist and never diagnose. You listen, validate, "
    "and offer grounded, practical support."
)

CRISIS_RESPONSE_EN = (
    "It sounds like you're going through something really heavy right now, "
    "and I want to make sure you get support beyond just this chat.\n\n"
    "In Korea, you can reach:\n"
    "- 자살예방상담전화 (Suicide Prevention Helpline): 109, available 24/7\n"
    "- 정신건강 위기상담전화 (Mental Health Crisis Line): 1577-0199, "
    "or dial 129 (Bogeonbokji Call Center)\n"
    "- 청소년전화 1388 if you're a younger student\n"
    "- Or call 112 / 119 or go to your nearest emergency room if you're "
    "in immediate danger.\n\n"
    "You don't have to go through this alone — please reach out to one "
    "of these, or someone you trust, right now."
)

CRISIS_RESPONSE_KO = (
    "지금 정말 힘든 시간을 보내고 계신 것 같아요. 이 대화만으로는 부족할 수 있으니, "
    "꼭 다른 도움도 함께 받으셨으면 해요.\n\n"
    "다음 번호로 연락하실 수 있습니다:\n"
    "- 자살예방상담전화: 109 (24시간 상담 가능)\n"
    "- 정신건강 위기상담전화: 1577-0199, 또는 보건복지콜센터 129\n"
    "- 청소년이시라면 청소년전화 1388\n"
    "- 위급한 상황이라면 112 / 119에 전화하시거나 가까운 응급실로 가주세요.\n\n"
    "혼자 견디지 않으셔도 됩니다 — 지금 이 중 한 곳이나, 믿을 수 있는 사람에게 "
    "꼭 연락해 보세요."
)

# Kept as the default name for backward compatibility with any code
# still importing CRISIS_RESPONSE directly (pre-language-support).
CRISIS_RESPONSE = CRISIS_RESPONSE_EN


def get_crisis_response(language: str = "en") -> str:
    """Fixed, human-reviewed crisis response - never LLM-generated,
    same reasoning as before, now available in both languages."""
    return CRISIS_RESPONSE_KO if language == "ko" else CRISIS_RESPONSE_EN


def _language_reminder_suffix(language: str) -> str:
    """
    Appended directly to the user's message (not just the system
    prompt) when language="ko". Models tend to follow the LANGUAGE of
    recent conversation turns more strongly than a single instruction
    stated once at the top of a long system prompt - this matters most
    when conversation history is included (counseling/general intents),
    since a run of earlier English turns can pull a reply back toward
    English even with the system-level instruction present. Placing a
    short reminder right next to the newest message (highest-attention
    position) makes the instruction much harder to drift away from.
    """
    return "\n\n(한국어로 답변해주세요.)" if language == "ko" else ""


def _language_instruction(language: str) -> str:
    """
    Appended to every system prompt when language="ko". Kept as one
    shared instruction rather than duplicating translated rule text in
    every build_*_prompt function below - the base model handles
    "respond in Korean" as an instruction reasonably well on its own;
    translating every individual formatting rule by hand would be a
    lot of surface area to keep in sync and get subtly wrong.
    """
    if language == "ko":
        return (
            "\n\nIMPORTANT: Respond in natural, conversational Korean "
            "(한국어), not English. Keep the same tone and structure "
            "rules above, just written in Korean."
        )
    return ""


def build_messages(user_message: str, history: list[dict] | None = None, language: str = "en") -> list[dict]:
    """
    General/casual conversation — greetings, check-ins, open-ended
    questions that don't match a more specific intent.
    """
    system = BASE_PERSONA + (
        "\n\nRules:\n"
        "1. If it's just a greeting or casual check-in, reply with ONE "
        "warm sentence and ONE casual question. Nothing else.\n"
        "2. Never ask more than one question per reply.\n"
        "3. Never offer a menu of options (\"Need help with X? Or Y?\").\n"
        "4. Keep replies short: 1-3 sentences for casual messages, max 5 "
        "for real questions.\n"
        "5. Don't invent university policies — say to check the "
        "handbook/advisor instead.\n\n"
        "Example - student says \"hello\":\n"
        "Good: \"Hey! Good to see you - how's the semester treating you "
        "so far?\"\n"
        "Bad: \"Hello! I'm here to help you succeed. What's on your "
        "mind? Need help with X? Or Y?\""
    ) + _language_instruction(language)
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message + _language_reminder_suffix(language)})
    return messages


def build_gpa_prompt(user_message: str, history: list[dict] | None = None, language: str = "en") -> list[dict]:
    """
    GPA improvement advice. The user's raw message is expected to
    contain their current/target GPA in free text (e.g. "I got 3.5, how
    do I reach 4.5?") since there's no stored profile yet.
    """
    system = BASE_PERSONA + (
        "\n\nA student wants GPA improvement advice.\n\n"
        "Rules:\n"
        "1. If they gave GPA numbers, use them directly in your advice.\n"
        "2. Give exactly 4 short bullet points, one line each. No sub-bullets.\n"
        "3. End with ONE short genuine encouraging sentence.\n"
        "4. Total reply under 100 words.\n"
        "5. Never promise a guaranteed outcome or timeline.\n"
        "6. If no GPA numbers given, just ask for them in one sentence "
        "instead of generic advice."
    ) + _language_instruction(language)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message + _language_reminder_suffix(language)},
    ]


def build_study_plan_prompt(user_message: str, history: list[dict] | None = None, language: str = "en") -> list[dict]:
    """
    Study plan generation for a course/topic/exam the student names.
    """
    system = BASE_PERSONA + (
        "\n\nA student wants a study plan.\n\n"
        "Rules:\n"
        "1. If they gave a timeframe, use it. If not, default to 5 days "
        "and say so in one line.\n"
        "2. Format as \"Day 1\", \"Day 2\", etc - one short line per day, "
        "naming the specific task, not vague advice like \"review material\".\n"
        "3. Include exactly one rest/lighter day if the plan spans 5+ days.\n"
        "4. No introduction paragraph, no closing summary paragraph - "
        "just the schedule and one short encouraging line at the end.\n"
        "5. Keep it realistic: 1-3 focused tasks per day, not a wall of tasks."
    ) + _language_instruction(language)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message + _language_reminder_suffix(language)},
    ]


def build_quiz_prompt(user_message: str, history: list[dict] | None = None, language: str = "en") -> list[dict]:
    """
    Quiz / practice question generation on a topic the student names.
    """
    system = BASE_PERSONA + (
        "\n\nA student wants practice questions.\n\n"
        "Rules:\n"
        "1. Generate exactly 5 multiple-choice questions on the topic "
        "given - not more.\n"
        "2. Four options each, labeled A-D.\n"
        "3. After ALL 5 questions, one \"Answers\" section listing the "
        "correct letter and a one-sentence reason - not a paragraph per "
        "question.\n"
        "4. No introduction before the questions, no summary after the "
        "answers.\n"
        "5. If no topic is clear, ask the student to specify one instead "
        "of guessing."
    ) + _language_instruction(language)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message + _language_reminder_suffix(language)},
    ]


def build_explanation_prompt(user_message: str, history: list[dict] | None = None, language: str = "en") -> list[dict]:
    """
    Concept explanation, e.g. "explain recursion" or "what is entropy".
    """
    system = BASE_PERSONA + (
        "\n\nA student wants a concept explained.\n\n"
        "Rules:\n"
        "1. One-sentence plain definition first.\n"
        "2. Then ONE short concrete example (2-3 sentences).\n"
        "3. Then, only if genuinely useful, one sentence on a common "
        "mistake students make with this concept.\n"
        "4. Total under 120 words. No headers, no bullet list unless the "
        "concept has clear sequential steps."
    ) + _language_instruction(language)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message + _language_reminder_suffix(language)},
    ]


def build_counseling_prompt(
    user_message: str,
    history: list[dict] | None,
    retrieved_chunks: list[str],
    emotion_info: dict,
    trend_info: dict | None = None,
    language: str = "en",
) -> list[dict]:
    """
    Stress / career counseling response. Combines RAG context
    (retrieved from rag.retrieve()) with the inferred emotional state
    of THIS message (from inference.infer_emotional_state()), and
    optionally a multi-session trend summary (from
    trend_service.get_recent_trend()) so Nova can be lightly aware of
    patterns across sessions without sounding like it's surveilling
    the student.

    Not used for crisis-risk messages — those are intercepted earlier
    in chatbot_engine.handle_message() and answered with the fixed
    CRISIS_RESPONSE above, never by the LLM.
    """
    context_block = (
        "\n".join(f"- {c}" for c in retrieved_chunks)
        if retrieved_chunks
        else "(no specific reference material retrieved - respond from "
             "general supportive-listening principles)"
    )

    trend_block = ""
    if trend_info:
        trend_block = (
            f"\n\nMulti-session pattern (background context only, over "
            f"{trend_info['entries_count']} recent check-ins): mood has "
            f"generally been {trend_info['direction']}, most often about "
            f"{trend_info['dominant_stress_type']} stress."
        )

    system = COUNSELING_PERSONA + (
        f"\n\nInferred student state (this message): emotion={emotion_info.get('emotion')}, "
        f"stress_type={emotion_info.get('stress_type')}."
        f"{trend_block}\n\n"
        "Reference material (use only if genuinely relevant, don't force it in):\n"
        f"{context_block}\n\n"
        "Rules:\n"
        "1. Start by validating the feeling in one sentence - be specific "
        "to what they said, not a generic \"I understand\".\n"
        "2. Keep the tone conversational, not clinical. No bullet-point "
        "advice dumps unless they explicitly ask for concrete steps.\n"
        "3. Ask at most one gentle follow-up question, only if it helps "
        "you understand them better - don't interrogate.\n"
        "4. Never diagnose, never claim to be a therapist, never promise "
        "outcomes.\n"
        "5. If the reference material has a directly relevant point, "
        "weave it in naturally as part of the conversation - don't quote "
        "it as a list.\n"
        "6. If the student seems to want to talk to a real person, "
        "encourage that rather than positioning yourself as a substitute.\n"
        "7. If a multi-session pattern is given and it's worsening, you "
        "may gently acknowledge the stretch they've been having ONCE, in "
        "passing - never describe the pattern clinically or make it "
        "sound like you're tracking/monitoring them.\n"
        "8. Total reply under 130 words."
    ) + _language_instruction(language)
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message + _language_reminder_suffix(language)})
    return messages