"""
services/chatbot_service.py
----------------------------
Bridges the API layer (routers/chatbot.py) and the AI layer
(chatbot_engine.py). Handles DB persistence — chatbot_engine.py stays
stateless per its own docstring.

NOTE: the original version of this file wasn't available when this
was written, so this is a full reconstruction based on the function
names/signatures routers/chatbot.py already expects
(get_chat_reply, generate_study_plan, generate_gpa_tips, generate_quiz,
generate_explanation, get_user_chat_history), plus the new
get_counseling_reply for the counseling endpoint. If your original
file did more (e.g. rate limiting, extra validation), merge that back
in — this covers the wiring needed for RAG + emotion logging to work.
"""
from sqlalchemy.orm import Session

import chatbot_engine
import trend_service
from models import ChatHistory, EmotionLog, User
from config import DEMO_USER_ID, DEMO_USER_EMAIL, DEMO_USER_NAME


def ensure_user_exists(db: Session, user_id: int) -> None:
    """
    Auto-creates a placeholder User row for user_id if one doesn't
    exist yet. This is what lets the chatbot work with no login flow:
    the frontend can just send messages (optionally with a user_id it
    picked itself, e.g. a browser-generated ID), and this fills in the
    FK row ChatHistory/EmotionLog need, instead of requiring a real
    signup/login first.

    For the default demo user (DEMO_USER_ID), a friendly placeholder
    name/email is used. For any other user_id the frontend invents,
    a generic placeholder is used instead.
    """
    existing = db.query(User).filter(User.id == user_id).first()
    if existing:
        return

    if user_id == DEMO_USER_ID:
        db.add(User(id=user_id, fullname=DEMO_USER_NAME, email=DEMO_USER_EMAIL, password=""))
    else:
        db.add(User(id=user_id, fullname=f"Guest {user_id}", email=f"guest{user_id}@undergraduate.local", password=""))
    db.commit()


def _recent_history(db: Session, user_id: int, limit: int = 6) -> list[dict]:
    """
    Pull the last `limit` turns for this user, oldest first, in the
    {"role": ..., "content": ...} shape chatbot_engine.py expects.
    """
    rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.created_at.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()

    history = []
    for row in rows:
        history.append({"role": "user", "content": row.message})
        history.append({"role": "assistant", "content": row.reply})
    return history


def get_chat_reply(db: Session, user_id: int, message: str, language: str = "en") -> dict:
    history = _recent_history(db, user_id)
    result = chatbot_engine.handle_message(message, history, language=language)  # ← add language=language
    ...

def get_counseling_reply(db: Session, user_id: int, message: str, language: str = "en") -> dict:
    history = _recent_history(db, user_id)
    trend_info = trend_service.get_recent_trend(db, user_id)
    result = chatbot_engine.handle_message(message, history, trend_info, language=language)  # ← add language=language
    ...
    if not result["error"]:
        chat_row = ChatHistory(user_id=user_id, message=message, reply=result["response"])
        db.add(chat_row)
        db.flush()  # assigns chat_row.id without committing yet

        emotion_info = result.get("emotion")
        if emotion_info:
            db.add(EmotionLog(
                user_id=user_id,
                chat_history_id=chat_row.id,
                emotion=emotion_info.get("emotion", "neutral"),
                stress_type=emotion_info.get("stress_type", "general"),
                risk_flag=1 if emotion_info.get("risk") else 0,
            ))
        db.commit()

    emotion_info = result.get("emotion") or {}
    return {
        "reply": result["response"] or "Sorry, I couldn't process that right now.",
        "emotion": emotion_info.get("emotion"),
        "stress_type": emotion_info.get("stress_type"),
        "risk_flag": bool(emotion_info.get("risk")),
    }


def generate_study_plan(request) -> dict:
    prompt = (
        f"Subject: {request.subject}. Goal: {request.goal}. "
        f"Time available: {request.time_available}."
    )
    result = chatbot_engine.handle_message(prompt)
    return {"reply": result["response"]}


def generate_gpa_tips(request) -> dict:
    prompt = (
        f"My current GPA is {request.current_gpa}, target is {request.target_gpa}. "
        f"Subjects: {', '.join(request.subjects)}."
    )
    result = chatbot_engine.handle_message(prompt)
    return {"reply": result["response"]}


def generate_quiz(request) -> dict:
    prompt = f"Give me {request.num_questions} practice questions on {request.topic}."
    result = chatbot_engine.handle_message(prompt)
    return {"reply": result["response"]}


def generate_explanation(request) -> dict:
    prompt = f"Explain {request.topic}."
    result = chatbot_engine.handle_message(prompt)
    return {"reply": result["response"]}


def get_user_chat_history(db: Session, user_id: int):
    return (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.created_at.asc())
        .all()
    )