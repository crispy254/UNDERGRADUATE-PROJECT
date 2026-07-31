"""
routers/chatbot.py
-------------------
HTTP layer for the AI chatbot (academic companion + stress/career
counseling). Real logic lives in services/chatbot_service.py; this
file only handles request/response wiring.

NO LOGIN REQUIRED (per supervisor's direction — this is a simple
interface, not a multi-user authenticated app). Every endpoint resolves
a user_id via resolve_user_id() below instead of a
Depends(get_current_user) auth dependency:
  - if the client sends a user_id in the request body, that's used
  - otherwise it falls back to config.DEMO_USER_ID

This keeps ChatHistory/EmotionLog rows tied to a user_id (so the
existing DB schema and trend-tracking logic don't need to change), but
nobody has to log in or out to use the app. If real multi-user auth
gets added later, swap resolve_user_id() for the auth dependency and
nothing else in this file needs to change.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from config import DEMO_USER_ID
from schemas import (
    ChatRequest,
    ChatResponse,
    StudyPlanRequest,
    GPATipsRequest,
    QuizRequest,
    ExplainRequest,
    ChatHistoryResponse,
    CounselingRequest,
    CounselingResponse,
)
from services.chatbot_service import (
    get_chat_reply,
    generate_study_plan,
    generate_gpa_tips,
    generate_quiz,
    generate_explanation,
    get_user_chat_history,
    get_counseling_reply,
    ensure_user_exists,
)

router = APIRouter(
    prefix="/chat",
    tags=["AI Chatbot"]
)


def resolve_user_id(request_user_id: int | None, db: Session) -> int:
    """Returns request_user_id if given, otherwise the demo user's ID.
    Also makes sure that user row actually exists (auto-creates the
    demo user on first use) so FK constraints on ChatHistory/EmotionLog
    never fail."""
    user_id = request_user_id if request_user_id is not None else DEMO_USER_ID
    ensure_user_exists(db, user_id)
    return user_id


@router.post("/", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    user_id = resolve_user_id(request.user_id, db)
    return get_chat_reply(db, user_id, request.message, language=request.language)  # ← pass language to service


@router.post("/counseling", response_model=CounselingResponse)
def counseling(
    request: CounselingRequest,
    db: Session = Depends(get_db),
):
    # Stress / career counseling — routes through RAG + emotion
    # inference in chatbot_engine.py, and logs inferred emotional
    # state via services/chatbot_service.py.
    user_id = resolve_user_id(request.user_id, db)
    return get_counseling_reply(db, user_id, request.message, language=request.language)  # ← pass language to service


@router.post("/study-plan")
def study_plan(request: StudyPlanRequest):
    return generate_study_plan(request)


@router.post("/gpa-tips")
def gpa_tips(request: GPATipsRequest):
    return generate_gpa_tips(request)


@router.post("/quiz")
def quiz(request: QuizRequest):
    return generate_quiz(request)


@router.post("/explain")
def explain(request: ExplainRequest):
    return generate_explanation(request)


@router.get("/history", response_model=list[ChatHistoryResponse])
def chat_history(
    user_id: int = DEMO_USER_ID,
    db: Session = Depends(get_db),
):
    return get_user_chat_history(db, user_id)