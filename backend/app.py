"""
app.py
------
FastAPI application entry point.

Run with:
    uvicorn app:app --reload

On startup this:
  1. Creates all DB tables if they don't exist yet (Base.metadata.create_all)
  2. Seeds a demo user (see config.DEMO_USER_ID) so the no-login flow
     always has a valid user_id to attach chats to
  3. Warm-starts the RL strategy bandit from any historical feedback
     already in strategy_logs, so it doesn't reset to "no preference"
     every time the server restarts
  4. Mounts the chatbot router

Other team members' routers (materials, pastpapers, projects,
resources, scholarships, upload, users) are mounted too IF their files
are present — each import is wrapped in try/except so this app still
boots even if a teammate's module isn't finished yet or isn't in this
checkout. Check the startup log output for which routers actually
loaded.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOWED_ORIGINS, DEMO_USER_ID, DEMO_USER_EMAIL, DEMO_USER_NAME
from database import Base, engine, SessionLocal
from models import User, StrategyLog
import strategy_selector


def _seed_demo_user() -> None:
    """Creates the demo user row if it doesn't exist yet, so the very
    first request under the no-login flow already has a valid user_id
    to write ChatHistory/EmotionLog rows against."""
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.id == DEMO_USER_ID).first():
            db.add(User(
                id=DEMO_USER_ID,
                fullname=DEMO_USER_NAME,
                email=DEMO_USER_EMAIL,
                password="",
            ))
            db.commit()
    finally:
        db.close()


def _warm_start_bandit() -> None:
    """Replays every historical (context, action, reward) row from
    strategy_logs into the in-memory bandit, so learned preferences
    survive a server restart instead of resetting to uniform."""
    db = SessionLocal()
    try:
        rows = (
            db.query(StrategyLog)
            .filter(StrategyLog.reward.isnot(None))
            .order_by(StrategyLog.created_at)
            .all()
        )
        n = strategy_selector.warm_start(
            [(r.emotion, r.stress_type, r.strategy, r.reward) for r in rows]
        )
        print(f"[app.py] Warm-started bandit from {n} historical feedback rows")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    Base.metadata.create_all(bind=engine)
    _seed_demo_user()
    _warm_start_bandit()
    yield
    # --- shutdown --- (nothing to clean up currently)


app = FastAPI(
    title="Undergraduate AI Companion API",
    description="Academic tutoring + stress/career counseling chatbot backend.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Chatbot router (this is the piece built in this project) ---
from routers.chatbot import router as chatbot_router  # noqa: E402
app.include_router(chatbot_router)
# --- Feedback / RL router (thumbs-up/down → strategy bandit) ---
from routers.feedback import router as feedback_router  # noqa: E402
app.include_router(feedback_router)

# --- Teammates' routers: mounted only if present, so the app still
# boots without them during development. ---
_optional_routers = [
    ("routers.materials", "router", "/materials", "Materials"),
    ("routers.pastpapers", "router", "/pastpapers", "Past Papers"),
    ("routers.projects", "router", "/projects", "Projects"),
    ("routers.resources", "router", "/resources", "Resources"),
    ("routers.scholarships", "router", "/scholarships", "Scholarships"),
    ("routers.upload", "router", "/upload", "Upload"),
    ("routers.users", "router", "/users", "Users"),
]

_loaded, _skipped = [], []
for module_path, attr_name, _prefix, label in _optional_routers:
    try:
        module = __import__(module_path, fromlist=[attr_name])
        app.include_router(getattr(module, attr_name))
        _loaded.append(label)
    except ImportError:
        _skipped.append(label)

if _skipped:
    print(f"[app.py] Skipped routers not found yet: {', '.join(_skipped)}")
if _loaded:
    print(f"[app.py] Loaded routers: {', '.join(_loaded)}")


@app.get("/health")
def health():
    """Basic liveness check — does NOT check Ollama, since that's a
    separate concern (see llm.check_ollama_status for that)."""
    return {"status": "ok"}

# --- Voice (Whisper) router ---
from routers import voice  # noqa: E402
app.include_router(voice.router)
from fastapi.staticfiles import StaticFiles

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static assets (3D avatar model, etc.) ---
app.mount("/static", StaticFiles(directory="static"), name="static")