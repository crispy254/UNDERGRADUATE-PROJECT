"""
config.py
---------
Centralized settings, loaded from environment variables / a .env file.
Every other module that needs a config value (DB URL, demo-mode flag,
etc.) should import from here rather than reading os.environ directly,
so there's one place to see everything the app depends on.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # reads a .env file in the project root, if present


# --- Database ---
# Defaults to a local SQLite file so the project runs with ZERO setup
# (no Docker, no Postgres install needed) for demos/development.
# Point DATABASE_URL at a real Postgres instance for production, e.g.:
#   postgresql://user:password@localhost:5432/undergraduate_db
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")


# --- Auth / demo mode ---
# The supervisor asked for a simple interface with no login/logout flow
# for this version. REQUIRE_AUTH is the single switch to flip that
# behavior back on later without rewriting the routers: when False,
# endpoints use DEMO_USER_ID instead of a real logged-in user.
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
DEMO_USER_ID = int(os.getenv("DEMO_USER_ID", "1"))
DEMO_USER_EMAIL = os.getenv("DEMO_USER_EMAIL", "demo@undergraduate.local")
DEMO_USER_NAME = os.getenv("DEMO_USER_NAME", "Demo Student")


# --- Ollama / LLM ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# --- CORS ---
# Comma-separated list of allowed frontend origins, e.g.
#   ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]