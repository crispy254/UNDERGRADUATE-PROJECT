"""
database.py
------------
SQLAlchemy engine, session factory, and the declarative Base that
every model in models.py inherits from.

Uses config.DATABASE_URL, which defaults to a local SQLite file so the
whole project runs with zero external setup. Switch to Postgres by
setting DATABASE_URL in a .env file — nothing else needs to change,
since all the ORM code in models.py is database-agnostic.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

# check_same_thread=False is only needed for SQLite (FastAPI can call
# the same session from different threads); it's ignored for other
# database backends.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session and always closes it,
    even if the request raises."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()