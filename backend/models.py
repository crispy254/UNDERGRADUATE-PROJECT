
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
from datetime import datetime
 
from database import Base
 
 
class User(Base):
    __tablename__ = "users"
 
    id = Column(Integer, primary_key=True, index=True)
    fullname = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
 
 
class UploadedFile(Base):
    __tablename__ = "uploaded_files"
 
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(255), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User")
 
 
# MEMBER 2 - Academic Resources (notes, past papers, study guides)
class Resource(Base):
    __tablename__ = "resources"
 
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # "note" | "pastpaper" | "study_guide"
    course = Column(String(100), nullable=True)
    year = Column(String(10), nullable=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(255), nullable=False)
    uploaded_by = Column(Integer, ForeignKey("users.id"))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    uploader = relationship("User")
 
 
# MEMBER 1 - AI & Chatbot
class ChatHistory(Base):
    __tablename__ = "chat_history"
 
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    message = Column(Text, nullable=False)
    reply = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User")
 
 
# MEMBER 1 - AI & Chatbot: inferred emotional state per counseling turn.
# Kept as a separate table (rather than columns on ChatHistory) so
# academic-intent turns don't carry always-null emotion columns, and so
# this can be queried independently for per-user trend tracking, e.g.
# "has this student shown rising stress signals over the last 2 weeks."
class EmotionLog(Base):
    __tablename__ = "emotion_logs"
 
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    chat_history_id = Column(Integer, ForeignKey("chat_history.id"), nullable=True)
    emotion = Column(String(50), nullable=False)       # e.g. anxious, overwhelmed, sad, neutral
    stress_type = Column(String(50), nullable=False)   # e.g. academic, career, interpersonal, general
    risk_flag = Column(Integer, default=0)              # 0/1 - stored as int for cross-DB (e.g. SQLite) safety
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User")
 
 
# MEMBER 3 - Scholarships & Projects
class Scholarship(Base):
    __tablename__ = "scholarships"
 
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    field = Column(String(100), nullable=True)
    deadline = Column(String(50), nullable=True)
    link = Column(String(255), nullable=True)
    added_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
 
 
class Project(Base):
    __tablename__ = "projects"
 
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True)
    link = Column(String(255), nullable=True)
    github_url = Column(String(255), nullable=True)
    file_path = Column(String(255), nullable=True)
    added_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
 
class StrategyLog(Base):
    """
    One row per RL strategy decision. `reward` starts NULL (no
    feedback yet) and gets filled in by POST /feedback once the
    student clicks thumbs up/down - see routers/feedback.py.
 
    This table is BOTH how the bandit persists across server restarts
    (replayed via strategy_selector.warm_start() at app startup - see
    app.py wiring notes) AND the raw data for the evaluation section -
    every row is one (context, action, reward) sample for computing
    precision/recall/F1 on strategy selection.
    """
    __tablename__ = "strategy_logs"
 
    id = Column(Integer, primary_key=True, index=True)
    chat_history_id = Column(Integer, ForeignKey("chat_history.id"), nullable=True)
    emotion = Column(String(50), nullable=False)
    stress_type = Column(String(50), nullable=False)
    strategy = Column(String(50), nullable=False)   # one of strategy_selector.STRATEGIES
    reward = Column(Float, nullable=True)            # NULL until feedback arrives; +1.0 / -1.0 after
    created_at = Column(DateTime, default=datetime.utcnow)
 