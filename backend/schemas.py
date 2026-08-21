from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime



class UserCreate(BaseModel):
    fullname: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    fullname: str
    email: EmailStr

    class Config:
        from_attributes = True
class Token(BaseModel):
    access_token: str
    token_type: str


# ---- MEMBER 1 - Chatbot (academic) ----

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[int] = None  # falls back to DEMO_USER_ID when auth is off (see config.REQUIRE_AUTH)
    language: str = "en"          # ← add this line

class ChatResponse(BaseModel):
    reply: str


class StudyPlanRequest(BaseModel):
    subject: str
    goal: str
    time_available: str


class GPATipsRequest(BaseModel):
    current_gpa: float
    target_gpa: float
    subjects: list[str]


class QuizRequest(BaseModel):
    topic: str
    num_questions: int = 5


class ExplainRequest(BaseModel):
    topic: str


class ChatHistoryResponse(BaseModel):
    id: int
    message: str
    reply: str
    created_at: datetime

    class Config:
        from_attributes = True


# ---- MEMBER 1 - Chatbot (stress / career counseling) ----

class CounselingRequest(BaseModel):
    message: str
    user_id: Optional[int] = None
    language: str = "en"          # ← add this line

class CounselingResponse(BaseModel):
    reply: str
    emotion: str | None = None
    stress_type: str | None = None
    risk_flag: bool = False
    feedback_id: int | None = None
    audio_base64: str | None = None   # ← add this


class EmotionLogResponse(BaseModel):
    id: int
    emotion: str
    stress_type: str
    risk_flag: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ---- MEMBER 2 - Academic Resources ----

class ResourceResponse(BaseModel):
    id: int
    title: str
    type: str
    course: Optional[str] = None
    year: Optional[str] = None
    filename: str
    filepath: str
    uploaded_at: datetime

    class Config:
        from_attributes = True
class ResourceUpdate(BaseModel):
    title: Optional[str] = None
    course: Optional[str] = None
    year: Optional[str] = None


# ---- MEMBER 3 - Scholarships & Projects ----

class ScholarshipCreate(BaseModel):
    title: str
    description: Optional[str] = None
    field: Optional[str] = None
    deadline: Optional[str] = None
    link: Optional[str] = None


class ScholarshipResponse(ScholarshipCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ProjectCreate(BaseModel):
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    link: Optional[str] = None
    github_url: Optional[str] = None


class ProjectResponse(ProjectCreate):
    id: int
    file_path: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    link: Optional[str] = None
    github_url: Optional[str] = None