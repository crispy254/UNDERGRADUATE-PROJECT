"""
trend_service.py
-----------------
Multi-session trend analysis over EmotionLog, used to give the
counseling prompt awareness of patterns across time (e.g. "this
student's stress has been climbing this week") rather than treating
every message as an isolated event.

Deliberately lives in the service layer, not chatbot_engine.py --
chatbot_engine.py is documented as stateless, so all DB access happens
here, and the result is passed INTO chatbot_engine.handle_message() as
a plain dict.

This is intentionally simple (rule-based severity ranking + a
first-half-vs-second-half comparison), not a trained time-series
model. It's meant to catch obvious drift, not make clinical claims.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from models import EmotionLog

# Emotions ranked roughly least -> most distressed. Used only to detect
# directional drift (e.g. hopeful -> anxious -> overwhelmed over time).
# This is NOT a clinical severity scale -- just enough signal to flag
# "trending worse" vs. "trending better" for prompt context.
EMOTION_SEVERITY = {
    "hopeful": 0,
    "neutral": 1,
    "angry": 2,
    "sad": 2,
    "anxious": 3,
    "overwhelmed": 4,
}

LOOKBACK_DAYS = 14
DIRECTION_THRESHOLD = 0.75   # min average-severity delta to call it a real trend, not noise
RISK_REVIEW_THRESHOLD = 2    # >=2 risk-flagged turns in the window -> flag for human review


def _avg_severity(entries: List[EmotionLog]) -> float:
    if not entries:
        return 1.0
    scores = [EMOTION_SEVERITY.get(e.emotion, 1) for e in entries]
    return sum(scores) / len(scores)


def get_recent_trend(db: Session, user_id: int, days: int = LOOKBACK_DAYS) -> Optional[Dict]:
    """
    Returns a summary of this user's emotional pattern over the last
    `days` days, or None if there isn't enough history yet (fewer than
    2 logged entries -- a single data point has no "trend").

    Returns:
        {
            "dominant_emotion": str,
            "dominant_stress_type": str,
            "direction": "improving" | "worsening" | "stable",
            "entries_count": int,
            "risk_count": int,
            "needs_review": bool,   # for admin monitoring -- never surface this to the student directly
        }
    """
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(EmotionLog)
        .filter(EmotionLog.user_id == user_id, EmotionLog.created_at >= since)
        .order_by(EmotionLog.created_at.asc())
        .all()
    )

    if len(rows) < 2:
        return None

    emotions = [r.emotion for r in rows]
    stress_types = [r.stress_type for r in rows]
    risk_count = sum(1 for r in rows if r.risk_flag)

    dominant_emotion = max(set(emotions), key=emotions.count)
    dominant_stress_type = max(set(stress_types), key=stress_types.count)

    mid = len(rows) // 2
    first_half = rows[:mid] or rows
    second_half = rows[mid:]

    delta = _avg_severity(second_half) - _avg_severity(first_half)
    if delta >= DIRECTION_THRESHOLD:
        direction = "worsening"
    elif delta <= -DIRECTION_THRESHOLD:
        direction = "improving"
    else:
        direction = "stable"

    return {
        "dominant_emotion": dominant_emotion,
        "dominant_stress_type": dominant_stress_type,
        "direction": direction,
        "entries_count": len(rows),
        "risk_count": risk_count,
        "needs_review": risk_count >= RISK_REVIEW_THRESHOLD,
    }


def get_flagged_users(db: Session, days: int = LOOKBACK_DAYS) -> List[int]:
    """
    Returns user_ids whose recent EmotionLog history crosses the
    review threshold -- for an admin monitoring view (the proposal's
    "관리자용 모니터링 기능"). This only returns IDs; wire it behind
    your actual admin-auth dependency before exposing it via a route,
    since it's sensitive data.
    """
    since = datetime.utcnow() - timedelta(days=days)
    rows = db.query(EmotionLog).filter(EmotionLog.created_at >= since).all()

    by_user: Dict[int, int] = {}
    for r in rows:
        if r.risk_flag:
            by_user[r.user_id] = by_user.get(r.user_id, 0) + 1

    return [uid for uid, count in by_user.items() if count >= RISK_REVIEW_THRESHOLD]