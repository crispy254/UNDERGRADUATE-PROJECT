"""
routers/feedback.py
--------------------
The other half of the RL loop: a student clicks thumbs-up/down under a
Nova reply, this endpoint records the reward against the right
StrategyLog row, and immediately updates the live bandit in
strategy_selector.py - so the very next request in that same session
can already reflect the new preference (no retraining step, no delay).
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models import StrategyLog
import strategy_selector

router = APIRouter(prefix="/feedback", tags=["Feedback / RL"])


class FeedbackRequest(BaseModel):
    feedback_id: int   # StrategyLog.id, returned in the counseling response
    helpful: bool       # True = thumbs up, False = thumbs down


@router.post("/")
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    log_row = db.query(StrategyLog).filter(StrategyLog.id == request.feedback_id).first()
    if log_row is None:
        raise HTTPException(status_code=404, detail="feedback_id not found")

    if log_row.reward is not None:
        # Already rated - don't double-count the same turn's feedback
        # into the bandit if the student clicks twice.
        return {"status": "already recorded", "feedback_id": request.feedback_id}

    reward = 1.0 if request.helpful else -1.0
    log_row.reward = reward
    db.commit()

    strategy_selector.record_feedback(
        log_row.emotion, log_row.stress_type, log_row.strategy, reward
    )

    return {
        "status": "recorded",
        "feedback_id": request.feedback_id,
        "strategy": log_row.strategy,
        "reward": reward,
    }