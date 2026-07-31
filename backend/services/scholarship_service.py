from sqlalchemy.orm import Session
from fastapi import HTTPException
from models import Scholarship

# MEMBER 3 - Scholarships
# Replace recommend_scholarships() with real logic later
# (e.g. matching by field of study, GPA eligibility, etc.)


def create_scholarship(db: Session, data, added_by: int):
    scholarship = Scholarship(
        title=data.title,
        description=data.description,
        field=data.field,
        deadline=data.deadline,
        link=data.link,
        added_by=added_by
    )

    db.add(scholarship)
    db.commit()
    db.refresh(scholarship)

    return scholarship


def list_scholarships(db: Session, field: str = None):
    query = db.query(Scholarship)

    if field:
        query = query.filter(Scholarship.field == field)

    return query.order_by(Scholarship.created_at.desc()).all()


def get_scholarship_by_id(db: Session, scholarship_id: int):
    scholarship = db.query(Scholarship).filter(Scholarship.id == scholarship_id).first()

    if not scholarship:
        raise HTTPException(status_code=404, detail="Scholarship not found")

    return scholarship


def recommend_scholarships(db: Session, current_user):
    # MEMBER 3: replace with real recommendation logic
    return db.query(Scholarship).order_by(Scholarship.created_at.desc()).limit(5).all()