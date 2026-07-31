from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from auth import get_current_user
from schemas import ScholarshipCreate, ScholarshipResponse
from services.scholarship_service import (
    create_scholarship,
    list_scholarships,
    get_scholarship_by_id,
    recommend_scholarships
)

router = APIRouter(
    prefix="/scholarships",
    tags=["Scholarships"]
)


@router.post("/", response_model=ScholarshipResponse)
def add_scholarship(
    scholarship: ScholarshipCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return create_scholarship(db, scholarship, current_user.id)


@router.get("/", response_model=list[ScholarshipResponse])
def get_scholarships(
    field: str = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return list_scholarships(db, field)


@router.get("/recommend", response_model=list[ScholarshipResponse])
def get_recommended_scholarships(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    # MEMBER 3: replace with real recommendation logic
    return recommend_scholarships(db, current_user)


@router.get("/{scholarship_id}", response_model=ScholarshipResponse)
def get_scholarship(
    scholarship_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return get_scholarship_by_id(db, scholarship_id)