import os
import shutil
from uuid import uuid4

from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session

from database import get_db
from auth import get_current_user
from schemas import ResourceResponse, ResourceUpdate
from services.resources_service import (
    save_resource,
    list_resources_by_type,
    list_all_resources,
    get_resource_by_id,
    update_resource,
    delete_resource
)

router = APIRouter(
    prefix="/resources",
    tags=["Academic Resources"]
)

UPLOAD_DIR = "Uploads"
RESOURCE_TYPE = "study_guide"


@router.post("/", response_model=ResourceResponse)
def upload_study_guide(
    title: str = Form(...),
    course: str = Form(None),
    year: str = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    unique_name = f"{uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, unique_name)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return save_resource(
        db=db,
        title=title,
        type=RESOURCE_TYPE,
        course=course,
        year=year,
        filename=file.filename,
        filepath=filepath,
        uploaded_by=current_user.id
    )


@router.get("/", response_model=list[ResourceResponse])
def get_all_resources(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return list_all_resources(db)


@router.get("/{resource_id}", response_model=ResourceResponse)
def get_resource(
    resource_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return get_resource_by_id(db, resource_id)


@router.patch("/{resource_id}", response_model=ResourceResponse)
def edit_resource(
    resource_id: int,
    updates: ResourceUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return update_resource(db, resource_id, updates)


@router.delete("/{resource_id}")
def remove_resource(
    resource_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return delete_resource(db, resource_id)