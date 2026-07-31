import os
import shutil

from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session

from database import get_db
from auth import get_current_user
from schemas import ProjectCreate, ProjectResponse, ProjectUpdate
from services.project_service import (
    create_project,
    create_project_with_file,
    list_projects,
    get_project_by_id,
    recommend_projects,
    update_project,
    delete_project
)

router = APIRouter(
    prefix="/projects",
    tags=["Projects"]
)

UPLOAD_DIR = "Uploads"


@router.post("/", response_model=ProjectResponse)
def add_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return create_project(db, project, current_user.id)


@router.post("/upload", response_model=ProjectResponse)
def add_project_with_file(
    title: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    github_url: str = Form(None),
    file: UploadFile = File(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    file_path = None

    if file:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, f"{current_user.id}_{file.filename}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    project_data = {
        "title": title,
        "description": description,
        "category": category,
        "github_url": github_url,
        "file_path": file_path
    }

    return create_project_with_file(db, project_data, added_by=current_user.id)


@router.get("/", response_model=list[ProjectResponse])
def get_projects(
    category: str = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return list_projects(db, category)


@router.get("/recommend", response_model=list[ProjectResponse])
def get_recommended_projects(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return recommend_projects(db, current_user)


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return get_project_by_id(db, project_id)


@router.patch("/{project_id}", response_model=ProjectResponse)
def edit_project(
    project_id: int,
    updates: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return update_project(db, project_id, updates)


@router.delete("/{project_id}")
def remove_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    return delete_project(db, project_id)