from sqlalchemy.orm import Session
from fastapi import HTTPException
from models import Project


def create_project(db: Session, data, added_by: int):
    project = Project(
        title=data.title,
        description=data.description,
        category=data.category,
        link=data.link,
        github_url=data.github_url,
        added_by=added_by
    )

    db.add(project)
    db.commit()
    db.refresh(project)

    return project


def create_project_with_file(db: Session, project_data: dict, added_by: int):
    project = Project(
        title=project_data["title"],
        description=project_data["description"],
        category=project_data["category"],
        github_url=project_data.get("github_url"),
        file_path=project_data.get("file_path"),
        added_by=added_by
    )

    db.add(project)
    db.commit()
    db.refresh(project)

    return project


def list_projects(db: Session, category: str = None):
    query = db.query(Project)

    if category:
        query = query.filter(Project.category == category)

    return query.order_by(Project.created_at.desc()).all()


def get_project_by_id(db: Session, project_id: int):
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return project


def recommend_projects(db: Session, current_user):
    return db.query(Project).order_by(Project.created_at.desc()).limit(5).all()


def update_project(db: Session, project_id: int, updates):
    project = get_project_by_id(db, project_id)

    for field, value in updates.model_dump(exclude_unset=True).items():
        setattr(project, field, value)

    db.commit()
    db.refresh(project)

    return project


def delete_project(db: Session, project_id: int):
    project = get_project_by_id(db, project_id)

    db.delete(project)
    db.commit()

    return {"detail": "Project deleted"}