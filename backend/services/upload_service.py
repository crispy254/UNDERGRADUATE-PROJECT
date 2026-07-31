from sqlalchemy.orm import Session
from models import UploadedFile


def save_uploaded_file(
    db: Session,
    owner_id: int,
    filename: str,
    filepath: str
):
    db_file = UploadedFile(
        filename=filename,
        filepath=filepath,
        owner_id=owner_id
    )

    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    return db_file