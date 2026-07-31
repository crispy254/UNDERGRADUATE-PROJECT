from sqlalchemy.orm import Session

from models import User
from schemas import UserCreate
from auth import hash_password



def create_user(db: Session, user: UserCreate):

    db_user = User(
        fullname=user.fullname,
        email=user.email,
        password=hash_password(user.password)
    )

    db.add(db_user)

    db.commit()

    db.refresh(db_user)

    return db_user


def get_user_by_email(db: Session, email: str):

    return db.query(User).filter(User.email == email).first()
#log in service
from auth import verify_password


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)

    if not user:
        return None

    if not verify_password(password, user.password):
        return None

    return user