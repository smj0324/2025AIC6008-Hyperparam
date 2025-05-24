# crud.py
from sqlalchemy.orm import Session
from typing import List, Optional
from tuneparam.database.schema import User

# CREATE
def create_user(db: Session, user_data: dict) -> User:
    new_user = User(**user_data)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# READ - 단일 사용자 (by username)
def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

# READ - 전체 사용자 리스트
def get_all_users(db: Session) -> List[User]:
    return db.query(User).order_by(User.created_at.desc()).all()

# UPDATE - 사용자 정보 (by username)
def update_user_by_username(db: Session, username: str, update_data: dict) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if user:
        for key, value in update_data.items():
            if hasattr(user, key):
                setattr(user, key, value)
        db.commit()
        db.refresh(user)
    return user

# DELETE - 사용자 삭제 (by username)
def delete_user_by_username(db: Session, username: str) -> bool:
    user = db.query(User).filter(User.username == username).first()
    if user:
        db.delete(user)
        db.commit()
        return True
    return False
