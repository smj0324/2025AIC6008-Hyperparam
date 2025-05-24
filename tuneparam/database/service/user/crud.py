# crud.py
from sqlalchemy.orm import Session
from tuneparam.database.schema.models import User

def create_user(db: Session, user_data: dict) -> User:
    new_user = User(**user_data)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# READ (단일 사용자 by id)
def get_user_by_username(db: Session, user_name: str) -> User:
    return db.query(User).filter(User.username == user_name).first()
#
# READ (전체 사용자 리스트)
def get_all_users(db: Session) -> list[User]:
    return db.query(User).order_by(User.created_at.desc()).all()

# UPDATE
def update_user_by_username(db: Session, user_name: str, update_data: dict) -> User:
    user = db.query(User).filter(User.username == user_name).first()

    if user:
        for key, value in update_data.items():
            setattr(user, key, value)
        db.commit()
        db.refresh(user)
    return user
#
# DELETE
def delete_user_by_username(db: Session, user_name: str) -> bool:
    user = db.query(User).filter(User.username == user_name).first()
    if user:
        db.delete(user)
        db.commit()
        return True
    return False