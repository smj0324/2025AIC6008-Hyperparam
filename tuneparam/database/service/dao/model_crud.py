# crud_model.py
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional

from tuneparam.database.schema.models import Model, User


# CREATE - 특정 유저에게 모델 추가
def create_model_for_user(db: Session, username: str, model_data: dict) -> Optional[Model]:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None

    new_model = Model(**model_data)
    new_model.user = user  # 외래키 연결
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return new_model


# READ - 특정 유저의 모든 모델 조회
def get_models_by_username(db: Session, username: str) -> List[Model]:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return []
    return user.models


# UPDATE - 모델 정보 수정 (복합키 기반)
def update_model_for_user(db: Session, username: str, model_size: str, model_type: str, update_data: dict) -> Optional[
    Model]:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None

    model = db.query(Model).filter(
        Model.user_id == user.id,
        Model.model_size == model_size,
        Model.model_type == model_type
    ).first()

    if model:
        for key, value in update_data.items():
            if hasattr(model, key):
                setattr(model, key, value)
        db.commit()
        db.refresh(model)
        return model
    return None


# DELETE - 모델 삭제 (복합키 기반)
def delete_model_for_user(db: Session, username: str, model_size: str, model_type: str) -> bool:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False

    model = db.query(Model).filter(
        Model.user_id == user.id,
        Model.model_size == model_size,
        Model.model_type == model_type
    ).first()

    if model:
        db.delete(model)
        db.commit()
        return True
    return False
