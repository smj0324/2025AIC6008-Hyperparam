# crud_model.py
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from tuneparam.database.schema import Model, User


# CREATE - 특정 유저에게 모델 추가
def create_model_for_user(db: Session, username: str, model_data: dict) -> Model:
    user = db.query(User).filter(User.username == username).first()

    new_model = Model(**model_data)

    if user:
        new_model.user = user  # 외래키 연결
    else:
        new_model.user_id = None  # 명시적으로 None 설정 (nullable=True이기 때문)

    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return new_model


# READ - 특정 유저의 모든 모델 조회
def get_model_by_version_and_type(db: Session, version: str, model_type: str) -> Optional[Model]:
    return db.query(Model).filter(
        Model.version == version,
        Model.model_type == model_type
    ).first()


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
