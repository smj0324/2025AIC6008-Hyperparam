from sqlalchemy.orm import Session
from tuneparam.database.schema import TrainingLog
from typing import List, Optional

def create_training_log(db: Session, model_id: int, log_data: dict) -> TrainingLog:
    new_log = TrainingLog(**log_data, model_id=model_id)
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    return new_log

def get_training_logs_by_model(db: Session, model_id: int) -> List[TrainingLog]:
    return db.query(TrainingLog).filter(TrainingLog.model_id == model_id).order_by(TrainingLog.epoch).all()

def update_training_log(db: Session, log_id: int, update_data: dict) -> Optional[TrainingLog]:
    log = db.query(TrainingLog).filter(TrainingLog.id == log_id).first()
    if log:
        for key, value in update_data.items():
            setattr(log, key, value)
        db.commit()
        db.refresh(log)
        return log
    return None

def delete_training_logs_by_model(db: Session, model_id: int) -> int:
    logs = db.query(TrainingLog).filter(TrainingLog.model_id == model_id)
    count = logs.count()
    if count > 0:
        logs.delete(synchronize_session='fetch')
        db.commit()
    return count

def get_all_training_logs(db: Session) -> List[TrainingLog]:
    return db.query(TrainingLog).order_by(TrainingLog.model_id, TrainingLog.epoch).all()
