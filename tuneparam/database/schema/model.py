from sqlalchemy import (
    Column, Integer, String, ForeignKey, DateTime,
    UniqueConstraint
)

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from tuneparam.database.db import Base

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True, name="id")
    model_size = Column(String, nullable=False, name="model_size")
    model_type = Column(String, nullable=False, name="model_type")
    dataset_size = Column(String, name="dataset_size")
    dataset_type = Column(String, name="dataset_type")
    goal = Column(String, name="goal")
    total_epoch = Column(Integer, default=0, nullable=False, name="total_epoch")  # 이 부분 추가
    created_at = Column(DateTime(timezone=True), server_default=func.now(), name="created_at")

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, name="user_id")
    user = relationship("User", back_populates="models", lazy="selectin")

    training_logs = relationship(
        "TrainingLog",
        back_populates="model",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("user_id", "model_size", "model_type", name="uq_user_modelsize_type"),
    )
