from sqlalchemy import (
    Column, Integer, String, ForeignKey, DateTime,
    UniqueConstraint
)

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from tuneparam.database.db import Base

from sqlalchemy import UniqueConstraint

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True, name="id")
    model_size = Column(String, nullable=True, name="model_size")
    model_type = Column(String, nullable=True, name="model_type")  # 모델 이름이라고 가정
    dataset_size = Column(String, nullable=True, name="dataset_size")
    dataset_type = Column(String, nullable=True, name="dataset_type")
    goal = Column(String, nullable=True, name="goal")
    total_epoch = Column(Integer, default=0, nullable=True, name="total_epoch")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), name="created_at")
    version = Column(String, nullable=False, name="version")

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, name="user_id")
    user = relationship("User", back_populates="models", lazy="selectin")

    training_logs = relationship(
        "TrainingLog",
        back_populates="model",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("version", "model_type", name="uq_version_modeltype"),
    )
