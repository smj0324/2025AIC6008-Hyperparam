from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from tuneparam.database.db import Base


class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    epoch = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=False)
    val_accuracy = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    model = relationship("Model", back_populates="training_logs", lazy="selectin")