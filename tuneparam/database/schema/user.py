from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from tuneparam.database.db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True, name="id")
    username = Column(String, nullable=False, unique=True, name="username")  # ✅ 고유 제약 추가
    version = Column(String, nullable=False, name="version")
    hardware = Column(String, name="hardware")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), name="created_at")

    models = relationship(
        "Model",
        back_populates="user",
        cascade="all, delete-orphan",  # ✅ 모델 삭제 시 함께 삭제
        lazy="selectin"
    )

