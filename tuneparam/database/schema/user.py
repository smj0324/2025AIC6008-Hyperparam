from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from tuneparam.database.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True, name="id")
    username = Column(String, nullable=False, name="username")
    version = Column(String, nullable=False, name="version")
    hardware = Column(String, name="hardware")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), name="created_at")

    models = relationship("Model", back_populates="user", cascade="all, delete", lazy="selectin")
