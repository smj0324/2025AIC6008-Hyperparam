# models.py
from sqlalchemy import Column, Integer, String, DateTime, func
from tuneparam.database.db import Base

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)


class User(Base):
    __tablename__ = "User"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, nullable=False)
    version = Column(String, nullable=False)
    hardware = Column(String)
    model_size = Column(String)
    dataset_size = Column(String)
    model_type = Column(String)
    dataset_type = Column(String)
    goal = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())