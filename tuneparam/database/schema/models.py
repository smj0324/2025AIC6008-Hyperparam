# models.py
from sqlalchemy import Column, Integer, String
from tuneparam.database.db  import Base

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)
