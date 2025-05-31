from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os
# 현재 파일 위치에서 최상위 디렉토리(예: 3단계 위) 가져오기
db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATABASE_URL = f"sqlite:///{db_dir}/my_database.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
    future=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()