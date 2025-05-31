from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# todo 이부분경로 각자 최상위 디렉토리로 설정해주세요~
db_dir = "C:/Users/a/Desktop/code/team/2025AIC6008-Hyperparam"

# 데이터베이스 파일 경로
DATABASE_URL = f"sqlite:///{db_dir}/my_database.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
    future=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()