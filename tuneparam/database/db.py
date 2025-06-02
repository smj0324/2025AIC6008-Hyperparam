from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os
db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

user_suffix = 'sg'  # 'sg', 'mj', 'hm' 중 선택

# suffix에 따라 파일명 분기
if user_suffix == 'sg':
    db_name = 'my_database_sg.db'
elif user_suffix == 'mj':
    db_name = 'my_database_mj.db'
elif user_suffix == 'hm':
    db_name = 'my_database_hm.db'
else:
    raise ValueError(f"Unknown user_suffix: {user_suffix}")

# 최종 DB 경로 및 SQLAlchemy용 URL 구성
db_path = os.path.join(db_dir, db_name)
DATABASE_URL = f"sqlite:///{db_path}"

print("DATABASE_URL:", DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
    future=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()