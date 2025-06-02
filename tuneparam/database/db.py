from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os
db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 사용할 사용자 suffix 설정 ('sg', 'mj', 'hm' 중 하나)
user_suffix = 'mj'

# 사용자 suffix별 DB 파일명 매핑
db_name_map = {
    'sg': 'my_database_sg.db',
    'mj': 'my_database_mj.db',
    'hm': 'my_database_hm.db',
}

# DB 파일명 결정 (예외 처리 포함)
try:
    db_name = db_name_map[user_suffix]
except KeyError:
    raise ValueError(f"Unknown user_suffix: {user_suffix}. Expected one of {list(db_name_map.keys())}")

DATABASE_URL_ORIGIN = os.path.join(db_dir, db_name)
DATABASE_URL = f"sqlite:///{DATABASE_URL_ORIGIN}"

print("DATABASE_URL:", DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
    future=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()