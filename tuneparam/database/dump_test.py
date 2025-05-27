from tuneparam.database.db import SessionLocal, Base, engine
from tuneparam.database.schema import User, Model, TrainingLog
from tuneparam.database.service.dao import user_crud
from tuneparam.database.service.dao import model_crud
from tuneparam.database.service.dao import create_training_log
from tuneparam.database.CONT import *

# DB 테이블 생성 (이미 존재하면 건너뜀)
Base.metadata.create_all(bind=engine)

# Training 로그 10개 생성

# 세션 생성
db = SessionLocal()

# 데이터 삽입 로직
try:
    # 유저 생성
    user = user_crud.create_user(db=db, user_data=TEST_USER)

    # 모델 2개 생성
    model1 = model_crud.create_model_for_user(db=db, username=user.username, model_data=TEST_MODEL)
    model2 = model_crud.create_model_for_user(db=db, username=user.username, model_data=TEST_MODEL2)

    # 모델 1에 로그 10개 삽입
    for log_data in TRAINING_LOGS:
        create_training_log(db=db, model_id=model1.id, log_data=log_data)

    # 모델 2에 하나만 삽입
    create_training_log(db=db, model_id=model2.id, log_data=TEST_LOG1)

    db.commit()
    print("✅ 테스트 데이터 삽입 완료")

except Exception as e:
    db.rollback()
    print("❌ 에러 발생:", e)

finally:
    db.close()
