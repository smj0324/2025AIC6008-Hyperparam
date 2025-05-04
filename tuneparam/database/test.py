# main.py
from tuneparam.database.db import SessionLocal, Base, engine
from tuneparam.database.schema import models
from tuneparam.database.service.crud import (
    create_student, read_students, update_student, delete_student
)

# 테이블 생성 (처음 한 번만)
Base.metadata.create_all(bind=engine)

# 세션 선언 (한 번만)
session = SessionLocal()

# CRUD 사용 예시
create_student(session, "홍길동", 20)
create_student(session, "김영희", 22)

print("== 전체 조회 ==")
for s in read_students(session):
    print(s.id, s.name, s.age)

update_student(session, 1, new_name="홍길순")
delete_student(session, 2)

print("== 최종 조회 ==")
for s in read_students(session):
    print(s.id, s.name, s.age)

# 세션 닫기
session.close()
