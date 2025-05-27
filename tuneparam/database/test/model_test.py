import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tuneparam.database.db import Base
from tuneparam.database.schema import User, Model, TrainingLog
from tuneparam.database.CONT import TEST_USER, TEST_MODEL, TEST_UPDATE_MODEL, TEST_MODEL2
from tuneparam.database.service.dao import user_crud as crud_user
from tuneparam.database.service.dao import user_crud, model_crud, traininglog_curd

class TestModelCRUD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 인메모리 SQLite 데이터베이스 생성
        DATABASE_URL = "sqlite:///:memory:"
        cls.engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        self.session = self.SessionLocal()
        # 테스트 유저 미리 생성
        self.user = user_crud.create_user(db=self.session, user_data=TEST_USER)

    def tearDown(self):
        self.session.close()

    def test_create_model_for_user(self):
        model = model_crud.create_model_for_user(db=self.session, username=self.user.username, model_data=TEST_MODEL)
        model2 = model_crud.create_model_for_user(db=self.session, username=self.user.username, model_data=TEST_MODEL2)
        print(model2.model_size)
        print(model.model_size)

        self.assertIsNotNone(model.id)
        self.assertEqual(model.model_size, TEST_MODEL["model_size"])
        self.assertEqual(model.user_id, self.user.id)

    def test_get_model_by_user(self):
        model_crud.create_model_for_user(db=self.session, username=self.user.username, model_data=TEST_MODEL)
        models = model_crud.get_models_by_username(db=self.session, username=self.user.username)
        self.assertGreater(len(models), 0)

    def test_update_model(self):
        model_crud.create_model_for_user(db=self.session, username=self.user.username, model_data=TEST_MODEL)
        updated = model_crud.update_model_for_user(
            db=self.session,
            username=self.user.username,
            model_size=TEST_MODEL["model_size"],
            model_type=TEST_MODEL["model_type"],
            update_data=TEST_UPDATE_MODEL
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated.goal, TEST_UPDATE_MODEL["goal"])

    def test_delete_model(self):
        # 모델 생성
        model = model_crud.create_model_for_user(
            db=self.session,
            username=self.user.username,
            model_data=TEST_MODEL
        )

        # 모델 삭제 (username, model_size, model_type로 삭제)
        result = model_crud.delete_model_for_user(
            db=self.session,
            username=self.user.username,
            model_size=model.model_size,
            model_type=model.model_type
        )

        self.assertTrue(result)

        # 삭제 확인
        deleted = self.session.query(Model).filter_by(id=model.id).first()
        self.assertIsNone(deleted)