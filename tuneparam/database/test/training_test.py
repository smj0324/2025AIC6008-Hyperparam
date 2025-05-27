import unittest
from tuneparam.database.db import Base, create_engine, sessionmaker
from tuneparam.database.schema import User, Model, TrainingLog
from tuneparam.database.service.dao import user_crud, model_crud, traininglog_curd

import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tuneparam.database.db import Base
from tuneparam.database.service.dao import user_crud
from tuneparam.database.service.dao import model_crud

from tuneparam.database.service.dao import (
    create_training_log,
    get_training_logs_by_model,
    delete_training_logs_by_model
)
from tuneparam.database.CONT import (
    TEST_USER, TEST_MODEL, TEST_LOG1, TEST_LOG2, TEST_LOG3
)


class TestTrainingLogCRUD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DATABASE_URL = "sqlite:///:memory:"
        cls.engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        self.session = self.SessionLocal()
        self.user = user_crud.create_user(db=self.session, user_data=TEST_USER)
        self.model = model_crud.create_model_for_user(db=self.session, username=self.user.username, model_data=TEST_MODEL)

    def tearDown(self):
        self.session.close()

    def test_create_training_log(self):
        log = create_training_log(db=self.session, model_id=self.model.id, log_data=TEST_LOG1)
        print(log.epoch)
        print(log.loss)
        print(log.id)
        self.assertIsNotNone(log.id)
        self.assertEqual(log.epoch, TEST_LOG1["epoch"])
        self.assertAlmostEqual(log.loss, TEST_LOG1["loss"], places=4)

    def test_get_training_logs_by_model(self):
        create_training_log(db=self.session, model_id=self.model.id, log_data=TEST_LOG1)
        create_training_log(db=self.session, model_id=self.model.id, log_data=TEST_LOG2)
        logs = get_training_logs_by_model(db=self.session, model_id=self.model.id)
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0].epoch, TEST_LOG1["epoch"])

    def test_delete_training_logs_by_model(self):
        create_training_log(db=self.session, model_id=self.model.id, log_data=TEST_LOG1)
        create_training_log(db=self.session, model_id=self.model.id, log_data=TEST_LOG2)
        count_deleted = delete_training_logs_by_model(db=self.session, model_id=self.model.id)
        self.assertEqual(count_deleted, 2)
        remaining_logs = get_training_logs_by_model(db=self.session, model_id=self.model.id)
        self.assertEqual(len(remaining_logs), 0)
