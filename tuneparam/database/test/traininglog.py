import unittest
from tuneparam.database.db import Base, create_engine, sessionmaker
from tuneparam.database.schema.models import User, Model, TrainingLog
from tuneparam.database.service.user import user_crud
from tuneparam.database.service.model import model_crud
from tuneparam.database.service.traininglog_crud import create_training_log, get_training_logs_by_model, delete_training_logs_by_model

TEST_USER = {
    "username": "alice",
    "version": "1.0",
    "hardware": "A100"
}

TEST_MODEL = {
    "model_size": "Base",
    "model_type": "Resnet",
    "dataset_size": "Small",
    "dataset_type": "Image",
    "goal": "Accuracy"
}

TEST_LOG = {
    "epoch": 0,
    "loss": 0.7178,
    "accuracy": 0.4875,
    "val_loss": 0.7261,
    "val_accuracy": 0.5
}

class TestTrainingLogCRUD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DATABASE_URL = "sqlite:///:memory:"
        cls.engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        self.session = self.SessionLocal()
        # 유저, 모델 생성
        self.user = user_crud.create_user(db=self.session, user_data=TEST_USER)
        self.model = model_crud.create_model_for_user(db=self.session, username=self.user.username, model_data=TEST_MODEL)

    def tearDown(self):
        self.session.close()

    def test_create_training_log(self):
        log = create_training_log(db=self.session, model_id=self.model.id, log_data=TEST_LOG)
        self.assertIsNotNone(log.id)
        self.assertEqual(log.epoch, TEST_LOG["epoch"])
        self.assertEqual(log.loss, TEST_LOG["loss"])

    def test_get_training_logs_by_model(self):
        create_training_log(db=self.session, model_id=self.model.id, log_data=TEST_LOG)
        logs = get_training_logs_by_model(db=self.session, model_id=self.model.id)
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0].epoch, TEST_LOG["epoch"])

    def test_delete_training_logs_by_model(self):
        create_training_log(db=self.session, model_id=self.model.id, log_data=TEST_LOG)
        count = delete_training_logs_by_model(db=self.session, model_id=self.model.id)
        self.assertEqual(count, 1)
        logs_after_delete = get_training_logs_by_model(db=self.session, model_id=self.model.id)
        self.assertEqual(len(logs_after_delete), 0)
