# main.py
from tuneparam.database.db import SessionLocal, Base, engine
from tuneparam.database.schema import models
from tuneparam.database.CONT import TEST_USER, TEST_UPDATE_USER
from tuneparam.database.service.user import crud
import unittest

# def create_user(session, user_data):
#     print("create_user")
#     new_user = crud.create_user(db=session, user_data=user_data)
#     assert new_user.id is not None
#     print(new_user)
#
#
# def get_user_by_id(session, user_name):
#     print("get_user_by_id")
#     user = crud.get_user_by_id(db=session, user_name=user_name)
#     assert user is not None
#     print("Successfully retrieved user:", user.username)
#
#
# def get_user_all(session):
#     print("get_user_all")
#     users = crud.get_all_users(db=session)
#     assert len(users) > 0
#     print("Successfully retrieved all users:", [user.username for user in users])
#
# def get_update_user(session, user_name):
#     print("get_update_user")
#     update_data = TEST_UPDATE_USER
#     updated_user = crud.update_user(db=session, user_name=user_name, update_data=update_data)
#     assert updated_user is not None
#     print("Successfully updated user:", updated_user.hardware)
#
# if __name__ == '__main__':
#     session = SessionLocal()
#     # create_user(session=session, user_data=TEST_USER)
#     # get_user_by_id(session=session, user_name=TEST_USER['username'])
#     get_user_all(session=session)
#     print("Origin : ", TEST_USER['hardware'])
#     get_update_user(session=session, user_name=TEST_USER['username'])

class TestUserCRUD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 메모리 DB 연결 세팅
        from tuneparam.database.db import Base, create_engine, sessionmaker

        DATABASE_URL = "sqlite:///:memory:"
        cls.engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)
    def setUp(self):
        self.session = self.SessionLocal()

    def tearDown(self):
        self.session.close()
    def test_create_user(self):
        user = crud.create_user(db=self.session, user_data=TEST_USER)
        print("User name : ", user.username)
        self.assertIsNotNone(user.id)
        self.assertEqual(user.username, TEST_USER["username"])

    def test_get_user_by_username(self):
        created_user = crud.create_user(db=self.session, user_data=TEST_USER)

        fetched_user = crud.get_user_by_username(db=self.session, user_name=created_user.username)
        self.assertIsNotNone(fetched_user)
        self.assertEqual(fetched_user.username, TEST_USER["username"])
    
    def test_get_all_users(self):
        crud.create_user(db=self.session, user_data=TEST_USER)
        users = crud.get_all_users(db=self.session)
        print(users)
        self.assertGreater(len(users), 0)
    #
    def test_update_user(self):
        user = crud.create_user(db=self.session, user_data=TEST_USER)
        updated_user = crud.update_user_by_username(db=self.session, user_name=user.username, update_data=TEST_UPDATE_USER)
        self.assertEqual(updated_user.hardware, TEST_UPDATE_USER["hardware"])
    #
    def test_delete_user(self):
        user = crud.create_user(db=self.session, user_data=TEST_USER)
        result = crud.delete_user_by_username(db=self.session, user_name=user.username)
        self.assertTrue(result)
        user = crud.get_user_by_username(db=self.session, user_name=user.username)
        self.assertIsNone(user)
