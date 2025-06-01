from tuneparam.database.db import Base, engine
from tuneparam.database.schema import User, Model, TrainingLog

def main():
    Base.metadata.create_all(bind=engine)
