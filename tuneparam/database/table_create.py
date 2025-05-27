from tuneparam.database.db import Base, engine
from tuneparam.database.schema import User, Model, TrainingLog

Base.metadata.create_all(bind=engine)