from tuneparam.database.db import Base, engine
from tuneparam.database.schema import models  # 꼭 import 필요

Base.metadata.create_all(bind=engine)