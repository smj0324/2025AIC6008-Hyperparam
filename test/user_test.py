import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tuneparam.gui.main import launch_experiment
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from mobilenet import test_moblinet
from lstm import test_lstm

from tuneparam.models import test_resnet
from tuneparam.database import table_create, dump_test
from tuneparam.database.db import DATABASE_URL

# __file__을 기준으로 절대경로 생성
# 현재 파일 기준 두 단계 위로 이동 후, tuneparam/my_database.db 경로 조합
db_path = DATABASE_URL
print(db_path)

# # suffix에 따라 파일명 분기
# if user_suffix == 'sg':
#     db_name = 'my_database_sg.db'
# elif user_suffix == 'mj':
#     db_name = 'my_database_mj.db'
# elif user_suffix == 'hm':
#     db_name = 'my_database_hm.db'
# else:
#     raise ValueError(f"Unknown user_suffix: {user_suffix}")

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}. Creating database...")
    table_create.main()  # 실제 함수명에 따라 수정
    dump_test.main()     # 실제 함수명에 따라 수정
else:
    print(f"Database found at {db_path}.")
#
# model, X_train, y_train, training_params = test_moblinet()
# model, X_train, y_train, training_params = test_lstm()
model, X_train, y_train, training_params = test_resnet()

launch_experiment(model, X_train[:50], y_train[:50], training_params=training_params)