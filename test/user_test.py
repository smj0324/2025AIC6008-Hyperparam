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

db_path = os.path.join(os.path.dirname(__file__), '..', 'tuneparam', 'my_database.db')

if not os.path.exists(db_path):
    print("Database not found. Creating database...")
    table_create.main()  # 또는 table_create.create_tables() 등, 함수명을 실제 정의에 맞게 수정
    dump_test.main()     # 마찬가지로 정의에 맞게 수정
else:
    print("Database found.")

# model, X_train, y_train, training_params = test_moblinet()
# model, X_train, y_train, training_params = test_lstm()
model, X_train, y_train, training_params = test_resnet()

launch_experiment(model, X_train, y_train, training_params=training_params)