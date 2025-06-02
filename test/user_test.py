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

db_path = DATABASE_URL
print(db_path)

if not os.path.exists(db_path):
    print(f"Database not found at {db_path} Creating database...")
    table_create.main()
    dump_test.main()
else:
    print(f"Database found at {db_path}.")
#
# model, X_train, y_train, training_params = test_moblinet()
# model, X_train, y_train, training_params = test_lstm()
model, X_train, y_train, training_params = test_resnet()

launch_experiment(model, X_train[:50], y_train[:50], training_params=training_params)