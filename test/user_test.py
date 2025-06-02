import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from tuneparam.database.db import DATABASE_URL_ORIGIN
from tuneparam.gui.main import launch_experiment
from mobilenet import test_moblinet
from random_search.mobilenet import start_moblinet_random_search
from random_search.lstm import start_lstm_random_search
from lstm import test_lstm
from tuneparam.models import test_resnet, test_random_search_resnet
from tuneparam.database import table_create, dump_test

db_path = DATABASE_URL_ORIGIN

if not os.path.exists(db_path):
    print("Database not found. Creating database...")
    table_create.main()
    dump_test.main()
else:
    print("Database found.")


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

mv_base_model_params = {
    "input_shape": (224, 224, 3),
    "alpha": 1.0,           
    "minimalistic": True,
    "include_top": False,
    "weights": "imagenet",
    "input_tensor": None,
    "pooling": "avg",
    "classifier_activation": "softmax",
    "include_preprocessing": True
}

# model, X_train, y_train, training_params = test_moblinet(mv_base_model_params)
model, X_train, y_train, training_params = test_lstm()
# model, X_train, y_train, training_params = test_resnet()
#model, X_train, y_train, training_params = test_random_search_resnet()

#model, X_train, y_train, training_params = start_moblinet_random_search(mv_base_model_params)
#model, X_train, y_train, training_params = start_lstm_random_search(mv_base_model_params)
launch_experiment(model, X_train, y_train, training_params=training_params)