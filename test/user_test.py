import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from tuneparam.gui.main import launch_experiment
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from mobilenet import test_moblinet
from lstm import test_lstm
from resnet import test_resnet

# model, X_train, y_train, training_params = test_moblinet()
# model, X_train, y_train, training_params = test_lstm()
model, X_train, y_train, training_params = test_resnet()

launch_experiment(model, X_train[:100], y_train[:100], training_params=training_params)