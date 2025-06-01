from test.resnet import build_and_compile_model
import sys
import os
from gui.main import launch_experiment

def retrain_resnet(X_train_images, y_train_labels, gpt_output):
    rec = gpt_output["recommendations"]
    model = build_and_compile_model(rec)

    launch_experiment(model, X_train_images, y_train_labels, training_params=rec)


