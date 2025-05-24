import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

TEST_USER = {
        "username": "alice",
        "version": "1.0",
        "hardware": "A100",
        "model_size": "Base",
        "dataset_size": "Small",
        "model_type": "Resnet",
        "dataset_type": "Image",
        "goal": "Accuracy"
}

TEST_UPDATE_USER = {
        "username": "alice",
        "version": "1.0",
        "hardware": "4090",
        "model_size": "Base",
        "dataset_size": "Small",
        "model_type": "Resnet",
        "dataset_type": "Image",
        "goal": "Accuracy"
}