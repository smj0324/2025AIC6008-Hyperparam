import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

TEST_USER = {
    "username": "alice",
    "version": "1.0",
    "hardware": "A100"
}
TEST_MODEL = {
    "model_size": "Base",
    "dataset_size": "Small",
    "model_type": "Resnet",
    "dataset_type": "Image",
    "goal": "Accuracy"
}

TEST_MODEL2 = {
    "model_size": "Small",
    "dataset_size": "Small",
    "model_type": "Resnet",
    "dataset_type": "Image",
    "goal": "Accuracy"
}

TEST_UPDATE_USER = {
    "username": "alice",
    "version": "1.1",          # 버전 업그레이드
    "hardware": "RTX4090"      # 하드웨어 변경
}

TEST_UPDATE_MODEL = {
    "model_size": "Base",
    "dataset_size": "Full",       # 데이터셋 크기 변경
    "model_type": "Resnet",
    "dataset_type": "Image",
    "goal": "Speed"               # 목표 변경
}



TEST_LOG1 = {
    "epoch": 0,
    "loss": 0.7178,
    "accuracy": 0.4875,
    "val_loss": 0.7261,
    "val_accuracy": 0.5
}

TEST_LOG2 = {
    "epoch": 0,
    "loss": 0.7178,
    "accuracy": 0.4875,
    "val_loss": 0.7261,
    "val_accuracy": 0.5
}


TEST_LOG3 = {
    "epoch": 0,
    "loss": 0.7178,
    "accuracy": 0.4875,
    "val_loss": 0.7261,
    "val_accuracy": 0.5
}
