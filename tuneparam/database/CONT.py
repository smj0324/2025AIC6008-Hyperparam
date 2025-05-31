import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

TEST_USER = {
    "username": "alice",
    "hardware": "A100"
}
TEST_MODEL = {
    "model_size": "Base",
    "dataset_size": "Small",
    "model_type": "Resnet",
    "dataset_type": "Image",
    "goal": "Accuracy",
    "version": "2.0",
"total_epoch" : 100
}

TEST_MODEL2 = {
    "model_size": "Small",
    "dataset_size": "Small",
    "model_type": "Resnet",
    "dataset_type": "Image",
    "goal": "Accuracy",
    "version": "1.0",
    "total_epoch" : 200
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
    "loss": 1.234,
    "accuracy": 0.420,
    "val_loss": 1.210,
    "val_accuracy": 0.430
}


TRAINING_LOGS = [
    {"epoch": 0, "loss": 1.234, "accuracy": 0.420, "val_loss": 1.210, "val_accuracy": 0.430},
    {"epoch": 1, "loss": 1.001, "accuracy": 0.505, "val_loss": 1.010, "val_accuracy": 0.490},
    {"epoch": 2, "loss": 0.880, "accuracy": 0.565, "val_loss": 0.930, "val_accuracy": 0.525},
    {"epoch": 3, "loss": 0.775, "accuracy": 0.610, "val_loss": 0.880, "val_accuracy": 0.550},
    {"epoch": 4, "loss": 0.701, "accuracy": 0.655, "val_loss": 0.850, "val_accuracy": 0.570},
    {"epoch": 5, "loss": 0.640, "accuracy": 0.700, "val_loss": 0.830, "val_accuracy": 0.585},
    {"epoch": 6, "loss": 0.590, "accuracy": 0.735, "val_loss": 0.810, "val_accuracy": 0.600},
    {"epoch": 7, "loss": 0.550, "accuracy": 0.760, "val_loss": 0.795, "val_accuracy": 0.615},
    {"epoch": 8, "loss": 0.515, "accuracy": 0.785, "val_loss": 0.785, "val_accuracy": 0.625},
    {"epoch": 9, "loss": 0.480, "accuracy": 0.800, "val_loss": 0.770, "val_accuracy": 0.630},
]
