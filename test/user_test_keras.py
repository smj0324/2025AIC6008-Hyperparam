import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tuneparam.gui.main import launch_experiment
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

X_train = np.random.rand(100, 8)
y_train = np.random.randint(0, 2, 100)

model = Sequential([
    Dense(16, activation='relu', input_shape=(8,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

training_params = {
    "epochs": 100,                      # 전체 학습 반복 수
    "batch_size": 16,                 # 미니배치 크기
    "validation_split": 0.2,          # 학습 데이터의 20%를 검증에 사용
    "shuffle": True,                  # 에포크마다 데이터 셔플
    "verbose": 1,                     # 진행상황 출력(0, 1, 2)
    "initial_epoch": 0,               # 학습 시작 에포크 (resume 시)
    "validation_freq": 1,             # 몇 에포크마다 검증할지
    # "validation_data": (X_val, y_val), # 별도 검증셋을 직접 줄 때(선택)
    # "class_weight": {0: 1, 1: 2},    # 클래스 불균형시 가중치
    # "sample_weight": np.ones(len(X_train)), # 샘플별 가중치
    "max_queue_size": 10,             # generator 사용시
    "workers": 1,                     # 데이터 불러오기 멀티프로세스 수
    "use_multiprocessing": False,      # 멀티프로세싱 사용 여부
    # "validation_steps": None,        # generator 검증시
    # "steps_per_epoch": None,         # generator 학습시
    # "validation_batch_size": None,   # 검증시 배치 크기
}

launch_experiment(model, X_train, y_train, training_params=training_params)