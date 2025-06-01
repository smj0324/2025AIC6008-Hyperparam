import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from tuneparam.gui.main import launch_experiment
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from mobilenet import MobileNetV3Small

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
# launch_experiment(model, X_train, y_train, training_params=training_params)

X_train_images = np.random.randint(0, 256, size=(100, 224, 224, 3), dtype=np.uint8)
y_train_labels = np.random.randint(0, 2, 100) # 2개 클래스 (0 또는 1)

mv_training_params = {
    "input_shape": (224, 224, 3), # 이미지 입력 형태 명시
    "alpha": 1.0,
    "minimalistic": False,
    "include_top": False, # 전이 학습을 위해 최상위 분류 레이어 제거
    "weights": "imagenet", # ImageNet 사전 훈련 가중치 로드
    "input_tensor": None,
    "pooling": "avg", # 특징 추출 후 Global Average Pooling 적용하여 2D 벡터 생성
    "classifier_activation": "softmax", # 이 인자는 include_top=False 일 때 무시됨
    "include_preprocessing": True, # Keras가 제공하는 전처리 레이어 포함 (입력 [0-255])
    "name": "MobileNetV3Small_Base"
}

base_model = MobileNetV3Small(**mv_training_params)
x = base_model.output

num_classes = len(np.unique(y_train_labels))
predictions = Dense(num_classes, activation='softmax')(x) # 2개 클래스를 위한 Softmax

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

training_hyperparams = {
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001
}


launch_experiment(model, X_train_images, y_train_labels, training_params=training_hyperparams)
