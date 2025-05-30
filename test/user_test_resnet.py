import sys
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 현재 파일의 상위 디렉토리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 필요한 외부 모듈 임포트
from models.resnet import get_resnet_model
from tuneparam.gui.main import launch_experiment

# 하이퍼파라미터 설정
model_name = 'resnet18'  # 'resnet18' | 'resnet34' | 'resnet50'
batch_size = 64
epochs = 10
input_shape = (32, 32, 3)
num_classes = 100

# 데이터셋 로딩 및 전처리
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 모델 생성 및 컴파일
model: Model = get_resnet_model(model_name, input_shape, num_classes)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 학습 파라미터 구성
training_params = {
    "epochs": epochs,
    "batch_size": batch_size,
    "shuffle": True,
    "verbose": 1,
    "initial_epoch": 0,
    "validation_freq": 1,
    "validation_split": 0.2,     # 별도 검증셋이 없으므로 사용
    "max_queue_size": 10,
    "workers": 1,
    "use_multiprocessing": False
}

# launch_experiment 실행
launch_experiment(model, x_train, y_train, training_params=training_params)
