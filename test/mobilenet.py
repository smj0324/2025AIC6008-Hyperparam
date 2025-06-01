import sys
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.applications import MobileNetV3Small

current_script_dir = os.path.dirname(os.path.abspath(__file__))

project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..')) # '..'는 'test' 디렉토리의 상위, 즉 'aisa'
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

from tuneparam.gui.main import launch_experiment


# ===== 데이터 준비 (MobileNetV3Small 입력 형태에 맞게 수정) =====
X_train_images = np.random.randint(0, 256, size=(100, 224, 224, 3), dtype=np.uint8)
y_train_labels = np.random.randint(0, 2, 100)
num_classes = len(np.unique(y_train_labels)) # 실제 클래스 수 계산

# ===== MobileNetV3Small 설정 및 모델 구성 =====
mv_base_model_params = {
    "input_shape": (224, 224, 3), # 입력 이미지 형태 명시
    "alpha": 1.0, # 네트워크 폭 조절 (기본값)
    "minimalistic": False, # 고급 블록 사용 여부 (기본값)
    "include_top": False, # 전이 학습을 위해 최상위 분류 레이어 제거
    "weights": "imagenet", # ImageNet 사전 훈련 가중치 로드
    "input_tensor": None, # Keras 텐서를 입력으로 사용하지 않음
    "pooling": "avg", # 특징 추출 후 Global Average Pooling 적용하여 2D 벡터 생성
    "classifier_activation": "softmax", # include_top=False 일 때는 이 인자 무시됨
    "include_preprocessing": True # Keras가 제공하는 전처리 레이어 포함 (모델이 [0-255] 입력을 기대)
}

base_model = MobileNetV3Small(**mv_base_model_params)

x = base_model.output

predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # y_train_labels가 정수 인코딩되어 있다면
              metrics=['accuracy'])


training_hyperparams = {
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "validation_split": 0.2
}

launch_experiment(model, X_train_images, y_train_labels, training_params=training_hyperparams)

print("모델 구성 및 실험 실행 코드 준비 완료.")