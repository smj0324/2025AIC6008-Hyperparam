import sys
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow import keras

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

def test_lstm():
    # ===== 데이터 준비 (LSTM 입력 형태) =====
    # 시계열 샘플 100개, 각 샘플은 20타임스텝, 각 타임스텝은 8차원 벡터
    X_train_seq = np.random.random((100, 20, 8)).astype(np.float32)
    y_train_labels = np.random.randint(0, 3, 100)  # 클래스 3개짜리 분류 문제 (0, 1, 2)

    num_classes = len(np.unique(y_train_labels))

    # ===== LSTM 모델 구성 =====
    inputs = Input(shape=(20, 8))
    # LSTM units=16, cuDNN 사용 조건 모두 만족 (기본 옵션)
    x = LSTM(16)(inputs)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',  # y_train_labels가 정수 인코딩
        metrics=['accuracy']
    )

    training_hyperparams = {
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 0.001,
        "validation_split": 0.2
    }
    print("LSTM 모델 구성 및 실험 실행 코드 준비 완료.")
    return model, X_train_seq, y_train_labels, training_hyperparams

    launch_experiment(
        model, X_train_seq, y_train_labels,
        training_params=training_hyperparams
    )

    