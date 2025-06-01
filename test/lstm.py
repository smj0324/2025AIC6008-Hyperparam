import sys
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, Dropout
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

    # ===== LSTM 설정 및 모델 구성 =====
    lstm_model_params = {
        "units": 16,  # LSTM 유닛 수
        "dropout": 0.0,  # 입력 드롭아웃
        "recurrent_dropout": 0.0,  # 순환 드롭아웃
        "return_sequences": False,  # 시퀀스 반환 여부
        "activation": "tanh",  # 주 활성화 함수
        "recurrent_activation": "sigmoid",  # 순환 활성화 함수
        "use_bias": True,  # 편향 사용 여부
        "kernel_initializer": "glorot_uniform",  # 입력 가중치 초기화
        "recurrent_initializer": "orthogonal",  # 순환 가중치 초기화
        "bias_initializer": "zeros",  # 편향 초기화
        "unit_forget_bias": True,  # forget gate 편향 초기화
        "kernel_regularizer": None,  # 입력 가중치 정규화
        "recurrent_regularizer": None,  # 순환 가중치 정규화
        "bias_regularizer": None,  # 편향 정규화
        "activity_regularizer": None,  # 활성화 정규화
        "kernel_constraint": None,  # 입력 가중치 제약
        "recurrent_constraint": None,  # 순환 가중치 제약
        "bias_constraint": None,  # 편향 제약
        "implementation": 2,  # 구현 방식 (1 또는 2)
        "go_backwards": False,  # 역방향 처리 여부
        "stateful": False,  # 상태 유지 여부
        "time_major": False,  # 시간 축 우선 여부
        "unroll": False  # 루프 펼침 여부
    }

    # 모델 구성 파라미터
    model_config_params = {
        "num_layers": 1,  # LSTM 레이어 수
        "bidirectional": False,  # 양방향 LSTM 여부
        "sequence_length": 20,  # 시퀀스 길이
        "feature_dim": 8,  # 특징 차원
        "dense_dropout": 0.0  # Dense 레이어 전 드롭아웃
    }

    inputs = Input(shape=(model_config_params["sequence_length"], model_config_params["feature_dim"]))
    
    x = inputs
    # 다중 레이어 LSTM 구성
    for i in range(model_config_params["num_layers"]):
        if i < model_config_params["num_layers"] - 1:
            # 마지막 레이어가 아닌 경우 return_sequences=True
            current_lstm_params = lstm_model_params.copy()
            current_lstm_params["return_sequences"] = True
        else:
            # 마지막 레이어
            current_lstm_params = lstm_model_params.copy()
        
        # Bidirectional LSTM 사용 여부
        if model_config_params["bidirectional"]:
            x = Bidirectional(LSTM(**current_lstm_params))(x)
        else:
            x = LSTM(**current_lstm_params)(x)
    
    # Dense 레이어 전 드롭아웃
    if model_config_params["dense_dropout"] > 0:
        x = Dropout(model_config_params["dense_dropout"])(x)
    
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

    # launch_experiment(
    #     model, X_train_seq, y_train_labels,
    #     training_params=training_hyperparams
    # )