import sys
import os

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import keras

imdb = keras.datasets.imdb
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)


def build_and_compile_model(training_params: dict) -> Model:
    """
    training_params 딕셔너리 기반으로 LSTM 모델을 생성하고 컴파일.

    Parameters:
        training_params (dict): 하이퍼파라미터 및 모델 구성 딕셔너리

    Returns:
        model (Model): 컴파일된 Keras 모델
    """
    
    # 데이터 파라미터
    vocab_size = training_params.get("vocab_size", 10000)
    max_length = training_params.get("max_length", 500)
    embedding_dim = training_params.get("embedding_dim", 128)
    
    # LSTM 파라미터 추출
    units = training_params.get("units", 64)
    dropout = training_params.get("dropout", 0.0)
    recurrent_dropout = training_params.get("recurrent_dropout", 0.0)
    return_sequences = training_params.get("return_sequences", False)
    activation = training_params.get("activation", "tanh")
    recurrent_activation = training_params.get("recurrent_activation", "sigmoid")
    use_bias = training_params.get("use_bias", True)
    kernel_initializer = training_params.get("kernel_initializer", "glorot_uniform")
    recurrent_initializer = training_params.get("recurrent_initializer", "orthogonal")
    bias_initializer = training_params.get("bias_initializer", "zeros")
    unit_forget_bias = training_params.get("unit_forget_bias", True)
    kernel_regularizer = training_params.get("kernel_regularizer", None)
    recurrent_regularizer = training_params.get("recurrent_regularizer", None)
    bias_regularizer = training_params.get("bias_regularizer", None)
    activity_regularizer = training_params.get("activity_regularizer", None)
    kernel_constraint = training_params.get("kernel_constraint", None)
    recurrent_constraint = training_params.get("recurrent_constraint", None)
    bias_constraint = training_params.get("bias_constraint", None)
    implementation = training_params.get("implementation", 2)
    go_backwards = training_params.get("go_backwards", False)
    stateful = training_params.get("stateful", False)
    time_major = training_params.get("time_major", False)
    unroll = training_params.get("unroll", False)
    
    # 모델 구성 파라미터
    num_layers = training_params.get("num_layers", 1)
    bidirectional = training_params.get("bidirectional", False)
    dense_dropout = training_params.get("dense_dropout", 0.0)
    
    # 컴파일 파라미터
    learning_rate = training_params.get("learning_rate", 0.001)
    optimizer_name = training_params.get("optimizer", "adam").lower()
    momentum = training_params.get("momentum", 0.0)
    weight_decay = training_params.get("weight_decay", 0.0)
    label_smoothing = training_params.get("label_smoothing", 0.0)
    loss_name = training_params.get("loss", "binary_crossentropy")
    metrics = training_params.get("metrics", ["accuracy"])

    # LSTM 파라미터 딕셔너리
    lstm_model_params = {
        "units": 64,
        "dropout": dropout,
        "return_sequences": return_sequences,
        "activation": activation,
        "recurrent_activation": recurrent_activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "recurrent_initializer": recurrent_initializer,
        "bias_initializer": bias_initializer,
        "unit_forget_bias": unit_forget_bias,
        "kernel_regularizer": kernel_regularizer,
        "recurrent_regularizer": recurrent_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "recurrent_constraint": recurrent_constraint,
        "bias_constraint": bias_constraint,
        "go_backwards": go_backwards,
        "stateful": stateful,
        "time_major": time_major,
        "unroll": unroll
    }

    # 모델 구성
    inputs = Input(shape=(max_length,))
    
    # 임베딩 레이어
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)(inputs)
    
    # 다중 레이어 LSTM 구성
    for i in range(num_layers):
        if i < num_layers - 1:
            # 마지막 레이어가 아닌 경우 return_sequences=True
            current_lstm_params = lstm_model_params.copy()
            current_lstm_params["return_sequences"] = True
        else:
            # 마지막 레이어
            current_lstm_params = lstm_model_params.copy()
        
        # Bidirectional LSTM 사용 여부
        if bidirectional:
            x = Bidirectional(LSTM(**current_lstm_params))(x)
        else:
            x = LSTM(**current_lstm_params)(x)
    
    # Dense 레이어 전 드롭아웃
    if dense_dropout > 0:
        x = Dropout(dense_dropout)(x)
    
    # 출력 레이어 (이진 분류)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=predictions)
    
    # 옵티마이저 설정
    try:
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, momentum=momentum, decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    except Exception as e:
        optimizer = Adam(learning_rate=learning_rate)

    # 손실 함수 설정
    if loss_name == "binary_crossentropy":
        loss = BinaryCrossentropy(label_smoothing=label_smoothing)
    else:
        loss = loss_name
    
    # 모델 컴파일
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def test_lstm(training_params=None):
    """IMDB 영화 리뷰 감정 분석용 LSTM 테스트"""
    
    # ===== IMDB 데이터 로드 및 전처리 =====
    max_features = 10000  # 어휘 크기
    max_length = 500      # 최대 시퀀스 길이

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    # 시퀀스 패딩
    x_train = pad_sequences(x_train, maxlen=max_length)
    x_test = pad_sequences(x_test, maxlen=max_length)

    # 레이블을 float32로 변환
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # ===== 하이퍼파라미터 기본값 정의 =====
    default_params = {
        "vocab_size": max_features,
        "max_length": max_length,
        "embedding_dim": 128,
        "units": 128,
        "dropout": 0.3,
        "recurrent_dropout": 0.3,
        "num_layers": 1,
        "bidirectional": False,
        "dense_dropout": 0.5,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 10,
        "validation_split": 0.2,
    }

    # 외부에서 전달된 training_params가 있으면 덮어씌움
    if training_params is not None:
        default_params.update(training_params)
    training_params = default_params

    # 모델 생성
    model = build_and_compile_model(training_params)

    print("✅ LSTM 모델 구성 및 실험 실행 준비 완료.")
    print(f"훈련 데이터 형태: {x_train.shape}")
    print(f"테스트 데이터 형태: {x_test.shape}")
    print(f"클래스 분포 - 긍정: {np.sum(y_train)}, 부정: {len(y_train) - np.sum(y_train)}")

    return model, x_train, y_train, training_params
