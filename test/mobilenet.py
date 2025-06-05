import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

# --- 데이터 전처리 함수 ---
def preprocess(image, label, img_size=32):
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    label = tf.squeeze(label)
    return image, label

# --- MobileNetV3 학습 및 모델 반환 함수 ---
def test_moblinet(
    mv_base_model_params,
    epochs=1,
    batch_size=32,
    learning_rate=0.05,
    img_size=32,
):
    """
    MobileNetV3Small 모델을 사용하여 CIFAR-10 데이터셋을 학습하고 관련 정보를 반환합니다.

    Args:
        mv_base_model_params (dict): MobileNetV3Small 기본 모델에 전달할 파라미터 (예: weights, minimalistic).
        epochs (int): 학습할 에포크 수.
        batch_size (int): 배치 크기.
        learning_rate (float): Adam 옵티마이저의 학습률.
        img_size (int): 이미지 리사이징 크기 (정사각형).

    Returns:
        tuple: (model, train_ds, y_train, training_hyperparams)
            - model (tf.keras.Model): 컴파일된 Keras 모델.
            - train_ds (tf.data.Dataset): 전처리된 학습 데이터셋.
            - y_train (numpy.ndarray): 원래 학습 레이블 (참고용).
            - training_hyperparams (dict): 학습 하이퍼파라미터.
    """

    # 1. 데이터 로드
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_classes = 10

    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.int64) # sparse_categorical_crossentropy를 위해 int64
    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.int64)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train_tf, y_train_tf))

    train_ds = train_ds.map(lambda x, y: preprocess(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE) \
                       .shuffle(10000) \
                       .batch(batch_size) \
                       .prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_test_tf, y_test_tf))
    val_ds = val_ds.map(lambda x, y: preprocess(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE) \
                   .batch(batch_size) \
                   .prefetch(tf.data.AUTOTUNE)

    # 2. 모델 로드 (전이 학습)
    # input_shape를 mv_base_model_params에 명시적으로 설정
    mv_base_model_params['input_shape'] = (img_size, img_size, 3)
    base_model = MobileNetV3Large(**mv_base_model_params)

    # 베이스 모델의 레이어들을 동결
    for layer in base_model.layers:
        layer.trainable = False

    # 베이스 모델 위에 새로운 분류 레이어를 추가
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # 특징 맵을 단일 벡터로 평균 풀링
    x = Dense(1024, activation='relu')(x) # 추가 완전 연결 레이어
    predictions = Dense(num_classes, activation='softmax')(x) # 최종 출력 레이어

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3. 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 학습 하이퍼파라미터 딕셔너리
    training_hyperparams = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_data": val_ds
    }

    return model, train_ds, y_train, training_hyperparams