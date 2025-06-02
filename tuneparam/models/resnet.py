import sys
import os
from gui.main import launch_experiment
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.optimizers.legacy import SGD
from copy import deepcopy
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def generate_random_hyperparams():
    learning_rate = 10 ** random.uniform(-4, -1)  # 0.0001 ~ 0.1 (log scale)
    batch_size = random.choice([32, 64, 128, 256])
    epochs = random.choice([20, 30, 40, 50, 70, 150, 200])
    optimizer = random.choice(["adam", "sgd"])
    data_augmentation = random.choice([True, False])  # 증강 켜고 끄기 랜덤

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": optimizer,
        "data_augmentation": data_augmentation,
    }

def conv_block(x, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def residual_block(x, filters, stride=1, downsample=False):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if downsample:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet(input_shape, num_classes, block_counts):
    inputs = layers.Input(shape=input_shape)
    x = conv_block(inputs, 64, kernel_size=3, stride=1)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    filters = 64
    for i, blocks in enumerate(block_counts):
        for j in range(blocks):
            stride = 1
            downsample = False
            if j == 0 and i != 0:
                stride = 2
                downsample = True
            x = residual_block(x, filters, stride=stride, downsample=downsample)
        filters *= 2
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def get_resnet_model(model_name, input_shape=(32, 32, 3), num_classes=10):
    model_name = model_name.lower()
    if model_name == 'resnet18':
        return build_resnet(input_shape, num_classes, [2, 2, 2, 2])
    elif model_name == 'resnet34':
        return build_resnet(input_shape, num_classes, [3, 4, 6, 3])
    elif model_name == 'resnet50':
        return tf.keras.applications.ResNet50(input_shape=input_shape, weights=None, classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def build_and_compile_model(training_params: dict, train_data, train_label):
    """
    training_params 딕셔너리 기반으로 ResNet 모델 생성 및 컴파일.
    data_augmentation 옵션에 따라 증강된 데이터도 반환.

    Parameters:
        training_params (dict): 하이퍼파라미터 및 모델 구성 딕셔너리
        train_data (np.array): 학습용 이미지 데이터
        train_label (np.array): 학습용 레이블(one-hot)

    Returns:
        model (Model): 컴파일된 Keras 모델
        x_train_out (np.array): 증강된 또는 원본 학습 데이터
        y_train_out (np.array): 증강된 또는 원본 학습 레이블
    """

    model_name = training_params.get("model_name", "resnet18")
    input_shape = training_params.get("input_shape", (32, 32, 3))
    num_classes = training_params.get("num_classes", 100)

    learning_rate = training_params.get("learning_rate", 0.001)
    optimizer_name = training_params.get("optimizer", "adam").lower()
    momentum = training_params.get("momentum", 0.0)
    weight_decay = training_params.get("weight_decay", 0.0)
    label_smoothing = training_params.get("label_smoothing", 0.0)
    loss_name = training_params.get("loss", "categorical_crossentropy")
    metrics = training_params.get("metrics", ["accuracy"])
    augment = training_params.get("data_augmentation", False)

    # 모델 생성
    model = get_resnet_model(model_name, input_shape, num_classes)
    print(model.summary())

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
    if loss_name == "categorical_crossentropy":
        loss = CategoricalCrossentropy(label_smoothing=label_smoothing)
    else:
        loss = loss_name

    # 모델 컴파일
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # 데이터 증강 적용
    if augment:
        augment_params = {
            'rotation_range': 15,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'horizontal_flip': True,
            'zoom_range': 0.1
        }
        datagen = ImageDataGenerator(**augment_params)
        datagen.fit(train_data)
        batch_size = train_data.shape[0]
        augmented_iter = datagen.flow(train_data, train_label, batch_size=batch_size, shuffle=False)
        x_train_aug, y_train_aug = next(augmented_iter)
        return model, x_train_aug, y_train_aug
    else:
        return model, train_data, train_label


def test_resnet():
    training_params = {
        "model_size": "resnet50",
        "input_shape": (32, 32, 3),
        "num_classes": 100,
        "epochs": 20,
        "batch_size": 64,
        "shuffle": True,
        "verbose": 1,
        "initial_epoch": 0,
        "validation_freq": 1,
        "validation_split": 0.2,
        "max_queue_size": 10,
        "workers": 1,
        "use_multiprocessing": False,

        "optimizer": "adam",
        "learning_rate": 0.1,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "dropout_rate": 0.0,
        "label_smoothing": 0.0,
        "scheduler": None,
        "data_augmentation": False,
        "batch_normalization": True,
        "initialization": "he_normal",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"]
    }

    num_classes = 100

    (x_train, y_train), (_, _) = cifar100.load_data()
    y_train = to_categorical(y_train, num_classes)
    x_train = x_train.astype('float32') / 255.0
    model, x_train, y_train = build_and_compile_model(training_params, x_train, y_train)

    return model, x_train, y_train, training_params


def test_random_search_resnet():
    base_params = {
        "model_size": "resnet50",
        "input_shape": (32, 32, 3),
        "num_classes": 100,
        "epochs": 20,
        "batch_size": 64,
        "shuffle": True,
        "verbose": 1,
        "initial_epoch": 0,
        "validation_freq": 1,
        "validation_split": 0.2,
        "max_queue_size": 10,
        "workers": 1,
        "use_multiprocessing": False,
        "optimizer": "adam",
        "learning_rate": 0.1,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "dropout_rate": 0.0,
        "label_smoothing": 0.0,
        "scheduler": None,
        "data_augmentation": False,
        "batch_normalization": True,
        "initialization": "he_normal",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"]
    }

    selected_variant = generate_random_hyperparams()
    training_params = deepcopy(base_params)
    training_params.update(selected_variant)

    print("Selected hyperparameter combination:")
    print(selected_variant)

    # 데이터 준비
    num_classes = training_params["num_classes"]
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    y_train = to_categorical(y_train, num_classes)
    x_train = x_train.astype('float32') / 255.0

    # 모델 빌드 및 컴파일
    model, x_train, y_train = build_and_compile_model(training_params, x_train, y_train)

    return model, x_train, y_train, training_params

def retrain_resnet(X_train_images, y_train_labels, gpt_output):
    rec = gpt_output["recommendations"]
    model, X_train_images, y_train_labels = build_and_compile_model(rec, X_train_images, y_train_labels)
    launch_experiment(model, X_train_images, y_train_labels, training_params=rec)


