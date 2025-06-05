import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def retrain_mobilenet(X_train_images, y_train_labels, gpt_output):
    from tuneparam.gui.main import launch_experiment
    rec = gpt_output["recommendations"]

    # ----------- 모델 생성 파라미터 -----------
    model_kwargs = dict(
        input_shape=(32, 32, 3),
        alpha=rec["alpha"] if rec["alpha"] else 1.0,
        minimalistic=rec["minimalistic"],
        weights="imagenet",
        include_top=False,
        input_tensor=rec["input_tensor"],
        classifier_activation=rec["classifier_activation"],
        include_preprocessing=rec["include_preprocessing"],
        pooling=None,
    )
    # None 제거
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    num_classes = len(np.unique(y_train_labels))

    base_model = MobileNetV3Large(**model_kwargs)

    for layer in base_model.layers:
        layer.trainable = False

    # 베이스 모델 위에 새로운 분류 레이어를 추가
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # 특징 맵을 단일 벡터로 평균 풀링
    x = Dense(1024, activation='relu')(x) # 추가 완전 연결 레이어
    predictions = Dense(num_classes, activation='softmax')(x) # 최종 출력 레이어

    model = Model(inputs=base_model.input, outputs=predictions)

    # ----------- 컴파일 파라미터 -----------
    optimizer_name = str(rec["optimizer"]).lower()
    lr = rec["learning_rate"]

    if optimizer_name == "adamw":
        optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=rec.get("weight_decay", 1e-4))
    else:
        optimizer = Adam(learning_rate=lr)
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # ----------- fit(훈련) 파라미터 -----------
    callbacks = []
    if str(rec.get("callbacks", "")).lower() == "earlystopping":
        callbacks.append(EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True))

    fit_kwargs = dict(
        epochs=rec["epochs"],
        batch_size=rec["batch_size"],
        verbose=rec["verbose"],
        shuffle=rec["shuffle"],
        class_weight=rec["class_weight"],
        sample_weight=rec["sample_weight"],
        initial_epoch=rec["initial_epoch"],
        steps_per_epoch=rec["steps_per_epoch"],
        validation_steps=rec["validation_steps"],
        validation_batch_size=rec["validation_batch_size"],
        validation_freq=rec["validation_freq"],
        max_queue_size=rec["max_queue_size"],
        workers=rec["workers"],
        use_multiprocessing=rec["use_multiprocessing"],
        validation_split=rec.get("validation_split", 0.2)
    )
    fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

    launch_experiment(model, X_train_images, y_train_labels, training_params=fit_kwargs)