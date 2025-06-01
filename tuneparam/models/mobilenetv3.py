import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping

def retrain_mobilenet(X_train_images, y_train_labels, gpt_output):
    from tuneparam.gui.main import launch_experiment
    rec = gpt_output["recommendations"]

    # ----------- 모델 생성 파라미터 -----------
    model_kwargs = dict(
        input_shape=(224, 224, 3),
        alpha=rec["alpha"] if rec["alpha"] else 1.0,
        minimalistic=rec["minimalistic"],
        include_top=rec["include_top"],   # 보통 False 후 Dense 붙이지만, True도 반영
        weights=rec["weights"] if rec["weights"] else None,
        input_tensor=rec["input_tensor"],
        pooling=rec["pooling"],
        classifier_activation=rec["classifier_activation"],
        include_preprocessing=rec["include_preprocessing"],
    )
    # None 제거
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    num_classes = len(np.unique(y_train_labels))

    base_model = MobileNetV3Small(**model_kwargs)

    if rec["include_top"]:
        # 이미 top까지 붙은 모델 반환 (Dense 생략)
        model = base_model
    else:
        x = base_model.output
        predictions = Dense(num_classes, activation=rec["classifier_activation"])(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    # ----------- 컴파일 파라미터 -----------
    optimizer_cls = AdamW if str(rec["optimizer"]).lower() == "adamw" else Adam
    optimizer = optimizer_cls(learning_rate=rec["learning_rate"])
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
        validation_split=rec["validation_split"],
        validation_data=rec["validation_data"],
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
        callbacks=callbacks if callbacks else None
    )
    # None 값은 fit에서 기본값 쓰도록 제거
    fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

    launch_experiment(model, X_train_images, y_train_labels, training_params=fit_kwargs)