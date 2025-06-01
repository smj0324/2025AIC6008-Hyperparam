import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam, AdamW, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

def retrain_lstm(X_train_seq, y_train_labels, gpt_output):
    from tuneparam.gui.main import launch_experiment
    rec = gpt_output["recommendations"]

    num_classes = len(np.unique(y_train_labels))

    # ----------- LSTM 모델 파라미터 (test_lstm과 동일한 구조) -----------
    lstm_model_params = dict(
        units=rec.get("hidden_size", 16),  # 스키마의 "hidden_size"를 "units"로 매핑
        dropout=rec.get("dropout", 0.0),
        recurrent_dropout=rec.get("recurrent_dropout", 0.0),
        return_sequences=rec.get("return_sequences", False),
        activation=rec.get("activation", "tanh"),
        recurrent_activation=rec.get("recurrent_activation", "sigmoid"),
        use_bias=rec.get("use_bias", True),
        kernel_initializer=rec.get("kernel_initializer", "glorot_uniform"),
        recurrent_initializer=rec.get("recurrent_initializer", "orthogonal"),
        bias_initializer=rec.get("bias_initializer", "zeros"),
        unit_forget_bias=rec.get("unit_forget_bias", True),
        kernel_regularizer=rec.get("kernel_regularizer"),
        recurrent_regularizer=rec.get("recurrent_regularizer"),
        bias_regularizer=rec.get("bias_regularizer"),
        activity_regularizer=rec.get("activity_regularizer"),
        kernel_constraint=rec.get("kernel_constraint"),
        recurrent_constraint=rec.get("recurrent_constraint"),
        bias_constraint=rec.get("bias_constraint"),
        implementation=rec.get("implementation", 2),
        go_backwards=rec.get("go_backwards", False),
        stateful=rec.get("stateful", False),
        time_major=rec.get("time_major", False),
        unroll=rec.get("unroll", False)
    )
    # None 값 제거
    lstm_model_params = {k: v for k, v in lstm_model_params.items() if v is not None}

    # 모델 구성 파라미터 (test_lstm과 동일한 구조)
    model_config_params = dict(
        num_layers=rec.get("num_layers", 1),
        bidirectional=rec.get("bidirectional", False),
        sequence_length=rec.get("sequence_length", 20),
        feature_dim=rec.get("embedding_dim", 8),  # 스키마의 "embedding_dim"을 "feature_dim"으로 매핑
        dense_dropout=rec.get("dense_dropout", 0.0)
    )

    # ===== LSTM 모델 구성 (test_lstm과 완전히 동일) =====
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
    
    # 출력 분류기 (스키마의 "output_classifier" 반영)
    classifier_activation = rec.get("output_classifier", "softmax")
    if classifier_activation and classifier_activation.lower() != "softmax":
        predictions = Dense(num_classes, activation=classifier_activation)(x)
    else:
        predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)

    # ----------- 컴파일 파라미터 -----------
    optimizer_name = str(rec.get("optimizer", "adam")).lower()
    learning_rate = rec.get("learning_rate", 0.001)
    
    if optimizer_name == "adamw":
        optimizer = AdamW(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # ----------- 콜백 설정 (스키마의 early_stopping_patience 사용) -----------
    callbacks = []
    if str(rec.get("callbacks", "")).lower() == "earlystopping":
        patience = rec.get("early_stopping_patience", 3)
        callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))

    # ----------- fit(훈련) 파라미터 -----------
    fit_kwargs = dict(
        epochs=rec.get("epochs", 5),
        batch_size=rec.get("batch_size", 16),
        verbose=rec.get("verbose", 1),
        validation_split=rec.get("validation_split", 0.2),
        validation_data=rec.get("validation_data"),
        shuffle=rec.get("shuffle", True),
        class_weight=rec.get("class_weight"),
        sample_weight=rec.get("sample_weight"),
        initial_epoch=rec.get("initial_epoch", 0),
        steps_per_epoch=rec.get("steps_per_epoch"),
        validation_steps=rec.get("validation_steps"),
        validation_batch_size=rec.get("validation_batch_size"),
        validation_freq=rec.get("validation_freq", 1),
        max_queue_size=rec.get("max_queue_size", 10),
        workers=rec.get("workers", 1),
        use_multiprocessing=rec.get("use_multiprocessing", False),
        callbacks=callbacks if callbacks else None
    )
    # None 값 제거
    fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

    launch_experiment(model, X_train_seq, y_train_labels, training_params=fit_kwargs)