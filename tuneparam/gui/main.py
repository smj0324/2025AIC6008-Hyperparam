import threading
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gui.part.utils import root, style, THEME_BG, set_theme, create_notebook_with_tabs, create_theme_buttons
from gui.part.main_tap import setup_main_tab
from gui.part.train_tab import setup_train_tab
from gui.part.log_tab import setup_log_tab
from gui.part.results_tab import setup_results_tab
from framework.keras_ import TrainingLogger
from database.service.dao import model_crud
from database.db import SessionLocal
from tuneparam.models import mobilenetv3, lstm, resnet
from keras.callbacks import EarlyStopping
import tensorflow as tf


global X_train, y_train
global model_type
X_train = None
y_train = None

def launch_experiment(
    model,
    X, y,
    training_params=None,
    preset_data=None,
        custom_callbacks=None,
        log_dir="logs"
):
    global X_train, y_train
    X_train = X
    y_train = y
    # ===== GUI 테마/레이아웃 =====
    set_theme("forest-light")
    root.option_add("*Font", '"Helvetica" 11')

    notebook, tab_main, tab_train, tab_results, tab_logs = create_notebook_with_tabs(root)
    notebook.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)

    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=0)

    params = training_params or {"epochs": 10, "batch_size": 32, "validation_split": 0.2}
    default_params = {"epochs": 10, "batch_size": 32, "validation_split": 0.2}

    summary_params = dict(training_params) if training_params else dict(default_params)
    _preset_logger = TrainingLogger(log_dir=log_dir, params=params, X=X_train, y=y_train, summary_params=summary_params)
    main_preset = preset_data or _preset_logger.get_preset_data_for_main_tab()

    # 탭 초기 설정
    train_handlers = setup_train_tab(tab_train)
    results_handlers = setup_results_tab(tab_results, train_parameters=summary_params, preset_logger =_preset_logger)
    logs_handlers = setup_log_tab(tab_logs)

    # 테마 변경 핸들러
    def handle_theme_change(theme_name):
        is_dark = theme_name == "forest-dark"
        set_theme(theme_name)
        if train_handlers and "update_theme" in train_handlers:
            train_handlers["update_theme"](is_dark)
        if results_handlers and "update_theme" in results_handlers:
            results_handlers["update_theme"](is_dark)
        if logs_handlers and "update_theme" in logs_handlers:
            logs_handlers["update_theme"](is_dark)

    theme_frame = create_theme_buttons(root, handle_theme_change)
    theme_frame.grid(row=0, column=1, sticky="ne", padx=(0, 10), pady=(10, 0))

    # ===== TrainingLogger/fit 갱신 함수 =====
    def start_training_with_log_dir(new_log_dir, user_info):
        global model_type
        logger = TrainingLogger(log_dir=new_log_dir, params=params, X=X_train, y=y_train, summary_params=summary_params)
        
        model_db_data = {
            "model_size": user_info['Model Size'],
            "dataset_size": user_info['Dataset Size'],
            "model_type": user_info['Model Type'],
            "dataset_type": user_info['Dataset Type'],
            "goal": user_info['Goal'],
            "total_epoch": None,
            'version' : user_info['Version'],
        }
        model_type = user_info['Model Type']

        db = SessionLocal()
        model_db = model_crud.create_model_for_user(db=db, username=None, model_data=model_db_data)
        db.close()
        logger.model_db = model_db
        
        train_handlers["start_monitoring"](new_log_dir, user_info)
        
        def fit_thread():
            callbacks = [logger]
            if custom_callbacks:
                callbacks += list(custom_callbacks)
            callbacks.append(EarlyStopping(monitor='val_loss', patience=10))

            import tensorflow as tf

            if isinstance(X_train, tf.data.Dataset):
                train_ds, val_ds = split_tf_dataset(X_train, val_ratio=params.get("validation_split", 0.2))
                fit_kwargs = dict(
                    epochs=params["epochs"],
                    callbacks=callbacks,
                    validation_data=val_ds
                )
                model.fit(train_ds, **fit_kwargs)
            else:
                fit_kwargs = dict(
                    epochs=params["epochs"],
                    batch_size=params["batch_size"],
                    callbacks=callbacks
                )
                if "validation_data" in params:
                    fit_kwargs["validation_data"] = params["validation_data"]
                else:
                    fit_kwargs["validation_split"] = params.get("validation_split", 0.2)

                model.fit(X_train, y_train, **fit_kwargs)

        threading.Thread(target=fit_thread, daemon=True).start()

    setup_main_tab(tab_main, notebook, tab_train, preset_data=main_preset,
                   set_log_dir_callback=start_training_with_log_dir, logger = _preset_logger)

    root.mainloop()
    
def start_retrain(gpt_output):
    global model_type
    if model_type == "MobilenetV3":
        mobilenetv3.retrain_mobilenet(X_train, y_train, gpt_output)
    elif model_type == "LSTM":
        lstm.retrain_lstm(X_train, y_train, gpt_output)
    elif model_type == "Resnet":
        resnet.retrain_resnet(X_train, y_train, gpt_output)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

def is_array_like(x):
    import numpy as np
    import tensorflow as tf
    return isinstance(x, (np.ndarray, tf.Tensor))

def split_tf_dataset(dataset, val_ratio=0.2):
    total_count = dataset.cardinality().numpy()
    val_count = int(total_count * val_ratio)
    train_count = total_count - val_count
    train_ds = dataset.take(train_count)
    val_ds = dataset.skip(train_count)
    return train_ds, val_ds