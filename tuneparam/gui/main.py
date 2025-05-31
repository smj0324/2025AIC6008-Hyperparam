import threading
import sys
import os

from tuneparam.gui.part.log_tab import setup_log_tab
from tuneparam.gui.part.results_tab import setup_results_tab
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gui.part.results_tab import setup_results_tab
from gui.part.utils import root, style, THEME_BG, set_theme, create_notebook_with_tabs, create_theme_buttons
from gui.part.main_tap import setup_main_tab
from gui.part.train_tab import setup_train_tab
from tuneparam.framework.keras_ import TrainingLogger

def launch_experiment(
    model,
    X_train, y_train,
    training_params=None,
    preset_data=None,
        custom_callbacks=None,
        log_dir="logs"
):
    # ===== GUI 테마/레이아웃 =====
    set_theme("forest-light")
    root.option_add("*Font", '"나눔스퀘어_ac Bold" 11')

    notebook, tab_main, tab_train, tab_results, tab_logs = create_notebook_with_tabs(root)
    notebook.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)

    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=0)

    params = training_params or {"epochs": 10, "batch_size": 32, "validation_split": 0.2}
    default_params = {"epochs": 10, "batch_size": 32, "validation_split": 0.2}

    summary_params = copy.deepcopy(training_params) if training_params else copy.deepcopy(default_params)
    _preset_logger = TrainingLogger(log_dir=log_dir, params=params, X=X_train, y=y_train, summary_params=summary_params)
    main_preset = preset_data or _preset_logger.get_preset_data_for_main_tab()

    # Train 탭 초기 설정
    train_handlers = setup_train_tab(tab_train)
    setup_results_tab(tab_results, train_parameters=summary_params, preset_logger =_preset_logger)

    # 테마 변경 핸들러
    def handle_theme_change(theme_name):
        is_dark = theme_name == "forest-dark"
        set_theme(theme_name)
        if train_handlers and "update_theme" in train_handlers:
            train_handlers["update_theme"](is_dark)

    theme_frame = create_theme_buttons(root, handle_theme_change)
    theme_frame.grid(row=0, column=1, sticky="ne", padx=(0, 10), pady=(10, 0))

    # ===== TrainingLogger/fit 갱신 함수 =====
    def start_training_with_log_dir(new_log_dir, user_info):
        logger = TrainingLogger(log_dir=new_log_dir, params=params, X=X_train, y=y_train, summary_params=summary_params)
        
        # Train 탭 업데이트
        train_handlers["start_monitoring"](new_log_dir, user_info)
        
        def fit_thread():
            callbacks = [logger]
            if custom_callbacks:
                callbacks += list(custom_callbacks)
            model.fit(
                X_train, y_train,
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                validation_split=params.get("validation_split", 0.2),
                callbacks=callbacks
            )
        threading.Thread(target=fit_thread, daemon=True).start()

    setup_main_tab(tab_main, notebook, tab_train, preset_data=main_preset,
                   set_log_dir_callback=start_training_with_log_dir, logger = _preset_logger)

    root.mainloop()
    
