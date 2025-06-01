import tkinter as tk
import os
from tkinter import ttk
from datetime import datetime
from tuneparam.gui.theme.fonts import DEFAULT_FONT, ERROR_FONT_LARGE, ERROR_FONT, ERROR_FONT_SMALL, ERROR_FONT_UNDERLINE
def setup_main_tab(tab_main, notebook, tab_train, preset_data=None, set_log_dir_callback=None, logger = None):
    username_var = tk.StringVar()
    version_var = tk.StringVar(value=preset_data.get("Version", "") if preset_data else "")
    hardware_var = tk.StringVar(value=preset_data.get("Hardware", "") if preset_data else "")
    model_size_var = tk.StringVar(value=preset_data.get("Model Size", "") if preset_data else "")
    dataset_size_var = tk.StringVar(value=preset_data.get("Dataset Size", "") if preset_data else "")
    dataset_type_var = tk.StringVar(value=preset_data.get("Dataset Type", "") if preset_data else "")
    goal_var = tk.StringVar(value=preset_data.get("Goal", "") if preset_data else "")

    model_type_value = preset_data.get("Model Type") if preset_data else ""
    if model_type_value is None or model_type_value == "":
        model_type_var = tk.StringVar(value="MobilenetV3")
    else:
        model_type_var = tk.StringVar(value=model_type_value)

    # ---- 폼 위젯을 리스트로 관리 ----
    form_widgets = []

    # ---- Main 탭 폼 레이아웃 ----
    label_username = ttk.Label(tab_main, text="Username")
    label_username.grid(row=0, column=0, sticky="w", padx=5, pady=(0, 2))
    form_widgets.append(label_username)

    label_version = ttk.Label(tab_main, text="Version")
    label_version.grid(row=0, column=1, sticky="w", padx=5, pady=(0, 2))
    form_widgets.append(label_version)

    entry_username = ttk.Entry(tab_main, textvariable=username_var, width=18)
    entry_username.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 15))
    form_widgets.append(entry_username)

    entry_version = ttk.Entry(tab_main, textvariable=version_var, width=18)
    entry_version.grid(row=1, column=1, sticky="ew", padx=5, pady=(0, 15))
    form_widgets.append(entry_version)

    label_hardware = ttk.Label(tab_main, text="Hardware")
    label_hardware.grid(row=2, column=0, sticky="w", padx=5)
    form_widgets.append(label_hardware)

    label_model_size = ttk.Label(tab_main, text="Model Size:")
    label_model_size.grid(row=2, column=1, sticky="w", padx=5)
    form_widgets.append(label_model_size)

    combo_hardware = ttk.Combobox(tab_main, textvariable=hardware_var, values=["Value", "A100", "RTX4090", "TPU"], state="readonly")
    combo_hardware.grid(row=3, column=0, sticky="ew", padx=5, pady=(0, 15))
    form_widgets.append(combo_hardware)

    combo_model_size = ttk.Combobox(tab_main, textvariable=model_size_var, values=["Value", "Small", "Base", "Large"], state="readonly")
    combo_model_size.grid(row=3, column=1, sticky="ew", padx=5, pady=(0, 15))
    form_widgets.append(combo_model_size)

    label_dataset_size = ttk.Label(tab_main, text="Dataset Size:")
    label_dataset_size.grid(row=4, column=0, sticky="w", padx=5)
    form_widgets.append(label_dataset_size)

    label_model_type = ttk.Label(tab_main, text="Model Type:")
    label_model_type.grid(row=4, column=1, sticky="w", padx=5)
    form_widgets.append(label_model_type)

    combo_dataset_size = ttk.Combobox(tab_main, textvariable=dataset_size_var, values=["Value", "Tiny", "Small", "Full"], state="readonly")
    combo_dataset_size.grid(row=5, column=0, sticky="ew", padx=5)
    form_widgets.append(combo_dataset_size)

    combo_model_type = ttk.Combobox(tab_main, textvariable=model_type_var, values=["MobilenetV3", "Resnet", "LSTM"], state="readonly")
    combo_model_type.grid(row=5, column=1, sticky="ew", padx=5)
    form_widgets.append(combo_model_type)

    label_dataset_type = ttk.Label(tab_main, text="Dataset Type:")
    label_dataset_type.grid(row=6, column=0, sticky="w", padx=5, pady=(10,0))
    form_widgets.append(label_dataset_type)

    label_goal = ttk.Label(tab_main, text="Goal:")
    label_goal.grid(row=6, column=1, sticky="w", padx=5, pady=(10,0))
    form_widgets.append(label_goal)

    combo_dataset_type = ttk.Combobox(tab_main, textvariable=dataset_type_var, values=["Value", "Image", "Text", "Tabular"], state="readonly")
    combo_dataset_type.grid(row=7, column=0, sticky="ew", padx=5)
    form_widgets.append(combo_dataset_type)

    combo_goal = ttk.Combobox(tab_main, textvariable=goal_var, values=["Value", "Accuracy", "Speed", "Memory"], state="readonly")
    combo_goal.grid(row=7, column=1, sticky="ew", padx=5)
    form_widgets.append(combo_goal)

    save_btn = ttk.Button(tab_main, text="Save", style="Accent.TButton")
    save_btn.grid(row=8, column=1, sticky="e", padx=5, pady=(40, 0))
    form_widgets.append(save_btn)

    tab_main.columnconfigure(0, weight=1)
    tab_main.columnconfigure(1, weight=1)

    # ---- 판정 및 에러 메시지 ----
    def check_train_condition(data):
        if data["Username"] and data["Hardware"] != "Value":
            return True
        return False

    def show_error_message_in_main():
        for widget in form_widgets:
            widget.grid_remove()

        for widget in tab_main.place_slaves():
            widget.destroy()

        error_frame = ttk.Frame(tab_main)
        error_frame.place(relx=0.5, rely=0.5, anchor="center")
        icon_label = ttk.Label(error_frame, text="⚠️", font=ERROR_FONT_LARGE)
        icon_label.pack(pady=(0, 10))
        msg1 = ttk.Label(error_frame, text="학습 로그를 못찾았어요!", font=ERROR_FONT)
        msg1.pack()

        msg_line = ttk.Frame(error_frame)
        msg_line.pack()

        msg2 = ttk.Label(msg_line, text="자세한 이유는 ", font=ERROR_FONT_SMALL)
        msg2.pack(side="left", anchor="center")
        link = ttk.Label(msg_line, text="링크를", foreground="#0066CC", cursor="hand2", font=ERROR_FONT_UNDERLINE)
        link.pack(side="left", anchor="center")
        msg3 = ttk.Label(msg_line, text=" 참고해주세요.", font=ERROR_FONT_SMALL)
        msg3.pack(side="left", anchor="center")

        def open_link(event):
            import webbrowser
            webbrowser.open("https://your-link-url.com")
        link.bind("<Button-1>", open_link)

        ttk.Button(error_frame, text="돌아가기", command=show_form_again).pack(pady=(15, 0))


    def show_form_again():
        for widget in tab_main.place_slaves():
            widget.destroy()
        for widget in form_widgets:
            widget.grid()

    save_already_clicked = [False]
    def on_save():
        if save_already_clicked[0]:
            print("저장 버튼은 일단 한 번 만~")
            return
        save_already_clicked[0] = True
        data = {
            "Username": username_var.get(),
            "Version": version_var.get(),
            "Hardware": hardware_var.get(),
            "Model Size": model_size_var.get(),
            "Dataset Size": dataset_size_var.get(),
            "Model Type": model_type_var.get(),
            "Dataset Type": dataset_type_var.get(),
            "Goal": goal_var.get()
        }
        logger.user_data = data
        result = check_train_condition(data)
        if result:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            username = username_var.get() or "user"
            version = version_var.get() or "version"
            custom_log_dir = os.path.join("logs", f"{username}_{version}_{now}")
            os.makedirs(custom_log_dir, exist_ok=True)

            if set_log_dir_callback:
                set_log_dir_callback(custom_log_dir, data)  # 모델 학습 시작 및 train 탭으로 전환
                notebook.select(tab_train)  # train 탭으로 전환
        else:
            show_error_message_in_main()

        for widget in form_widgets:
            widget.configure(state="disabled")
        save_btn.configure(state="disabled")

    save_btn.config(command=on_save)
