import tkinter as tk
from tkinter import ttk
import os
import json
import glob
import threading
import time

from matplotlib.figure import Figure
from tuneparam.gui.theme.fonts import DEFAULT_FONT, ERROR_FONT_SMALL
from tuneparam.database.db import SessionLocal
from tuneparam.database.service.dao import user_crud, create_training_log, get_all_models, get_model_by_version_and_type
from tuneparam.database.schema import Model, User
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.scrolledtext import ScrolledText


# 그래프용 작은 폰트 정의
GRAPH_FONT = ('Helvetica', 8)

def get_theme_colors(is_dark_theme=False):
    """테마에 따른 색상 반환"""
    if is_dark_theme:
        return {
            'bg': "#2b2b2b",
            'fg': "#ffffff",  # 더 밝은 흰색으로 변경
            'grid': "#404040",
            'axis': "#ffffff",  # 축 색상도 더 밝게
            'loss': "#ff4444",
            'val_loss': "#ffaa44",
            'acc': "#4a9eff",
            'val_acc': "#66ccff",
            'canvas_bg': "#2b2b2b"
        }
    else:
        return {
            'bg': "#ffffff",
            'fg': "#313131",
            'grid': "#f0f0f0",
            'axis': "#313131",
            'loss': "#ff0000",
            'val_loss': "#ff8800",
            'acc': "#0066cc",
            'val_acc': "#00aaff",
            'canvas_bg': "white"
        }

from tkinter import ttk


def setup_log_tab(tab_train, log_dir=None, is_dark_theme=False):
    theme = get_theme_colors(is_dark_theme)

    db = SessionLocal()
    models = get_all_models(db=db)
    db.close()

    # (model_type, version) 조합을 정렬
    type_version_pairs = sorted(set(
        (m.model_type, m.version) for m in models if m.model_type and m.version
    ))
    pair_labels = [f"{t} / {v}" for t, v in type_version_pairs]  # 표시용 문자열

    # UI 상단: 모델 선택 콤보박스
    frame = ttk.Frame(tab_train)
    frame.pack(fill='x', pady=5)

    ttk.Label(frame, text="Model (Type / Version):").pack(side='left', padx=5)
    model_pair_cb = ttk.Combobox(frame, values=pair_labels, state='readonly')
    model_pair_cb.pack(side='left', padx=5)

    # 메인 영역 프레임 (좌우로 나눔)
    main_frame = ttk.Frame(tab_train)
    main_frame.pack(fill='both', expand=True, padx=10, pady=10)

    left_frame = ttk.LabelFrame(main_frame, text="Training Parameters")
    left_frame.pack(side='left', fill='both', expand=False, padx=(0, 10))  # 줄어들도록 설정

    right_frame = ttk.LabelFrame(main_frame, text="Training Log")
    right_frame.pack(side='left', fill='both', expand=True)

    # 하이퍼파라미터 표시용 텍스트박스 (작은 높이, 너비, 글꼴)
    params_text = ScrolledText(
        left_frame,
        height=10,          # 높이 줄임
        width=50,           # 폭 고정
        wrap='word',
        font=("Helvetica", 9)  # 글꼴 작게
    )
    params_text.pack(fill='both', expand=True)

    def display_params_from_json(json_path):
        if not json_path or not os.path.exists(json_path):
            params_text.delete(1.0, 'end')
            params_text.insert('end', "No parameter info available.")
            return

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            params_info = data.get("params_info", {})
        except Exception as e:
            params_text.delete(1.0, 'end')
            params_text.insert('end', f"Error loading JSON: {e}")
            return

        formatted = "\n".join(f"{key}: {value}" for key, value in params_info.items())
        params_text.delete(1.0, 'end')
        params_text.insert('end', formatted)

    def draw_log_plot_from_db_logs(logs, is_dark_theme=False):
        if not logs:
            print("No training logs to display")
            return

        logs = sorted(logs, key=lambda x: x.epoch or 0)
        epochs = [log.epoch for log in logs]
        train_loss = [log.loss for log in logs]
        val_loss = [log.val_loss for log in logs]
        train_acc = [log.accuracy for log in logs]
        val_acc = [log.val_accuracy for log in logs]

        # epochs를 +1 해서 1부터 시작하게 만듦
        epochs_display = [e + 1 for e in epochs]

        # 기존 그래프 제거 (중복 방지)
        for widget in right_frame.winfo_children():
            widget.destroy()

        # 다크모드/라이트모드 색상 설정
        if is_dark_theme:
            bg_color = '#1e1e1e'  # 좀 더 어두운 거의 검정에 가까운 배경
            grid_color = '#2e2e2e'  # 약간 밝은 회색 격자
            axis_color = '#f0f0f0'  # 거의 흰색에 가까운 축 색상 (눈에 잘 띄게)

            train_loss_color = '#e24a33'  # 강렬한 진한 빨강
            val_loss_color = '#ffae42'  # 따뜻한 노랑 오렌지 (명확한 차이)
            train_acc_color = '#348abd'  # 진한 청색 (파랑 계열)
            val_acc_color = '#4daf4a'  # 진한 초록 (확실히 다른 색)

            title_color = '#f5f5f5'  # 밝은 회색/흰색에 가까운 타이틀
            label_color = '#dcdcdc'  # 약간 부드러운 밝은 회색 라벨
            legend_facecolor = '#1e1e1e'  # 배경과 동일하게 어둡게
            legend_edgecolor = '#555555'  # 진한 회색 테두리
            legend_label_color = 'white'
        else:
            bg_color = 'white'
            grid_color = '#f0f0f0'
            axis_color = '#313131'
            train_loss_color = '#d62728'  # 진한 빨강 (더 선명하고 깊은 빨강)
            val_loss_color = '#ff7f0e'  # 주황 (좀 더 밝고 명확한 주황)

            train_acc_color = '#1f77b4'  # 진한 파랑 (클래식 블루)
            val_acc_color = '#2ca02c'  # 초록 (분명한 초록)
            title_color = '#222222'  # 거의 검정에 가까운 짙은 회색
            label_color = '#444444'  # 진한 회색, 너무 강하지 않게
            legend_facecolor = 'white'
            legend_edgecolor = '#cccccc'
            legend_label_color = 'black'

        # Figure 생성 (배경색 설정)
        fig = Figure(figsize=(6, 4), dpi=100, constrained_layout=True, facecolor=bg_color)

        # Loss subplot
        ax1 = fig.add_subplot(211, facecolor=bg_color)
        ax1.plot(epochs_display, train_loss, label="Train Loss", color=train_loss_color, linewidth=2, marker='o')
        ax1.plot(epochs_display, val_loss, label="Val Loss", color=val_loss_color, linestyle='--', linewidth=2,
                 marker='o')
        ax1.set_ylabel("Loss", fontsize=10, color=label_color)
        ax1.set_title("Training & Validation Loss", fontsize=12, weight='bold', color=title_color)
        ax1.legend(fontsize=8, facecolor=legend_facecolor, edgecolor=legend_edgecolor, labelcolor=legend_label_color)
        ax1.grid(True, linestyle=':', linewidth=0.5, color=grid_color)
        ax1.tick_params(colors=axis_color)
        ax1.set_xlim(0.5, max(epochs_display) + 0.5)
        ax1.set_xticks(range(1, max(epochs_display) + 1))
        ax1.margins(x=0, y=0.1)

        # Accuracy subplot
        ax2 = fig.add_subplot(212, facecolor=bg_color)
        ax2.plot(epochs_display, train_acc, label="Train Accuracy", color=train_acc_color, linewidth=2, marker='o')
        ax2.plot(epochs_display, val_acc, label="Val Accuracy", color=val_acc_color, linestyle='--', linewidth=2,
                 marker='o')
        ax2.set_xlabel("Epoch", fontsize=10, color=label_color)
        ax2.set_ylabel("Accuracy", fontsize=10, color=label_color)
        ax2.set_title("Training & Validation Accuracy", fontsize=12, weight='bold', color=title_color)
        ax2.legend(fontsize=8, facecolor=legend_facecolor, edgecolor=legend_edgecolor, labelcolor=legend_label_color)
        ax2.grid(True, linestyle=':', linewidth=0.5, color=grid_color)
        ax2.tick_params(colors=axis_color)
        ax2.set_xlim(0.5, max(epochs_display) + 0.5)
        ax2.set_xticks(range(1, max(epochs_display) + 1))
        ax2.margins(x=0, y=0.1)

        # tkinter에 그래프 표시
        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill='both', expand=True)

    def load_logs_and_draw(model_type, version, is_dark_theme=False):
        db = SessionLocal()
        model = get_model_by_version_and_type(db=db, model_type=model_type, version=version)
        db.close()

        for widget in right_frame.winfo_children():
            widget.destroy()

        if not model:
            print(f"No model found for {model_type} v{version}")
            return

        # 왼쪽: 하이퍼파라미터 정보 표시
        display_params_from_json(model.init_info_path)

        # 오른쪽: 로그 그래프 표시
        logs = model.training_logs
        if not logs:
            print("No training logs found.")
            return

        draw_log_plot_from_db_logs(logs, is_dark_theme=is_dark_theme)

    def update_theme(is_dark):
        nonlocal is_dark_theme
        is_dark_theme = is_dark

        theme = get_theme_colors(is_dark)
        params_text.config(
            bg=theme['canvas_bg'],
            fg=theme['fg'],
            insertbackground=theme['fg']
        )
        # 현재 선택된 모델 타입/버전으로 그래프 다시 그리기 (다크모드 반영)
        selection = model_pair_cb.get()
        if " / " in selection:
            model_type, version = selection.split(" / ")
            load_logs_and_draw(model_type, version, is_dark_theme=is_dark_theme)

    def on_model_pair_selected(event):
        selection = model_pair_cb.get()
        if " / " not in selection:
            return
        model_type, version = selection.split(" / ")
        load_logs_and_draw(model_type, version, is_dark_theme=is_dark_theme)

    model_pair_cb.bind('<<ComboboxSelected>>', on_model_pair_selected)

    # 초기 자동 표시
    if pair_labels:
        model_pair_cb.current(0)
        default_model_type, default_version = type_version_pairs[0]
        load_logs_and_draw(default_model_type, default_version)

    return {"update_theme": update_theme}