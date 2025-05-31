import tkinter as tk
from tkinter import ttk
import os
import json
import glob
import threading
import time

from tuneparam.gui.theme.fonts import DEFAULT_FONT, ERROR_FONT_SMALL
from tuneparam.database.db import SessionLocal
from tuneparam.database.service.dao import user_crud, create_training_log, get_all_models, get_model_by_version_and_type
from tuneparam.database.schema import Model, User
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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


def setup_log_tab(tab_train, log_dir=None):
    db = SessionLocal()
    models = get_all_models(db=db)
    db.close()

    # (model_type, version) 조합을 정렬
    type_version_pairs = sorted(set(
        (m.model_type, m.version) for m in models if m.model_type and m.version
    ))
    pair_labels = [f"{t} / {v}" for t, v in type_version_pairs]  # 표시용 문자열

    # UI 구성
    frame = ttk.Frame(tab_train)
    frame.pack(fill='x', pady=5)

    ttk.Label(frame, text="Model (Type / Version):").pack(side='left', padx=5)
    model_pair_cb = ttk.Combobox(frame, values=pair_labels, state='readonly')
    model_pair_cb.pack(side='left', padx=5)

    log_frame = ttk.Frame(tab_train)
    log_frame.pack(fill='both', expand=True, pady=10, padx=10)

    def draw_log_plot_from_db_logs(logs):


        if not logs:
            print("No training logs to display")
            return

        # 정렬
        logs = sorted(logs, key=lambda x: x.epoch or 0)

        # 값 추출
        epochs = [log.epoch for log in logs]
        train_loss = [log.loss for log in logs]
        val_loss = [log.val_loss for log in logs]
        train_acc = [log.accuracy for log in logs]
        val_acc = [log.val_accuracy for log in logs]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

        # Loss Plot
        ax1.plot(epochs, train_loss, label="Train Loss", color='blue')
        ax1.plot(epochs, val_loss, label="Val Loss", color='orange')
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy Plot
        ax2.plot(epochs, train_acc, label="Train Accuracy", color='green')
        ax2.plot(epochs, val_acc, label="Val Accuracy", color='red')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        fig.tight_layout()

        # Tkinter에 출력
        canvas = FigureCanvasTkAgg(fig, master=log_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def load_logs_and_draw(model_type, version):
        # DB에서 모델 가져오기
        db = SessionLocal()
        model = get_model_by_version_and_type(db=db, model_type=model_type, version=version)
        db.close()

        for widget in log_frame.winfo_children():
            widget.destroy()

        if not model:
            print(f"No model found for {model_type} v{version}")
            return

        logs = model.training_logs
        if not logs:
            print("No training logs found.")
            return

        # 정렬 보장
        logs = sorted(logs, key=lambda x: x.epoch or 0)
        draw_log_plot_from_db_logs(logs)

    def on_model_pair_selected(event):
        selection = model_pair_cb.get()
        if " / " not in selection:
            return
        model_type, version = selection.split(" / ")
        load_logs_and_draw(model_type, version)

    model_pair_cb.bind('<<ComboboxSelected>>', on_model_pair_selected)

    # 초기 자동 표시
    if pair_labels:
        model_pair_cb.current(0)
        default_model_type, default_version = type_version_pairs[0]
        load_logs_and_draw(default_model_type, default_version)
