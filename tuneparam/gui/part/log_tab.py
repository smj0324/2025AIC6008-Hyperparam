import tkinter as tk
from tkinter import ttk
import os
import json
import glob
import threading
import time

from tuneparam.gui.theme.fonts import DEFAULT_FONT, ERROR_FONT_SMALL
from tuneparam.database.db import SessionLocal
from tuneparam.database.service.dao import user_crud
from tuneparam.gui.visual import *

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

def setup_log_tab(tab_train, log_dir=None, user_data=None, is_dark_theme=False):
    """Train 탭 설정 - 실시간 학습 상태 모니터링 기능 추가"""
    # 메인 변수들
    train_state = {
        "monitoring": False,
        "log_dir": log_dir,
        "user_data": user_data,
        "current_epoch": 0,
        "total_epochs": 0,
        "last_loss": 0,
        "last_accuracy": 0,
        "params_info": {},
        "is_dark_theme": is_dark_theme,
        "colors": get_theme_colors(is_dark_theme)
    }
    db = SessionLocal()
    user = user_crud.get_user_by_username(db, username="alice")
    print(user)
    db.close()

    colors = train_state["colors"]

    # init_info 파일에서 파라미터 로드
    if log_dir and os.path.exists(log_dir):
        init_info_files = glob.glob(os.path.join(log_dir, "init_info_*.json"))
        if init_info_files:
            try:
                latest_file = sorted(init_info_files)[-1]
                with open(latest_file, 'r', encoding='utf-8') as f:
                    init_data = json.load(f)
                    if 'params_info' in init_data:
                        train_state["params_info"] = init_data['params_info']
                        train_state["total_epochs"] = init_data['params_info'].get('epochs', 0)
                print(f"✅ Init info 파일 로드 성공: {latest_file}")
            except Exception as e:
                print(f"❌ Init info 파일 로드 오류: {e}")

    # 기본 그리드 설정
    tab_train.columnconfigure(0, weight=1)
    tab_train.columnconfigure(1, weight=1)
    tab_train.columnconfigure(2, weight=1)

    username = user.username
    model = user.models[0]

    training_logs = user.models[0].training_logs

    print(training_logs[0].val_accuracy)
    print(training_logs[1].val_accuracy)
    print(training_logs[2].val_accuracy)

    # 상태 메시지 표시 (첫 번째 행)
    status_label = ttk.Label(
        tab_train, 
        text=f"{username}님! {model.model_type} 모델을 이용한 학습을 시작합니다.",
        font=DEFAULT_FONT
    )
    status_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 5))
    
    # 설명 텍스트 (두 번째 행)
    desc_label = ttk.Label(
        tab_train, 
        text="현재 학습 상태와 파라미터입니다.", 
        font=DEFAULT_FONT
    )
    desc_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=(5, 10))
    
    # 진행 상태 표시
    progress_frame = ttk.Frame(tab_train)
    progress_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
    print(model.total_epoch)
    # 에포크 진행 표시
    epoch_label = ttk.Label(
        progress_frame, 
        text=f"전체 에포크: {str(model.total_epoch)}",
        font=DEFAULT_FONT
    )

    epoch_label.pack(side="left", padx=(0, 20))
    
    # 그래프 프레임 생성
    home_frame = ttk.LabelFrame(tab_train, text="Training Parameters")
    home_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
    
    loss_frame = ttk.LabelFrame(tab_train, text="Loss Graph")
    loss_frame.grid(row=3, column=1, sticky="nsew", padx=5, pady=5)
    
    acc_frame = ttk.LabelFrame(tab_train, text="Accuracy Graph")
    acc_frame.grid(row=3, column=2, sticky="nsew", padx=5, pady=5)
    
    # 파라미터 표시
    param_canvas = tk.Canvas(home_frame, bg=colors['canvas_bg'], highlightthickness=0)
    param_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # 초기 파라미터 텍스트 설정
    setup_initial_param_text(train_state, param_canvas)
    print(len(training_logs))

    # Loss 그래프 설정
    loss_canvas = tk.Canvas(loss_frame, bg=colors['canvas_bg'], highlightthickness=0, width=300, height=200)
    loss_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    setup_loss_graph(loss_canvas, colors)
    draw_loss_graph(loss_canvas, training_logs, colors)


    # Accuracy 그래프 설정
    acc_canvas = tk.Canvas(acc_frame, bg=colors['canvas_bg'], highlightthickness=0, width=300, height=200)
    acc_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    # setup_accuracy_graph(acc_canvas, colors)
    # draw_accuracy_graph(acc_canvas, training_logs, colors)  # ← 추가


# 메인 코드와 연결하는 부분
# 초기 파라미터 텍스트 설정
def setup_initial_param_text(train_state, param_canvas):
    """초기 파라미터 텍스트 설정"""
    text_content = "Training Configuration:\n\n"
    params_info = train_state.get("params_info", {})

    # 학습 관련 파라미터만 표시
    if params_info:
        training_params = {
            "Epochs": params_info.get("epochs", "N/A"),
            "Batch Size": params_info.get("batch_size", "N/A"),
            "Validation Split": f"{params_info.get('validation_split', 0) * 100}%",
            "Learning Rate": params_info.get("learning_rate", "N/A"),
            "Optimizer": params_info.get("optimizer", "N/A"),
            "Loss Function": params_info.get("loss", "N/A")
        }

        # None이 아닌 값만 표시
        for key, value in training_params.items():
            if value not in [None, "N/A"]:
                text_content += f"{key}: {value}\n"
    else:
        text_content = "학습 파라미터를 불러오는 중..."

    param_canvas.delete("all")
    param_canvas.create_text(
        10, 10,
        text=text_content,
        anchor="nw",
        font=DEFAULT_FONT,
        fill=train_state["colors"]['fg'],
        tags="param_text"
    )