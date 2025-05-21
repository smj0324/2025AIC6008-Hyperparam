import tkinter as tk
from tkinter import ttk
import os
import json
import glob
import threading
import time
from tuneparam.gui.theme.fonts import DEFAULT_FONT, ERROR_FONT_SMALL

def setup_train_tab(tab_train, log_dir=None, user_data=None):
    """Train 탭 설정 - 실시간 학습 상태 모니터링 기능 추가"""
    # 메인 변수들
    train_state = {
        "monitoring": False,
        "log_dir": log_dir,
        "user_data": user_data,
        "current_epoch": 0,
        "total_epochs": 0,
        "last_loss": 0,
        "last_accuracy": 0
    }
    
    # JSON 파일에서 데이터 로드
    if log_dir and os.path.exists(log_dir):
        json_path = os.path.join(log_dir, "parameters.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    train_state["user_data"] = json.load(f)
                print(f"JSON 파일 로드 성공: {json_path}")
            except Exception as e:
                print(f"JSON 파일 로드 오류: {e}")
    
    # 기본 그리드 설정
    tab_train.columnconfigure(0, weight=1)
    tab_train.columnconfigure(1, weight=1)
    tab_train.columnconfigure(2, weight=1)
    
    # 사용자 데이터 가져오기
    username = train_state["user_data"].get("Username", "사용자") if train_state["user_data"] else "사용자"
    model_type = train_state["user_data"].get("Model Type", "Mobilenet") if train_state["user_data"] else "Mobilenet"
    
    # 상태 메시지 표시 (첫 번째 행)
    status_label = ttk.Label(
        tab_train, 
        text=f"{username}님! {model_type} 모델을 이용한 학습을 하고 있습니다.", 
        font=DEFAULT_FONT
    )
    status_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 5))
    
    # 설명 텍스트 (두 번째 행)
    desc_label = ttk.Label(
        tab_train, 
        text="현재 상태와 hyperparameter입니다.", 
        font=DEFAULT_FONT
    )
    desc_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=(5, 10))
    
    # 진행 상태 표시 (새로 추가)
    progress_frame = ttk.Frame(tab_train)
    progress_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
    
    # 에포크 진행 표시 (좌측)
    epoch_label = ttk.Label(
        progress_frame, 
        text="에포크: 0/0", 
        font=DEFAULT_FONT
    )
    epoch_label.pack(side="left", padx=(0, 20))
    
    # Loss 표시 (중앙)
    loss_label = ttk.Label(
        progress_frame, 
        text="Loss: 0.000", 
        font=DEFAULT_FONT
    )
    loss_label.pack(side="left", padx=(0, 20))
    
    # Accuracy 표시 (우측)
    acc_label = ttk.Label(
        progress_frame, 
        text="Accuracy: 0.000", 
        font=DEFAULT_FONT
    )
    acc_label.pack(side="left")
    
    # 그래프 프레임 생성 (네 번째 행)
    home_frame = ttk.LabelFrame(tab_train, text="Parameters")
    home_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
    
    loss_frame = ttk.LabelFrame(tab_train, text="Loss")
    loss_frame.grid(row=3, column=1, sticky="nsew", padx=5, pady=5)
    
    acc_frame = ttk.LabelFrame(tab_train, text="Accuracy")
    acc_frame.grid(row=3, column=2, sticky="nsew", padx=5, pady=5)
    
    # Home 프레임 내용 - 사용자 데이터 표시
    param_canvas = tk.Canvas(home_frame, bg="white", highlightthickness=0)
    param_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # 사용자 데이터 텍스트 만들기
    param_text_content = ""
    if train_state["user_data"]:
        for key, value in train_state["user_data"].items():
            param_text_content += f"{key}: {value}\n"
    else:
        param_text_content = "아직 데이터가 로드되지 않았습니다."
    
    # 하이퍼파라미터 텍스트 추가
    param_text = param_canvas.create_text(
        10, 10, 
        text=param_text_content, 
        anchor="nw", 
        font=DEFAULT_FONT
    )
    
    # Loss 그래프 설정
    loss_canvas = tk.Canvas(loss_frame, bg="white", highlightthickness=0)
    loss_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    setup_loss_graph(loss_canvas)
    
    # Accuracy 그래프 설정
    acc_canvas = tk.Canvas(acc_frame, bg="white", highlightthickness=0)
    acc_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    setup_accuracy_graph(acc_canvas)
    
    # 파라미터 텍스트 업데이트 함수
    def update_param_text(data):
        text_content = ""
        train_state["user_data"] = data
        
        for key, value in data.items():
            text_content += f"{key}: {value}\n"
        
        param_canvas.delete("all")
        param_canvas.create_text(
            10, 10, 
            text=text_content, 
            anchor="nw", 
            font=DEFAULT_FONT
        )
    
    # 상태 업데이트 함수
    def update_status(new_text=None):
        if new_text:
            status_label.config(text=new_text)
        else:
            username = train_state["user_data"].get("Username", "사용자") if train_state["user_data"] else "사용자"
            model_type = train_state["user_data"].get("Model Type", "모델") if train_state["user_data"] else "모델"
            status_label.config(text=f"{username}님! {model_type} 모델을 이용한 학습을 하고 있습니다.")
    
    # 학습 모니터링 시작 함수
    def start_monitoring(new_log_dir=None, data=None):
        if train_state["monitoring"]:
            return
        
        if new_log_dir:
            train_state["log_dir"] = new_log_dir
        
        if data:
            train_state["user_data"] = data
            update_param_text(data)
            update_status()
        
        # init_info 파일 확인하여 총 에포크 수 설정
        init_info_files = glob.glob(os.path.join(train_state["log_dir"], "init_info_*.json"))
        if init_info_files:
            try:
                latest_file = sorted(init_info_files)[-1]
                with open(latest_file, 'r', encoding='utf-8') as f:
                    init_data = json.load(f)
                    if 'params_info' in init_data and 'epochs' in init_data['params_info']:
                        train_state["total_epochs"] = init_data['params_info']['epochs']
                        epoch_label.config(text=f"에포크: 0/{train_state['total_epochs']}")
            except Exception as e:
                print(f"Init info 로드 오류: {e}")
        
        train_state["monitoring"] = True
        
        # 모니터링 스레드 시작
        def monitoring_thread():
            while train_state["monitoring"]:
                try:
                    # epoch_log 파일 확인
                    epoch_log_files = glob.glob(os.path.join(train_state["log_dir"], "epoch_log_*.json"))
                    if epoch_log_files:
                        latest_file = sorted(epoch_log_files)[-1]
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            epoch_data = json.load(f)
                            
                        if 'epochs' in epoch_data and len(epoch_data['epochs']) > 0:
                            # 최신 에포크 정보 가져오기
                            latest_epoch = epoch_data['epochs'][-1]
                            epoch_num = latest_epoch.get('epoch', 0)
                            loss = latest_epoch.get('loss', 0)
                            val_loss = latest_epoch.get('val_loss', 0)
                            acc = latest_epoch.get('accuracy', latest_epoch.get('acc', 0))
                            val_acc = latest_epoch.get('val_accuracy', latest_epoch.get('val_acc', 0))
                            
                            # 진행 상황 업데이트
                            train_state["current_epoch"] = epoch_num + 1  # 0-based -> 1-based
                            train_state["last_loss"] = loss
                            train_state["last_accuracy"] = acc
                            
                            # UI 업데이트 (스레드 안전하게)
                            tab_train.after(0, lambda: epoch_label.config(
                                text=f"에포크: {train_state['current_epoch']}/{train_state['total_epochs']}"
                            ))
                            tab_train.after(0, lambda: loss_label.config(
                                text=f"Loss: {loss:.4f}"
                            ))
                            tab_train.after(0, lambda: acc_label.config(
                                text=f"Accuracy: {acc:.4f}"
                            ))
                            
                            # 그래프 업데이트
                            all_loss = [epoch.get('loss', 0) for epoch in epoch_data['epochs']]
                            all_acc = [epoch.get('accuracy', epoch.get('acc', 0)) 
                                      for epoch in epoch_data['epochs']]
                            
                            val_loss_data = None
                            if all('val_loss' in epoch for epoch in epoch_data['epochs']):
                                val_loss_data = [epoch.get('val_loss', 0) for epoch in epoch_data['epochs']]
                            
                            val_acc_data = None
                            if all(('val_accuracy' in epoch or 'val_acc' in epoch) for epoch in epoch_data['epochs']):
                                val_acc_data = [epoch.get('val_accuracy', epoch.get('val_acc', 0)) 
                                              for epoch in epoch_data['epochs']]
                            
                            tab_train.after(0, lambda: update_loss_graph(
                                loss_canvas, all_loss, val_loss_data
                            ))
                            tab_train.after(0, lambda: update_accuracy_graph(
                                acc_canvas, all_acc, val_acc_data
                            ))
                    
                    # 1초 대기
                    time.sleep(1)
                except Exception as e:
                    print(f"모니터링 오류: {e}")
                    time.sleep(5)  # 오류 발생 시 더 오래 대기
        
        threading.Thread(target=monitoring_thread, daemon=True).start()
    
    # 학습 모니터링 중지 함수
    def stop_monitoring():
        train_state["monitoring"] = False
    
    # 핸들러 반환
    return {
        "update_status": update_status,
        "update_param_text": update_param_text,
        "start_monitoring": start_monitoring,
        "stop_monitoring": stop_monitoring
    }

# Loss 그래프 설정
def setup_loss_graph(canvas):
    width, height = 200, 150
    
    # X축, Y축 그리기
    canvas.create_line(30, height-30, width-10, height-30, width=2)  # X축
    canvas.create_line(30, 10, 30, height-30, width=2)  # Y축
    
    # 축 레이블
    canvas.create_text(width/2, height-10, text="Epoch", font=DEFAULT_FONT)
    canvas.create_text(15, height/2, text="Loss", font=DEFAULT_FONT, angle=90)

# Loss 그래프 업데이트
def update_loss_graph(canvas, loss_data, val_loss_data=None):
    width, height = 200, 150
    
    # 기존 선 삭제
    canvas.delete("loss_line")
    canvas.delete("val_loss_line")
    
    if not loss_data:
        return
    
    # 최대값 계산 (최소값 0.1 보장)
    max_loss = max(max(loss_data) if loss_data else 0,
                  max(val_loss_data) if val_loss_data else 0, 
                  0.1)
    
    # 훈련 손실 그리기
    points = []
    epochs = len(loss_data)
    
    for i, loss in enumerate(loss_data):
        x = 30 + i * ((width-40) / (epochs-1 if epochs > 1 else 1))
        y = (height-30) - ((loss / max_loss) * (height-40))
        points.append(x)
        points.append(y)
    
    if len(points) >= 4:  # 최소 2개 포인트 필요
        canvas.create_line(points, fill="red", width=2, smooth=1, tags="loss_line")
    
    # 검증 손실 그리기 (있는 경우)
    if val_loss_data and len(val_loss_data) > 0:
        val_points = []
        
        for i, val_loss in enumerate(val_loss_data):
            x = 30 + i * ((width-40) / (epochs-1 if epochs > 1 else 1))
            y = (height-30) - ((val_loss / max_loss) * (height-40))
            val_points.append(x)
            val_points.append(y)
        
        if len(val_points) >= 4:
            canvas.create_line(val_points, fill="orange", width=2, smooth=1, 
                             dash=(4, 2), tags="val_loss_line")

# Accuracy 그래프 설정
def setup_accuracy_graph(canvas):
    width, height = 200, 150
    
    # X축, Y축 그리기
    canvas.create_line(30, height-30, width-10, height-30, width=2)  # X축
    canvas.create_line(30, 10, 30, height-30, width=2)  # Y축
    
    # 축 레이블
    canvas.create_text(width/2, height-10, text="Epoch", font=DEFAULT_FONT)
    canvas.create_text(15, height/2, text="Accuracy", font=DEFAULT_FONT, angle=90)

# Accuracy 그래프 업데이트
def update_accuracy_graph(canvas, acc_data, val_acc_data=None):
    width, height = 200, 150
    
    # 기존 선 삭제
    canvas.delete("acc_line")
    canvas.delete("val_acc_line")
    
    if not acc_data:
        return
    
    # 훈련 정확도 그리기
    points = []
    epochs = len(acc_data)
    
    for i, acc in enumerate(acc_data):
        x = 30 + i * ((width-40) / (epochs-1 if epochs > 1 else 1))
        y = (height-30) - (acc * (height-40))
        points.append(x)
        points.append(y)
    
    if len(points) >= 4:  # 최소 2개 포인트 필요
        canvas.create_line(points, fill="blue", width=2, smooth=1, tags="acc_line")
    
    # 검증 정확도 그리기 (있는 경우)
    if val_acc_data and len(val_acc_data) > 0:
        val_points = []
        
        for i, val_acc in enumerate(val_acc_data):
            x = 30 + i * ((width-40) / (epochs-1 if epochs > 1 else 1))
            y = (height-30) - (val_acc * (height-40))
            val_points.append(x)
            val_points.append(y)
        
        if len(val_points) >= 4:
            canvas.create_line(val_points, fill="cyan", width=2, smooth=1, 
                             dash=(4, 2), tags="val_acc_line")

# 메인 코드와 연결하는 부분
def integrate_with_main(tab_main, notebook, tab_train):
    """main 탭과 train 탭 연결"""
    # Train 탭 설정
    train_handlers = setup_train_tab(tab_train)
    
    def set_log_dir_callback(new_log_dir, data):
        # 메인 탭의 모든 위젯 비활성화
        for widget in tab_main.winfo_children():
            if hasattr(widget, 'configure') and hasattr(widget, 'cget'):
                if 'state' in widget.configure():
                    widget.configure(state="disabled")
        
        # Train 탭 데이터 업데이트 및 모니터링 시작
        train_handlers["update_status"](f"{data.get('Username', '사용자')}님! {data.get('Model Type', '모델')} 모델을 이용한 학습을 하고 있습니다.")
        train_handlers["update_param_text"](data)
        train_handlers["start_monitoring"](new_log_dir, data)
    
    # Main 탭 설정 함수로 콜백 전달
    return set_log_dir_callback