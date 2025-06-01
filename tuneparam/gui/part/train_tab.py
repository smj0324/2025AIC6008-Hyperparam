import tkinter as tk
from tkinter import ttk
import os
import json
import glob
import threading
import time
from tuneparam.gui.theme.fonts import DEFAULT_FONT,DEFAULT_FONT_2, ERROR_FONT_SMALL

# 그래프용 작은 폰트 정의
GRAPH_FONT = ('Helvetica', 9)

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

def setup_train_tab(tab_train, log_dir=None, user_data=None, is_dark_theme=False):
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
    tab_train.columnconfigure(0, weight=1, minsize=150)  # 파라미터 박스: 최소 150px 
    tab_train.columnconfigure(1, weight=2, minsize=250)  # Loss 박스: 파라미터 박스의 2배 비중
    tab_train.columnconfigure(2, weight=2, minsize=250)  # Accuracy 박스도 동일
    tab_train.rowconfigure(3, weight=1, minsize=240)

    # 사용자 데이터 가져오기
    username = train_state["user_data"].get("Username", "사용자") if train_state["user_data"] else "사용자"
    model_type = train_state["user_data"].get("Model Type", "Mobilenet") if train_state["user_data"] else "Mobilenet"
    
    # 상태 메시지 표시 (첫 번째 행)
    status_label = ttk.Label(
        tab_train, 
        text=f"{username}님! {model_type} 모델을 이용한 학습을 시작합니다.", 
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
    
    # 에포크 진행 표시
    epoch_label = ttk.Label(
        progress_frame, 
        text=f"에포크: 0/{train_state['total_epochs']}", 
        font=DEFAULT_FONT
    )
    epoch_label.pack(side="left", padx=(0, 20))
    
    # Loss 표시
    loss_label = ttk.Label(
        progress_frame, 
        text="Loss: 0.000 (검증: 0.000)", 
        font=DEFAULT_FONT
    )
    loss_label.pack(side="left", padx=(0, 20))
    
    # Accuracy 표시
    acc_label = ttk.Label(
        progress_frame, 
        text="Accuracy: 0.000 (검증: 0.000)", 
        font=DEFAULT_FONT
    )
    acc_label.pack(side="left")
    
    # 그래프 프레임 생성
    home_frame = ttk.LabelFrame(tab_train, text="Training Parameters")
    home_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
    home_frame.rowconfigure(0, weight=1)
    home_frame.columnconfigure(0, weight=1)
    
    loss_frame = ttk.LabelFrame(tab_train, text="Loss Graph")
    loss_frame.grid(row=3, column=1, sticky="nsew", padx=5, pady=5)
    loss_frame.rowconfigure(0, weight=1)
    loss_frame.columnconfigure(0, weight=1)
    
    acc_frame = ttk.LabelFrame(tab_train, text="Accuracy Graph")
    acc_frame.grid(row=3, column=2, sticky="nsew", padx=5, pady=5)
    acc_frame.rowconfigure(0, weight=1)
    acc_frame.columnconfigure(0, weight=1)
    
    # 파라미터 표시
    param_canvas = tk.Canvas(home_frame, bg=colors['canvas_bg'], highlightthickness=0)
    param_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # 초기 파라미터 텍스트 설정
    setup_initial_param_text(train_state, param_canvas)
    
    # Loss 그래프 설정
    loss_canvas = tk.Canvas(loss_frame, bg=colors['canvas_bg'], highlightthickness=0)
    loss_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    setup_loss_graph(loss_canvas, colors)
    
    # Accuracy 그래프 설정
    acc_canvas = tk.Canvas(acc_frame, bg=colors['canvas_bg'], highlightthickness=0)
    acc_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    setup_accuracy_graph(acc_canvas, colors)
    
    # 크기 변경 감지 추가
    add_resize_handlers(loss_canvas, acc_canvas, train_state)
    
    # 테마 업데이트 함수
    def update_theme(is_dark):
        train_state["is_dark_theme"] = is_dark
        train_state["colors"] = get_theme_colors(is_dark)
        colors = train_state["colors"]
        
        # Canvas 배경색 업데이트
        param_canvas.configure(bg=colors['canvas_bg'])
        loss_canvas.configure(bg=colors['canvas_bg'])
        acc_canvas.configure(bg=colors['canvas_bg'])
        
        # 파라미터 텍스트 업데이트
        update_param_text(train_state, param_canvas)
        
        # 그래프 재설정
        setup_loss_graph(loss_canvas, colors)
        setup_accuracy_graph(acc_canvas, colors)
        
        # 현재 데이터로 그래프 업데이트
        if train_state.get("current_loss_data"):
            update_loss_graph(loss_canvas, train_state["current_loss_data"], train_state.get("current_val_loss_data"))
        if train_state.get("current_acc_data"):
            update_accuracy_graph(acc_canvas, train_state["current_acc_data"], train_state.get("current_val_acc_data"))
    
    # 파라미터 텍스트 업데이트 함수
    def update_param_text(train_state, param_canvas, data=None):
        """파라미터 텍스트 업데이트"""
        text_content = "Training Configuration:\n\n"
        colors = train_state["colors"]
        
        if data:
            # 학습 관련 파라미터만 필터링
            relevant_params = {
                k: v for k, v in data.items() 
                if k in ["Epochs", "Batch Size", "Validation Split", "Learning Rate", "Optimizer", "Loss Function"]
            }
            for key, value in relevant_params.items():
                if value is not None:
                    text_content += f"{key}: {value}\n"
        else:
            params_info = train_state.get("params_info", {})
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
            font=(DEFAULT_FONT_2),
            fill=colors['fg'],
            tags="param_text"
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
        
        # init_info 파일 확인하여 총 에포크 수 설정
        init_info_files = glob.glob(os.path.join(train_state["log_dir"], "init_info_*.json"))
        if init_info_files:
            try:
                latest_file = sorted(init_info_files)[-1]
                with open(latest_file, 'r', encoding='utf-8') as f:
                    init_data = json.load(f)
                    if 'params_info' in init_data:
                        train_state["params_info"] = init_data['params_info']
                        train_state["total_epochs"] = init_data['params_info'].get('epochs', 0)
                        epoch_label.config(text=f"에포크: 0/{train_state['total_epochs']}")
                        
                        # 파라미터 텍스트 업데이트
                        update_param_text(train_state, param_canvas, data)
                        
            except Exception as e:
                print(f"❌ Init info 로드 오류: {e}")
        
        train_state["monitoring"] = True
        
        # 모니터링 스레드 시작
        def monitoring_thread():
            while train_state["monitoring"]:
                try:
                    # epoch_log 파일 확인
                    epoch_log_files = glob.glob(os.path.join(train_state["log_dir"], "epoch_log_*.jsonl"))
                    if epoch_log_files:
                        latest_file = sorted(epoch_log_files)[-1]
                        epochs_data = []
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    epoch_data = json.loads(line.strip())
                                    epochs_data.append(epoch_data)
                                except json.JSONDecodeError:
                                    continue
                        
                        if epochs_data:
                            # 최신 에포크 정보 가져오기
                            latest_epoch = epochs_data[-1]
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
                                text=f"Loss: {loss:.4f} (검증: {val_loss:.4f})"
                            ))
                            tab_train.after(0, lambda: acc_label.config(
                                text=f"Accuracy: {acc:.4f} (검증: {val_acc:.4f})"
                            ))
                            
                            # 파라미터 텍스트 업데이트 추가
                            tab_train.after(0, lambda: update_param_text(train_state, param_canvas))
                            
                            # 그래프 업데이트용 데이터 준비
                            all_loss = [epoch.get('loss', 0) for epoch in epochs_data]
                            all_acc = [epoch.get('accuracy', epoch.get('acc', 0)) 
                                      for epoch in epochs_data]
                            
                            val_loss_data = None
                            if all('val_loss' in epoch for epoch in epochs_data):
                                val_loss_data = [epoch.get('val_loss', 0) for epoch in epochs_data]
                            
                            val_acc_data = None
                            if all(('val_accuracy' in epoch or 'val_acc' in epoch) for epoch in epochs_data):
                                val_acc_data = [epoch.get('val_accuracy', epoch.get('val_acc', 0)) 
                                              for epoch in epochs_data]
                            
                            # 현재 데이터 저장
                            train_state["current_loss_data"] = all_loss
                            train_state["current_val_loss_data"] = val_loss_data
                            train_state["current_acc_data"] = all_acc
                            train_state["current_val_acc_data"] = val_acc_data
                            
                            # 그래프 업데이트
                            tab_train.after(0, lambda: update_loss_graph(
                                loss_canvas, all_loss, val_loss_data
                            ))
                            tab_train.after(0, lambda: update_accuracy_graph(
                                acc_canvas, all_acc, val_acc_data
                            ))
                            
                            # 학습이 완료되면 모니터링 중지
                            if train_state["current_epoch"] >= train_state["total_epochs"]:
                                train_state["monitoring"] = False
                                tab_train.after(0, lambda: status_label.config(
                                    text=f"✅ {train_state['user_data'].get('Username', '사용자')}님의 모델 학습이 완료되었습니다!"
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
        "update_param_text": lambda data=None: update_param_text(train_state, param_canvas, data),
        "start_monitoring": start_monitoring,
        "stop_monitoring": stop_monitoring,
        "update_theme": update_theme
    }

def setup_loss_graph(canvas, colors):
    """Loss 그래프 초기 설정"""
    canvas.update_idletasks()
    width = canvas.winfo_reqwidth() or 290
    height = canvas.winfo_reqheight() or 200
    
    # 배경 그리드는 초기화 시에만 그리기
    canvas.delete("all")
    
    # 배경 그리드
    #for i in range(50, width-80, 30):
    #    canvas.create_line(i, 20, i, height-40, fill=colors['grid'], width=1, tags="background")
    #for i in range(height-40, 20, -30):
    #    canvas.create_line(50, i, width-80, i, fill=colors['grid'], width=1, tags="background")
    
    # 축 그리기
    #canvas.create_line(50, height-40, width-80, height-40, width=2, fill=colors['axis'], tags="background")
    #canvas.create_line(50, 20, 50, height-40, width=2, fill=colors['axis'], tags="background")
    
    # 축 레이블
    #canvas.create_text(width/2, height-10, text="Epoch", font=GRAPH_FONT, fill=colors['fg'], tags="background")
    #canvas.create_text(15, height/2, text="Loss", font=GRAPH_FONT, angle=90, fill=colors['fg'], tags="background")
    # 상단/하단 여백 값과 좌/우 여백을 일관되게 정의
    top = 15
    bottom = 60
    left = 50
    right = 20

    # 배경 그리드
    for i in range(left, width - right - 30, 30):
        canvas.create_line(i, top, i, height - bottom, fill=colors['grid'], width=1, tags="background")
    for i in range(height - bottom, top, -30):
        canvas.create_line(left, i, width - right, i, fill=colors['grid'], width=1, tags="background")

    # 축 그리기 (x축: height-bottom, y축: left~height-bottom)
    canvas.create_line(left, height - bottom, width - right, height - bottom,
                       width=2, fill=colors['axis'], tags="background")
    canvas.create_line(left, top, left, height - bottom,
                       width=2, fill=colors['axis'], tags="background")

    # 축 레이블
    canvas.create_text((width - right + left) / 2, height - bottom + 20,
                       text="Epoch", font=GRAPH_FONT, fill=colors['fg'], tags="background")
    canvas.create_text(15, (height - bottom + top) / 2,
                       text="Loss", font=GRAPH_FONT, angle=90, fill=colors['fg'], tags="background")

def update_loss_graph(canvas, loss_data, val_loss_data=None):
    """Loss 그래프 업데이트 - 기존 여백 설정 유지"""
    # 캔버스 크기를 안전하게 가져오기
    canvas.update_idletasks()
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    
    # 크기가 유효하지 않으면 기본값 사용
    if width <= 1:
        width = canvas.winfo_reqwidth() or 290
    if height <= 1:
        height = canvas.winfo_reqheight() or 200
        
    colors = get_theme_colors(canvas.cget('bg') == '#2b2b2b')
    
    # 기존 코드와 동일한 여백 설정
    left, right, top, bottom = 50, 20, 15, 60
    
    plot_w = width - left - right
    plot_h = height - top - bottom
    
    # 기존 요소 삭제
    canvas.delete("loss_line", "val_loss_line", "loss_point", "val_loss_point", 
                  "y_labels", "x_labels", "legend")
    
    if not loss_data:
        return
    
    # 최대값과 최소값 계산 (기존 로직 유지)
    all_losses = loss_data + (val_loss_data if val_loss_data else [])
    max_loss = max(max(all_losses), 0.1)
    min_loss = min(min(all_losses), max_loss * 0.9)
    
    # Y축 값 표시 (기존 방식)
    num_y_labels = 5
    for i in range(num_y_labels):
        y_val = min_loss + (max_loss - min_loss) * (num_y_labels - 1 - i) / (num_y_labels - 1)
        y_pos = 10 + i * (height - 35) / (num_y_labels - 1)
        canvas.create_text(45, y_pos, text=f"{y_val:.2f}", anchor="e", 
                         tags="y_labels", font=GRAPH_FONT, fill=colors['fg'])
    
    # X축 값 표시 (기존 방식)
    epochs = len(loss_data)
    num_ticks = min(6, epochs)
    for t in range(1, num_ticks+1):
        # 1~epochs 범위에서 균등 분할 지점 계산
        idx = int(round(t * epochs / num_ticks)) - 1
        idx = max(0, min(idx, epochs-1))
        # 50 과 width-60 사이 plot_w 만큼 비율대로 이동
        x_pos = 50 + idx * ((width - 60) / (epochs-1 if epochs > 1 else 1))
        # 1-based label
        canvas.create_text(
            x_pos, height - 15,
            text=str(idx+1),
            tags="x_labels",
            font=GRAPH_FONT,
            fill=colors['fg']
        )
    
    # 훈련 손실 그리기 (기존 방식)
    points = []
    for i, loss in enumerate(loss_data):
        x = 50 + i * ((width-60) / (epochs-1 if epochs > 1 else 1))
        y = (height-25) - ((loss - min_loss) / (max_loss - min_loss) * (height-35))
        points.append(x)
        points.append(y)
        canvas.create_oval(x-2, y-2, x+2, y+2, fill=colors['loss'], tags="loss_point")
    
    if len(points) >= 4:
        canvas.create_line(points, fill=colors['loss'], width=2, smooth=0, tags="loss_line")
    
    # 검증 손실 그리기 (기존 방식)
    if val_loss_data and len(val_loss_data) > 0:
        val_points = []
        for i, val_loss in enumerate(val_loss_data):
            x = 50 + i * ((width-60) / (epochs-1 if epochs > 1 else 1))
            y = (height-25) - ((val_loss - min_loss) / (max_loss - min_loss) * (height-35))
            val_points.append(x)
            val_points.append(y)
            canvas.create_oval(x-2, y-2, x+2, y+2, fill=colors['val_loss'], tags="val_loss_point")
        
        if len(val_points) >= 4:
            canvas.create_line(val_points, fill=colors['val_loss'], width=2, smooth=1, 
                             dash=(4, 2), tags="val_loss_line")
    
    # 기존 스타일 범례 그리기 (간단한 형태)
    legend_items = [
        ("Train Loss", colors['loss'], ()),
        ("Val   Loss", colors['val_loss'], (4,2))
    ]

    # 여유(margin) 및 텍스트 폭 계산
    padding = 2
    line_len = 10
    text_pad = 3
    item_height = 16
    max_text_chars = max(len(label) for label, _, _ in legend_items)
    text_width = max_text_chars * 6

    # 박스 크기/위치 계산 (캔버스 우측 위)
    canvas_w = width
    box_w = padding*2 + line_len + text_pad + text_width
    box_h = padding*2 + len(legend_items)*item_height
    box_x = canvas_w - box_w - padding
    box_y = padding

    # 박스와 항목 그리기
    canvas.create_rectangle(
        box_x, box_y, box_x + box_w, box_y + box_h,
        fill=colors['canvas_bg'], outline=colors['axis'], width=1, tags="legend"
    )
    for i, (label, col, dash) in enumerate(legend_items):
        y = box_y + padding + i*item_height + item_height//2
        # 컬러라인
        canvas.create_line(
            box_x + padding, y,
            box_x + padding + line_len, y,
            fill=col, width=2, dash=dash, tags="legend"
        )
        # 텍스트
        canvas.create_text(
            box_x + padding + line_len + text_pad, y,
            text=label, anchor="w",
            font=GRAPH_FONT, fill=colors['fg'], tags="legend"
        )

def setup_accuracy_graph(canvas, colors):
    """Accuracy 그래프 초기 설정"""
    canvas.update_idletasks()
    width = canvas.winfo_reqwidth() or 290
    height = canvas.winfo_reqheight() or 200
    
    canvas.delete("all")
    
    # 배경 그리드
    #for i in range(50, width-80, 30):
    #    canvas.create_line(i, 20, i, height-40, fill=colors['grid'], width=1, tags="background")
    #for i in range(height-40, 20, -30):
    #    canvas.create_line(50, i, width-80, i, fill=colors['grid'], width=1, tags="background")
    
    # 축 그리기
    #canvas.create_line(50, height-40, width-80, height-40, width=2, fill=colors['axis'], tags="background")
    #canvas.create_line(50, 20, 50, height-40, width=2, fill=colors['axis'], tags="background")
    
    # 축 레이블
    #canvas.create_text(width/2, height-10, text="Epoch", font=GRAPH_FONT, fill=colors['fg'], tags="background")
    #canvas.create_text(15, height/2, text="Accuracy", font=GRAPH_FONT, angle=90, fill=colors['fg'], tags="background")
    # 여백 정의 (Loss와 동일)
    left, right, top, bottom = 50, 20, 15, 60

    # 배경 그리드
    for i in range(left, width - right - 30, 30):
        canvas.create_line(i, top, i, height - bottom, fill=colors['grid'], width=1, tags="background")
    for i in range(height - bottom, top, -30):
        canvas.create_line(left, i, width - right, i, fill=colors['grid'], width=1, tags="background")

    # 축 그리기
    canvas.create_line(left, height - bottom, width - right, height - bottom,
                       width=2, fill=colors['axis'], tags="background")
    canvas.create_line(left, top, left, height - bottom,
                       width=2, fill=colors['axis'], tags="background")

    # 축 레이블
    canvas.create_text((width - right + left) / 2, height - bottom + 20,
                       text="Epoch", font=GRAPH_FONT, fill=colors['fg'], tags="background")
    canvas.create_text(15, (height - bottom + top) / 2,
                       text="Accuracy", font=GRAPH_FONT, angle=90, fill=colors['fg'], tags="background")
def update_accuracy_graph(canvas, acc_data, val_acc_data=None):
    """Accuracy 그래프 업데이트 - 기존 방식 유지"""
    # 캔버스 크기를 안전하게 가져오기
    canvas.update_idletasks()
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    
    if width <= 1:
        width = canvas.winfo_reqwidth() or 290
    if height <= 1:
        height = canvas.winfo_reqheight() or 200
        
    colors = get_theme_colors(canvas.cget('bg') == '#2b2b2b')
    
    # 기존 코드와 동일한 여백 설정
    left, right, top, bottom = 50, 20, 10, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    
    # 기존 요소 삭제 (legend 태그 포함)
    canvas.delete("acc_line","val_acc_line",
                  "acc_point","val_acc_point",
                  "x_labels","y_labels","legend")

    if not acc_data:
        return
    
    # Y축 눈금 숫자 (기존 방식)
    num_y = 5
    for i in range(num_y):
        y_val = i / (num_y - 1)
        y = height - bottom - (i / (num_y-1)) * plot_h
        canvas.create_text(
            left-5, y,
            text=f"{y_val:.1f}",
            anchor="e",
            tags="y_labels",
            font=GRAPH_FONT,
            fill=colors['fg']
        )

    # Accuracy 곡선 계산 (기존 방식)
    epochs = len(acc_data)
    # 훈련 정확도
    pts_acc = []
    for i, v in enumerate(acc_data):
        x = left + (i/(epochs-1 if epochs>1 else 1)) * plot_w
        y = height - bottom - (v * plot_h)
        pts_acc += [x, y]
        canvas.create_oval(x-2, y-2, x+2, y+2,
                           fill=colors['acc'], tags="acc_point")
    canvas.create_line(pts_acc, fill=colors['acc'], width=2,
                       smooth=0, tags="acc_line")

    # 검증 정확도 (기존 방식)
    if val_acc_data:
        pts_val = []
        for i, v in enumerate(val_acc_data):
            x = left + (i/(epochs-1 if epochs>1 else 1)) * plot_w
            y = height - bottom - (v * plot_h)
            pts_val += [x, y]
            canvas.create_oval(x-2, y-2, x+2, y+2,
                               fill=colors['val_acc'], tags="val_acc_point")
        canvas.create_line(pts_val, fill=colors['val_acc'],
                           width=2, dash=(4,2), smooth=0,
                           tags="val_acc_line")

    # X축 tick 6개 균등 분할 (기존 방식)
    num_ticks = min(6, epochs)
    for t in range(1, num_ticks+1):
        idx = int(round(t * epochs / num_ticks)) - 1
        idx = max(0, min(idx, epochs-1))
        x = left + (idx/(epochs-1 if epochs>1 else 1)) * plot_w
        canvas.create_text(x, height-15,
                           text=str(idx+1),
                           tags="x_labels",
                           font=GRAPH_FONT,
                           fill=colors['fg'])

    # 기존 스타일 범례 그리기 (간단한 형태)
    legend_items = [
        ("Train Acc", colors['acc'], ()),
        ("Val   Acc", colors['val_acc'], (4,2))
    ]
    
    pad = 2
    ln_len = 10
    txt_pad = 3
    ih = 16
    maxc = max(len(l) for l,_,_ in legend_items)
    tw = maxc * 6
    bw = pad*2 + ln_len + txt_pad + tw
    bh = pad*2 + len(legend_items)*ih
    bx = width - bw - pad
    by = pad
    
    canvas.create_rectangle(bx, by, bx+bw, by+bh,
                            fill=colors['canvas_bg'],
                            outline=colors['axis'], tags="legend")
    for i, (lbl, col, dash) in enumerate(legend_items):
        yy = by + pad + i*ih + ih//2
        canvas.create_line(bx+pad, yy, bx+pad+ln_len, yy,
                           fill=col, width=2, dash=dash, tags="legend")
        canvas.create_text(bx+pad+ln_len+txt_pad, yy,
                           text=lbl, anchor="w",
                           font=GRAPH_FONT, fill=colors['fg'],
                           tags="legend")


# 크기 변경 감지 함수 추가
def add_resize_handlers(loss_canvas, acc_canvas, train_state):
    """캔버스 크기 변경 감지 및 자동 업데이트"""
    def on_loss_resize(event):
        # 짧은 지연 후 업데이트 (연속 리사이즈 방지)
        if train_state.get("current_loss_data"):
            loss_canvas.after(100, lambda: update_loss_graph(
                loss_canvas,
                train_state["current_loss_data"],
                train_state.get("current_val_loss_data")
            ))
    
    def on_acc_resize(event):
        if train_state.get("current_acc_data"):
            acc_canvas.after(100, lambda: update_accuracy_graph(
                acc_canvas,
                train_state["current_acc_data"],
                train_state.get("current_val_acc_data")
            ))
    
    # 크기 변경 이벤트 바인딩
    loss_canvas.bind('<Configure>', on_loss_resize)
    acc_canvas.bind('<Configure>', on_acc_resize)


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


# 초기 파라미터 텍스트 설정
def setup_initial_param_text(train_state, param_canvas):
    """초기 파라미터 텍스트 설정"""
    text_content = "Training Configuration:\n\n"
    params_info = train_state.get("params_info", {})
    colors = train_state["colors"]
    
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
        fill=colors['fg'],
        tags="param_text"
    )