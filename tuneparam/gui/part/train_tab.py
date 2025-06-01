import tkinter as tk
from tkinter import ttk
import os
import json
import glob
import threading
import time
from tuneparam.gui.theme.fonts import DEFAULT_FONT, DEFAULT_FONT_2

# 그래프용 작은 폰트 정의
GRAPH_FONT = ('Helvetica', 8)

def get_theme_colors(is_dark_theme=False):
    """테마에 따른 색상 반환"""
    if is_dark_theme:
        return {
            'bg': "#2b2b2b",
            'fg': "#ffffff",
            'grid': "#404040",
            'axis': "#ffffff",
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
    # ────────────────────────────────────────────────────────────────────
    # 1) train_state 초기화, 그리고 즉시 init_info_*.json 로드
    # ────────────────────────────────────────────────────────────────────
    train_state = {
        "monitoring": False,
        "log_dir": log_dir,
        "user_data": user_data,
        "current_epoch": 0,
        "total_epochs": 0,
        "last_loss": 0,
        "last_accuracy": 0,
        "params_info": {},        # 여기서 파라미터가 채워질 예정
        "is_dark_theme": is_dark_theme,
        "colors": get_theme_colors(is_dark_theme)
    }
    colors = train_state["colors"]

    # ▶ init_info_*.json 파일 로드 (탐색하여 가장 마지막 파일을 읽는다)
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

    # ────────────────────────────────────────────────────────────────────
    # 2) 탭 레이아웃 구성 (3열: 파라미터 / Loss 그래프 / Accuracy 그래프)
    # ────────────────────────────────────────────────────────────────────
    tab_train.columnconfigure(0, weight=1, minsize=150)
    tab_train.columnconfigure(1, weight=2, minsize=250)
    tab_train.columnconfigure(2, weight=2, minsize=250)
    tab_train.rowconfigure(3, weight=1, minsize=240)

    username = train_state["user_data"].get("Username", "사용자") if train_state["user_data"] else "사용자"
    model_type = train_state["user_data"].get("Model Type", "Mobilenet") if train_state["user_data"] else "Mobilenet"

    # 상태 메시지
    status_label = ttk.Label(
        tab_train,
        text=f"{username}님! {model_type} 모델을 이용한 학습을 시작합니다.",
        font=DEFAULT_FONT
    )
    status_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 5))

    # 설명 텍스트
    desc_label = ttk.Label(
        tab_train,
        text="현재 학습 상태와 파라미터입니다.",
        font=DEFAULT_FONT
    )
    desc_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=(5, 10))

    # 진행 상태 표시 (에포크 / Loss / Accuracy)
    progress_frame = ttk.Frame(tab_train)
    progress_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 10))

    epoch_label = ttk.Label(
        progress_frame,
        text=f"에포크: 0/{train_state['total_epochs']}",
        font=DEFAULT_FONT
    )
    epoch_label.pack(side="left", padx=(0, 20))

    loss_label = ttk.Label(
        progress_frame,
        text="Loss: 0.000 (검증: 0.000)",
        font=DEFAULT_FONT
    )
    loss_label.pack(side="left", padx=(0, 20))

    acc_label = ttk.Label(
        progress_frame,
        text="Accuracy: 0.000 (검증: 0.000)",
        font=DEFAULT_FONT
    )
    acc_label.pack(side="left")


    # ────────────────────────────────────────────────────────────────────
    # 3) 왼쪽: Training Parameters 영역
    # ────────────────────────────────────────────────────────────────────
    home_frame = ttk.LabelFrame(tab_train, text="Training Parameters")
    home_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
    home_frame.rowconfigure(0, weight=1)
    home_frame.columnconfigure(0, weight=1)

    param_canvas = tk.Canvas(home_frame, bg=colors['canvas_bg'], highlightthickness=0)
    param_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ▶ init_info 로 읽어온 params_info가 이미 있으므로, 최초에 바로 화면에 그려주기
    setup_initial_param_text(train_state, param_canvas)


    # ────────────────────────────────────────────────────────────────────
    # 4) 가운데: Loss 그래프 영역
    # ────────────────────────────────────────────────────────────────────
    loss_frame = ttk.LabelFrame(tab_train, text="Loss Graph")
    loss_frame.grid(row=3, column=1, sticky="nsew", padx=5, pady=5)
    loss_frame.rowconfigure(0, weight=1)
    loss_frame.columnconfigure(0, weight=1)

    # ★ 고정 크기(width, height) 제거 ★
    loss_canvas = tk.Canvas(loss_frame, bg=colors['canvas_bg'], highlightthickness=0)
    loss_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # 리사이즈되거나 탭 처음 생성될 때마다 “setup → update” 순서로 그리기
    def redraw_loss_canvas(event=None):
        setup_loss_graph(loss_canvas, colors)
        if train_state.get("current_loss_data"):
            update_loss_graph(
                loss_canvas,
                train_state["current_loss_data"],
                train_state.get("current_val_loss_data")
            )

    loss_canvas.bind("<Configure>", redraw_loss_canvas)
    redraw_loss_canvas()  # 최초 한 번 그려 주기


    # ────────────────────────────────────────────────────────────────────
    # 5) 오른쪽: Accuracy 그래프 영역
    # ────────────────────────────────────────────────────────────────────
    acc_frame = ttk.LabelFrame(tab_train, text="Accuracy Graph")
    acc_frame.grid(row=3, column=2, sticky="nsew", padx=5, pady=5)
    acc_frame.rowconfigure(0, weight=1)
    acc_frame.columnconfigure(0, weight=1)

    acc_canvas = tk.Canvas(acc_frame, bg=colors['canvas_bg'], highlightthickness=0)
    acc_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def redraw_acc_canvas(event=None):
        setup_accuracy_graph(acc_canvas, colors)
        if train_state.get("current_acc_data"):
            update_accuracy_graph(
                acc_canvas,
                train_state["current_acc_data"],
                train_state.get("current_val_acc_data")
            )

    acc_canvas.bind("<Configure>", redraw_acc_canvas)
    redraw_acc_canvas()


    # ────────────────────────────────────────────────────────────────────
    # 6) 이후: 모니터링 스레드, 상태 업데이트 등 (원본 코드와 거의 동일)
    # ────────────────────────────────────────────────────────────────────
    def update_status(new_text=None):
        if new_text:
            status_label.config(text=new_text)
        else:
            username = train_state["user_data"].get("Username", "사용자") if train_state["user_data"] else "사용자"
            model_type = train_state["user_data"].get("Model Type", "모델") if train_state["user_data"] else "모델"
            status_label.config(text=f"{username}님! {model_type} 모델을 이용한 학습을 하고 있습니다.")

    def start_monitoring(new_log_dir=None, data=None):
        if train_state["monitoring"]:
            return

        if new_log_dir:
            train_state["log_dir"] = new_log_dir
        if data:
            train_state["user_data"] = data

        # init_info 파일 다시 확인해서 total_epochs, params_info 업데이트
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
                        # ★ init_info가 변하면 바로 파라미터 캔버스에 그려 주기
                        update_param_text(train_state, param_canvas, data)
            except Exception as e:
                print(f"❌ Init info 로드 오류: {e}")

        train_state["monitoring"] = True

        # 모니터링 스레드
        def monitoring_thread():
            while train_state["monitoring"]:
                try:
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
                            latest_epoch = epochs_data[-1]
                            epoch_num = latest_epoch.get('epoch', 0)
                            loss = latest_epoch.get('loss', 0)
                            val_loss = latest_epoch.get('val_loss', 0)
                            acc = latest_epoch.get('accuracy', latest_epoch.get('acc', 0))
                            val_acc = latest_epoch.get('val_accuracy', latest_epoch.get('val_acc', 0))

                            train_state["current_epoch"] = epoch_num + 1
                            train_state["last_loss"] = loss
                            train_state["last_accuracy"] = acc

                            # UI 업데이트
                            tab_train.after(0, lambda: epoch_label.config(
                                text=f"에포크: {train_state['current_epoch']}/{train_state['total_epochs']}"
                            ))
                            tab_train.after(0, lambda: loss_label.config(
                                text=f"Loss: {loss:.4f} (검증: {val_loss:.4f})"
                            ))
                            tab_train.after(0, lambda: acc_label.config(
                                text=f"Accuracy: {acc:.4f} (검증: {val_acc:.4f})"
                            ))
                            tab_train.after(0, lambda: update_param_text(train_state, param_canvas))

                            # 데이터 준비
                            all_loss = [epoch.get('loss', 0) for epoch in epochs_data]
                            all_acc = [epoch.get('accuracy', epoch.get('acc', 0)) for epoch in epochs_data]

                            val_loss_data = None
                            if all('val_loss' in epoch for epoch in epochs_data):
                                val_loss_data = [epoch.get('val_loss', 0) for epoch in epochs_data]

                            val_acc_data = None
                            if all(('val_accuracy' in epoch or 'val_acc' in epoch) for epoch in epochs_data):
                                val_acc_data = [epoch.get('val_accuracy', epoch.get('val_acc', 0)) for epoch in epochs_data]

                            train_state["current_loss_data"] = all_loss
                            train_state["current_val_loss_data"] = val_loss_data
                            train_state["current_acc_data"] = all_acc
                            train_state["current_val_acc_data"] = val_acc_data

                            # 크로스-스레딩으로 그래프 업데이트
                            tab_train.after(0, lambda: update_loss_graph(
                                loss_canvas, all_loss, val_loss_data
                            ))
                            tab_train.after(0, lambda: update_accuracy_graph(
                                acc_canvas, all_acc, val_acc_data
                            ))

                            # 학습 완료 시
                            if train_state["current_epoch"] >= train_state["total_epochs"]:
                                train_state["monitoring"] = False
                                tab_train.after(0, lambda: status_label.config(
                                    text=f"✅ {train_state['user_data'].get('Username', '사용자')}님의 모델 학습이 완료되었습니다!"
                                ))

                    time.sleep(1)
                except Exception as e:
                    print(f"모니터링 오류: {e}")
                    time.sleep(5)

        threading.Thread(target=monitoring_thread, daemon=True).start()

    def stop_monitoring():
        train_state["monitoring"] = False

    def update_theme(is_dark):
        train_state["is_dark_theme"] = is_dark
        train_state["colors"] = get_theme_colors(is_dark)
        new_colors = train_state["colors"]

        # 캔버스 배경색
        param_canvas.configure(bg=new_colors['canvas_bg'])
        loss_canvas.configure(bg=new_colors['canvas_bg'])
        acc_canvas.configure(bg=new_colors['canvas_bg'])

        # 파라미터 텍스트도 업데이트
        update_param_text(train_state, param_canvas)

        # 그래프 배경(격자+축) 먼저 다시 그리기
        setup_loss_graph(loss_canvas, new_colors)
        setup_accuracy_graph(acc_canvas, new_colors)

        # 데이터가 있으면 다시 그리기
        if train_state.get("current_loss_data"):
            update_loss_graph(loss_canvas, train_state["current_loss_data"], train_state.get("current_val_loss_data"))
        if train_state.get("current_acc_data"):
            update_accuracy_graph(acc_canvas, train_state["current_acc_data"], train_state.get("current_val_acc_data"))

    return {
        "update_status": update_status,
        "update_param_text": lambda data=None: update_param_text(train_state, param_canvas, data),
        "start_monitoring": start_monitoring,
        "stop_monitoring": stop_monitoring,
        "update_theme": update_theme
    }


def setup_initial_param_text(train_state, param_canvas):
    """초기 파라미터 텍스트 설정 (init_info_*.json에서 읽어서 param_canvas에 바로 그려줌)"""
    text_content = "Training Configuration:\n\n"
    params_info = train_state.get("params_info", {})
    colors = train_state["colors"]

    if params_info:
        training_params = {
            "Epochs": params_info.get("epochs", "N/A"),
            "Batch Size": params_info.get("batch_size", "N/A"),
            "Validation Split": f"{params_info.get('validation_split', 0) * 100}%",
            "Learning Rate": params_info.get("learning_rate", "N/A"),
            "Optimizer": params_info.get("optimizer", "N/A"),
            "Loss Function": params_info.get("loss", "N/A")
        }
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
        font=DEFAULT_FONT_2,
        fill=colors['fg'],
        tags="param_text"
    )


def update_param_text(train_state, param_canvas, data=None):
    """학습 중 파라미터가 갱신될 때마다 param_canvas를 업데이트"""
    text_content = "Training Configuration:\n\n"
    colors = train_state["colors"]

    if data:  # 사용자가 넘겨준 새 데이터(예: Main 탭에서 넘어올 때)
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
        font=DEFAULT_FONT_2,
        fill=colors['fg'],
        tags="param_text"
    )


def setup_loss_graph(canvas, colors):
    """Loss 그래프 배경(격자+축)을 그리기"""
    canvas.update_idletasks()
    width = canvas.winfo_width() or 1
    height = canvas.winfo_height() or 1

    canvas.delete("all")

    left, right, top, bottom = 50, 20, 15, 35
    graph_left = left
    graph_right = width - right
    graph_top = top
    graph_bottom = height - bottom

    graph_width = graph_right - graph_left
    graph_height = graph_bottom - graph_top

    # 격자
    for x in range(graph_left, graph_right + 1, 30):
        canvas.create_line(x, graph_top, x, graph_bottom, fill=colors['grid'], width=1, tags="background")
    for y in range(graph_top, graph_bottom + 1, 30):
        canvas.create_line(graph_left, y, graph_right, y, fill=colors['grid'], width=1, tags="background")

    # 축
    canvas.create_line(graph_left, graph_bottom, graph_right, graph_bottom,
                       width=2, fill=colors['axis'], tags="background")
    canvas.create_line(graph_left, graph_top, graph_left, graph_bottom,
                       width=2, fill=colors['axis'], tags="background")

    # 축 레이블
    canvas.create_text((graph_left + graph_right) / 2, graph_bottom + 25,
                       text="Epoch", font=GRAPH_FONT, fill=colors['fg'], tags="background")
    canvas.create_text(graph_left - 30, (graph_top + graph_bottom) / 2,
                       text="Loss", font=GRAPH_FONT, angle=90, fill=colors['fg'], tags="background")


def update_loss_graph(canvas, loss_data, val_loss_data=None):
    """Loss 데이터(점+선)를 그리기"""
    canvas.update_idletasks()
    width = canvas.winfo_width() or 1
    height = canvas.winfo_height() or 1

    colors = get_theme_colors(canvas.cget('bg') == '#2b2b2b')

    left, right, top, bottom = 50, 20, 15, 35
    graph_left = left
    graph_right = width - right
    graph_top = top
    graph_bottom = height - bottom

    graph_width = graph_right - graph_left
    graph_height = graph_bottom - graph_top

    canvas.delete("loss_line", "val_loss_line",
                  "loss_point", "val_loss_point",
                  "y_labels", "x_labels", "legend")

    if not loss_data:
        return

    all_losses = loss_data + (val_loss_data if val_loss_data else [])
    max_loss = max(max(all_losses), 0.1)
    min_loss = min(min(all_losses), max_loss * 0.9)

    # Y축 눈금
    num_y = 5
    for i in range(num_y):
        y_val = min_loss + (max_loss - min_loss) * (num_y - 1 - i) / (num_y - 1)
        y_pos = graph_top + (i / (num_y - 1)) * graph_height
        canvas.create_text(graph_left - 5, y_pos,
                           text=f"{y_val:.2f}",
                           anchor="e",
                           tags="y_labels",
                           font=GRAPH_FONT,
                           fill=colors['fg'])

    # X축 눈금
    epochs = len(loss_data)
    num_ticks = min(6, epochs)
    for t in range(1, num_ticks + 1):
        idx = int(round(t * epochs / num_ticks)) - 1
        idx = max(0, min(idx, epochs - 1))
        x_pos = graph_left + (idx / (epochs - 1 if epochs > 1 else 1)) * graph_width
        canvas.create_text(x_pos, graph_bottom + 15,
                           text=str(idx + 1),
                           tags="x_labels",
                           font=GRAPH_FONT,
                           fill=colors['fg'])

    # Train Loss 그리기
    pts_train = []
    for i, loss in enumerate(loss_data):
        x = graph_left + (i / (epochs - 1 if epochs > 1 else 1)) * graph_width
        y = graph_bottom - ((loss - min_loss) / (max_loss - min_loss) * graph_height)
        x = max(graph_left, min(graph_right, x))
        y = max(graph_top, min(graph_bottom, y))
        pts_train += [x, y]
        canvas.create_oval(x - 2, y - 2, x + 2, y + 2,
                           fill=colors['loss'], tags="loss_point")

    if len(pts_train) >= 4:
        canvas.create_line(pts_train, fill=colors['loss'], width=2, smooth=0, tags="loss_line")

    # Val Loss 그리기 (옵션)
    if val_loss_data:
        pts_val = []
        for i, val in enumerate(val_loss_data):
            x = graph_left + (i / (epochs - 1 if epochs > 1 else 1)) * graph_width
            y = graph_bottom - ((val - min_loss) / (max_loss - min_loss) * graph_height)
            x = max(graph_left, min(graph_right, x))
            y = max(graph_top, min(graph_bottom, y))
            pts_val += [x, y]
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2,
                               fill=colors['val_loss'], tags="val_loss_point")

        if len(pts_val) >= 4:
            canvas.create_line(pts_val, fill=colors['val_loss'],
                               width=2, dash=(4, 2), smooth=0,
                               tags="val_loss_line")

    # 범례
    legend_items = [
        ("Train Loss", colors['loss'], ()),
        ("Val   Loss", colors['val_loss'], (4, 2))
    ]
    padding = 2
    line_len = 10
    text_pad = 3
    item_height = 16
    max_text_chars = max(len(label) for label, _, _ in legend_items)
    text_width = max_text_chars * 6

    canvas_w = width
    box_w = padding*2 + line_len + text_pad + text_width
    box_h = padding*2 + len(legend_items)*item_height
    box_x = canvas_w - box_w - padding
    box_y = padding

    canvas.create_rectangle(
        box_x, box_y, box_x + box_w, box_y + box_h,
        fill=colors['canvas_bg'], outline=colors['axis'], width=1, tags="legend"
    )
    for i, (label, col, dash) in enumerate(legend_items):
        y = box_y + padding + i*item_height + item_height//2
        canvas.create_line(
            box_x + padding, y,
            box_x + padding + line_len, y,
            fill=col, width=2, dash=dash, tags="legend"
        )
        canvas.create_text(
            box_x + padding + line_len + text_pad, y,
            text=label, anchor="w",
            font=GRAPH_FONT, fill=colors['fg'], tags="legend"
        )


def setup_accuracy_graph(canvas, colors):
    """Accuracy 그래프 배경(격자+축) 그리기"""
    canvas.update_idletasks()
    width = canvas.winfo_width() or 1
    height = canvas.winfo_height() or 1

    canvas.delete("all")

    left, right, top, bottom = 50, 20, 15, 35
    graph_left = left
    graph_right = width - right
    graph_top = top
    graph_bottom = height - bottom

    graph_width = graph_right - graph_left
    graph_height = graph_bottom - graph_top

    for x in range(graph_left, graph_right + 1, 30):
        canvas.create_line(x, graph_top, x, graph_bottom,
                           fill=colors['grid'], width=1, tags="background")
    for y in range(graph_top, graph_bottom + 1, 30):
        canvas.create_line(graph_left, y, graph_right, y,
                           fill=colors['grid'], width=1, tags="background")

    canvas.create_line(graph_left, graph_bottom, graph_right, graph_bottom,
                       width=2, fill=colors['axis'], tags="background")
    canvas.create_line(graph_left, graph_top, graph_left, graph_bottom,
                       width=2, fill=colors['axis'], tags="background")

    canvas.create_text((graph_left + graph_right) / 2, graph_bottom + 25,
                       text="Epoch", font=GRAPH_FONT, fill=colors['fg'], tags="background")
    canvas.create_text(graph_left - 30, (graph_top + graph_bottom) / 2,
                       text="Accuracy", font=GRAPH_FONT, angle=90,
                       fill=colors['fg'], tags="background")


def update_accuracy_graph(canvas, acc_data, val_acc_data=None):
    """Accuracy 데이터(점+선) 그리기"""
    canvas.update_idletasks()
    width = canvas.winfo_width() or 1
    height = canvas.winfo_height() or 1

    colors = get_theme_colors(canvas.cget('bg') == '#2b2b2b')

    left, right, top, bottom = 50, 20, 15, 35
    graph_left = left
    graph_right = width - right
    graph_top = top
    graph_bottom = height - bottom

    graph_width = graph_right - graph_left
    graph_height = graph_bottom - graph_top

    canvas.delete("acc_line", "val_acc_line",
                  "acc_point", "val_acc_point",
                  "y_labels", "x_labels", "legend")

    if not acc_data:
        return

    num_y = 6
    for i in range(num_y):
        y_val = i / (num_y - 1)
        y_pos = graph_bottom - (i / (num_y - 1)) * graph_height
        canvas.create_text(graph_left - 5, y_pos,
                           text=f"{y_val:.1f}",
                           anchor="e",
                           tags="y_labels",
                           font=GRAPH_FONT,
                           fill=colors['fg'])

    epochs = len(acc_data)
    num_ticks = min(6, epochs)
    for t in range(1, num_ticks + 1):
        idx = int(round(t * epochs / num_ticks)) - 1
        idx = max(0, min(idx, epochs - 1))
        x_pos = graph_left + (idx / (epochs - 1 if epochs > 1 else 1)) * graph_width
        canvas.create_text(x_pos, graph_bottom + 15,
                           text=str(idx + 1),
                           tags="x_labels",
                           font=GRAPH_FONT,
                           fill=colors['fg'])

    pts_acc = []
    for i, v in enumerate(acc_data):
        x = graph_left + (i / (epochs - 1 if epochs > 1 else 1)) * graph_width
        y = graph_bottom - (v * graph_height)
        x = max(graph_left, min(graph_right, x))
        y = max(graph_top, min(graph_bottom, y))
        pts_acc += [x, y]
        canvas.create_oval(x - 2, y - 2, x + 2, y + 2,
                           fill=colors['acc'], tags="acc_point")

    if len(pts_acc) >= 4:
        canvas.create_line(pts_acc, fill=colors['acc'], width=2, smooth=0, tags="acc_line")

    if val_acc_data:
        pts_val = []
        for i, v in enumerate(val_acc_data):
            x = graph_left + (i / (epochs - 1 if epochs > 1 else 1)) * graph_width
            y = graph_bottom - (v * graph_height)
            x = max(graph_left, min(graph_right, x))
            y = max(graph_top, min(graph_bottom, y))
            pts_val += [x, y]
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2,
                               fill=colors['val_acc'], tags="val_acc_point")

        if len(pts_val) >= 4:
            canvas.create_line(pts_val, fill=colors['val_acc'],
                               width=2, dash=(4, 2), smooth=0,
                               tags="val_acc_line")

    legend_items = [
        ("Train Acc", colors['acc'], ()),
        ("Val   Acc", colors['val_acc'], (4, 2))
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

    canvas.create_rectangle(bx, by, bx + bw, by + bh,
                            fill=colors['canvas_bg'],
                            outline=colors['axis'],
                            tags="legend")
    for i, (lbl, col, dash) in enumerate(legend_items):
        yy = by + pad + i*ih + ih//2
        canvas.create_line(bx + pad, yy, bx + pad + ln_len, yy,
                           fill=col, width=2, dash=dash, tags="legend")
        canvas.create_text(bx + pad + ln_len + txt_pad, yy,
                           text=lbl, anchor="w",
                           font=GRAPH_FONT, fill=colors['fg'],
                           tags="legend")


def add_resize_handlers(loss_canvas, acc_canvas, train_state):
    """캔버스 크기 변경 시 배경부터 다시 그리고, 데이터(점+선)도 다시 그려 주기"""
    def on_loss_resize(event):
        loss_canvas.after(50, lambda:
            setup_loss_graph(loss_canvas, train_state["colors"])
        )
        if train_state.get("current_loss_data"):
            loss_canvas.after(100, lambda:
                update_loss_graph(
                    loss_canvas,
                    train_state["current_loss_data"],
                    train_state.get("current_val_loss_data")
                )
            )

    def on_acc_resize(event):
        acc_canvas.after(50, lambda:
            setup_accuracy_graph(acc_canvas, train_state["colors"])
        )
        if train_state.get("current_acc_data"):
            acc_canvas.after(100, lambda:
                update_accuracy_graph(
                    acc_canvas,
                    train_state["current_acc_data"],
                    train_state.get("current_val_acc_data")
                )
            )

    loss_canvas.bind('<Configure>', on_loss_resize)
    acc_canvas.bind('<Configure>', on_acc_resize)


def integrate_with_main(tab_main, notebook, tab_train):
    """main 탭과 train 탭 연결"""
    train_handlers = setup_train_tab(tab_train)

    def set_log_dir_callback(new_log_dir, data):
        # 메인 탭 위젯 비활성화
        for widget in tab_main.winfo_children():
            if hasattr(widget, 'configure') and hasattr(widget, 'cget'):
                if 'state' in widget.configure():
                    widget.configure(state="disabled")

        # Train 탭 업데이트 및 모니터링 시작
        train_handlers["update_status"](
            f"{data.get('Username', '사용자')}님! {data.get('Model Type', '모델')} 모델을 이용한 학습을 하고 있습니다."
        )
        train_handlers["update_param_text"](data)
        train_handlers["start_monitoring"](new_log_dir, data)

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