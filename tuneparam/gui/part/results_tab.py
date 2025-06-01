from tuneparam.framework.keras_ import TrainingLogger
import tkinter as tk
from tkinter import ttk
from operator import itemgetter
import json
import threading
from tkinter import scrolledtext
from tuneparam.model import HyperparameterOptimizer

GRAPH_FONT = ('Helvetica', 8)
latest_gpt_output = {}
scrolledtext_widgets = []

def get_theme_colors(is_dark_theme=False):
    if is_dark_theme:
        return {
            'bg': "#313131",
            'fg': "#ffffff",
            'grid': "#404040",
            'axis': "#ffffff",
            'loss': "#ff4444",
            'val_loss': "#ffaa44",
            'acc': "#4a9eff",
            'val_acc': "#66ccff",
            'canvas_bg': "#313131"
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

def split_summary_by_keys(summary, keys):
    getter = itemgetter(*keys)
    included = dict(zip(keys, getter(summary)))
    excluded = {k: v for k, v in summary.items() if k not in keys}
    return included, excluded

def apply_gpt_params(tab_train, logger: TrainingLogger):
    print("파라미터 추천 시작")
    included, excluded = split_summary_by_keys(logger.summary, logger.params_key)
    print(included)
    print(excluded)
    print(logger.user_data)

    optimizer = HyperparameterOptimizer()

    recommendations = optimizer.recommend_params(
        current_params=included,
        training_results=excluded,
        model_name=logger.user_data['Model Type'],
        dataset_type=logger.user_data['Dataset Type'],
        goal=logger.user_data['Goal']
    )
    return recommendations

# === 테마 일괄 적용 ===
def apply_theme_recursively(widget, theme):
    """
    모든 하위 위젯에 대해 bg/fg/스타일 일괄 적용
    """
    # TK widgets
    if isinstance(widget, (tk.Frame, tk.LabelFrame)):
        widget.config(bg=theme['canvas_bg'])
    elif isinstance(widget, tk.Label):
        widget.config(bg=theme['canvas_bg'], fg=theme['fg'])
    elif isinstance(widget, tk.Button):
        widget.config(bg=theme['canvas_bg'], fg=theme['fg'],
                      activebackground=theme['acc'], activeforeground=theme['fg'])
    elif isinstance(widget, scrolledtext.ScrolledText):
        widget.config(bg=theme['canvas_bg'], fg=theme['fg'], insertbackground=theme['fg'])
    # TTK widgets
    elif isinstance(widget, ttk.LabelFrame):
        style = ttk.Style()
        style.configure("Custom.TLabelframe", background=theme['canvas_bg'], foreground=theme['fg'])
        widget.configure(style="Custom.TLabelframe")
    elif isinstance(widget, ttk.Button):
        style = ttk.Style()
        style.configure("Custom.TButton", background=theme['canvas_bg'], foreground=theme['fg'])
        widget.configure(style="Custom.TButton")

    # 재귀적으로 모든 자식 위젯에도 적용
    for child in widget.winfo_children():
        apply_theme_recursively(child, theme)

def update_scrolledtext_widgets_theme(theme):
    for widget in scrolledtext_widgets:
        widget.config(bg=theme['canvas_bg'], fg=theme['fg'], insertbackground=theme['fg'])

def show_gpt_params(tab_train, logger: TrainingLogger, is_dark_theme=False, retrain_callback=None):
    global latest_gpt_output
    theme = get_theme_colors(is_dark_theme)
    gpt_output = apply_gpt_params(tab_train, logger)
    print("GPT OUTPUT:", gpt_output)

    latest_gpt_output = gpt_output

    # 기존 GPT 프레임 삭제
    for child in tab_train.winfo_children():
        if getattr(child, "is_gpt_frame", False):
            child.destroy()
    recommendations = gpt_output["recommendations"]
    reasons = gpt_output["reasons"]
    expected_improvement = gpt_output.get("expected_improvement", "")

    # --- LLM 추천 하이퍼파라미터 섹션 ---
    gpt_frame = ttk.LabelFrame(tab_train, text="📌 LLM 추천 하이퍼파라미터", style="Custom.TLabelframe")

    gpt_frame.is_gpt_frame = True
    gpt_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    inner = tk.Frame(gpt_frame, bg=theme['canvas_bg'])
    inner.pack(fill="both", expand=True)

    rec_text = scrolledtext.ScrolledText(
        inner, width=55, height=10, wrap="word",
        font=("Helvetica", 11),
        bg=theme['canvas_bg'], fg=theme['fg'],
        borderwidth=0, relief="flat", highlightthickness=0, insertbackground=theme['fg']
    )
    scrolledtext_widgets.append(rec_text)
    for key in recommendations:
        rec_text.insert("end", f"{key}: {recommendations[key]}\n", "bold")
        if reasons.get(key):
            rec_text.insert("end", f"  reason: {reasons[key]}\n\n", "reason")
        else:
            rec_text.insert("end", "\n")
    rec_text.tag_config("bold", font=("Helvetica", 11, "bold"))
    rec_text.tag_config("reason", font=("Helvetica", 9, "normal"), foreground="#888888" if not is_dark_theme else "#bbbbbb")
    rec_text.config(state="disabled")
    rec_text.pack(fill="both", expand=True, padx=10, pady=(5, 5))

    improvement_label = tk.Label(
        inner, text="📈 Expected Improvement:", font=("Helvetica", 10, "bold"),
        bg=theme['canvas_bg'], fg=theme['acc']
    )
    improvement_label.pack(anchor="w", padx=10, pady=(5, 0))

    improvement_box = scrolledtext.ScrolledText(
        inner, height=3, wrap="word", font=("Helvetica", 10, "italic"),
        bg=theme['canvas_bg'], fg=theme['acc'],
        borderwidth=0, relief="flat", highlightthickness=0, insertbackground=theme['acc']
    )
    scrolledtext_widgets.append(improvement_box)
    improvement_box.insert("1.0", expected_improvement)
    improvement_box.config(state="disabled")
    improvement_box.pack(fill="x", expand=False, padx=10, pady=(0, 8))

    try:
        style = ttk.Style()
        style.configure("Custom.TLabelframe", background=theme['canvas_bg'])
        style.configure("Custom.TLabelframe.Label", background=theme['canvas_bg'], foreground=theme['fg'], font=("Helvetica", 12, "bold"))
    except Exception:
        pass

    if retrain_callback:
        retrain_callback()

def setup_results_tab(tab_train, train_parameters=None, preset_logger: TrainingLogger = None, user_data=None, is_dark_theme=False):
    theme = get_theme_colors(is_dark_theme)
    scrolledtext_widgets.clear()

    # === 버튼 row 프레임
    btn_row_frame = tk.Frame(tab_train, bg=theme['canvas_bg'])
    btn_row_frame.pack(side="top", fill="x", expand=False)

    retrain_button_below = tk.Button(
        btn_row_frame, text="재훈련",
        command=lambda: on_retrain_click(),
        bg=theme['canvas_bg'], fg=theme['fg'],
        activebackground=theme['acc'], activeforeground=theme['fg']
    )
    retrain_button_below.pack_forget()  # 처음에는 숨겨둠

    # --- (기존) 훈련 파라미터 프레임 ---
    if train_parameters:
        used_params_frame = ttk.LabelFrame(tab_train, text="사용된 훈련 파라미터", style="Custom.TLabelframe")
        used_params_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        inner = tk.Frame(used_params_frame, bg=theme['canvas_bg'])
        inner.pack(fill="both", expand=True)
        params_text = scrolledtext.ScrolledText(
            inner, width=40, height=10, wrap="word",
            font=("Helvetica", 10),
            bg=theme['canvas_bg'], fg=theme['fg'],
            borderwidth=0, relief="flat", highlightthickness=0, insertbackground=theme['fg']
        )
        scrolledtext_widgets.append(params_text)
        for key, value in train_parameters.items():
            params_text.insert("end", f"{key}: ", "bold")
            params_text.insert("end", f"{value}\n", "normal")
        params_text.tag_config("bold", font=("Helvetica", 10, "bold"))
        params_text.tag_config("normal", font=("Helvetica", 10))
        params_text.config(state="disabled")
        params_text.pack(fill="both", expand=True, padx=10, pady=(5, 5))

        loading_indicator = ttk.Progressbar(inner, mode="indeterminate")
        retrain_button = tk.Button(
            inner,
            text="하이퍼파라미터 추천받기",
            command=lambda: on_btn_click(),
            bg=theme['canvas_bg'], fg=theme['fg'],
            activebackground=theme['acc'], activeforeground=theme['fg']
        )
        retrain_button.pack(pady=(8, 10), anchor="center")

        def update_theme(is_dark):
            # 테마 업데이트 함수: 모든 위젯 일괄 반영
            nonlocal retrain_button_below, retrain_button, used_params_frame, inner, params_text
            theme = get_theme_colors(is_dark)
            style = ttk.Style()
            style.configure("UsedParams.TLabelframe", background=theme['canvas_bg'])
            style.configure("UsedParams.TLabelframe.Label", background=theme['canvas_bg'], foreground=theme['fg'])
            apply_theme_recursively(tab_train, theme)
            update_scrolledtext_widgets_theme(theme)

        def run_gpt_and_hide_loading():
            show_gpt_params(tab_train, preset_logger, is_dark_theme, retrain_callback=show_below_button)
            loading_indicator.stop()
            loading_indicator.pack_forget()

        def on_btn_click():
            retrain_button.config(state="disabled")
            loading_indicator.pack(pady=(0, 10), fill="x")
            loading_indicator.start(10)
            threading.Thread(target=run_gpt_and_hide_loading, daemon=True).start()

        def show_below_button():
            retrain_button_below.pack(pady=(20, 20))

        def on_retrain_click():
            from tuneparam.gui.main import start_retrain
            retrain_button_below.config(state="disabled")
            print("재훈련 버튼이 눌렸습니다.")
            if latest_gpt_output:
                start_retrain(latest_gpt_output)
            else:
                print("gpt_output이 없습니다. 먼저 하이퍼파라미터 추천을 받아주세요.")
            retrain_button_below.config(state="normal")

    return {"update_theme": update_theme}
