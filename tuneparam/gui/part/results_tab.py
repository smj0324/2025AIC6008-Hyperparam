# 그래프용 작은 폰트 정의
from tuneparam.framework.keras_ import TrainingLogger
from tkinter import ttk
from operator import itemgetter
import json

from tuneparam.model import HyperparameterOptimizer

GRAPH_FONT = ('Helvetica', 8)


def generate_hparam_prompt(current_params, training_results, dataset_type, goal, model_prompt):
    """
    하이퍼파라미터 튜닝을 위한 프롬프트 생성 함수.

    Args:
        current_params (dict): 현재 하이퍼파라미터 설정.
        training_results (dict or list): 학습 결과 로그.
        dataset_type (str): 예: 'image_classification'.
        goal (str): 예: 'maximize validation accuracy'.
        model_prompt (str): 모델 설명 또는 구성 요약.

    Returns:
        str: GPT에게 전달할 프롬프트 문자열.
    """
    user_prompt = f"""
    Current hyperparameters:
    {json.dumps(current_params, indent=2)}

    Training results:
    {json.dumps(training_results, indent=2)}

    Dataset type: {dataset_type}
    Optimization goal: {goal}

    {model_prompt}

    Respond in this JSON format only:
    {{
      "recommendations": {{
        "learning_rate": value,
        "batch_size": value,
        "epochs": value,
        "optimizer": "value"
      }},
      "reasons": {{
        "learning_rate": "reason",
        "batch_size": "reason",
        "epochs": "reason",
        "optimizer": "reason"
      }},
      "expected_improvement": "description"
    }}
    """
    return user_prompt.strip()


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
    optimizer = HyperparameterOptimizer()

    recommendations = optimizer.recommend_params(
        current_params=included,
        training_results=excluded,
        model_name="MobilenetV4",
        dataset_type="Image",
        goal="Accuracy"
    )
    print("Recommended hyperparameters:")
    print(json.dumps(recommendations, indent=2, ensure_ascii=False))

    # try:
    #     tab_train.lr_entry.delete(0, tk.END)
    #     tab_train.lr_entry.insert(0, "0.001")
    #
    #     tab_train.batch_entry.delete(0, tk.END)
    #     tab_train.batch_entry.insert(0, str(recommended_params['batch_size']))
    #
    #     tab_train.epochs_entry.delete(0, tk.END)
    #     tab_train.epochs_entry.insert(0, str(recommended_params['epochs']))
    # except AttributeError:
    #     print("Entry 위젯이 정의되어 있지 않습니다.")

def on_retrain_with_params(train_parameters):
    print("이 설정으로 재훈련을 시작합니다:")
    print("훈련 시작하기")
    for key, val in train_parameters.items():
        print(f"  {key}: {val}")


def show_gpt_params(tab_train, logger: TrainingLogger):
    # 요기서
    original_params = logger.summary

    print(original_params)

    # 기존에 있던 gpt_frame 삭제 (중복 방지)
    for child in tab_train.winfo_children():
        if getattr(child, "is_gpt_frame", False):
            child.destroy()

    gpt_frame = ttk.LabelFrame(tab_train, text="📌 LLM 추천 하이퍼파라미터")
    gpt_frame.is_gpt_frame = True
    gpt_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    for key, value in original_params.items():
        row = ttk.Frame(gpt_frame)
        row.pack(anchor="w", padx=10, pady=2)

        key_label = ttk.Label(row, text=f"{key}:", width=18)
        key_label.pack(side="left")

        value_label = ttk.Label(row, text=str(value))
        value_label.pack(side="left")

    apply_button = ttk.Button(
        gpt_frame,
        text="추천값 적용하기",
        command=lambda: apply_gpt_params(tab_train, logger)
    )
    apply_button.pack(pady=15)


def setup_results_tab(tab_train,  train_parameters=None, preset_logger : TrainingLogger = None):

    if train_parameters:
        used_params_frame = ttk.LabelFrame(tab_train, text="사용된 훈련 파라미터")
        used_params_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        for key, value in train_parameters.items():
            row = ttk.Frame(used_params_frame)
            row.pack(anchor="w", padx=10, pady=2)

            key_label = ttk.Label(row, text=f"{key}:", width=18)
            key_label.pack(side="left")

            value_label = ttk.Label(row, text=str(value))
            value_label.pack(side="left")

        retrain_button = ttk.Button(
            used_params_frame,
            text="하이퍼파라미터 추천받기",
            command=lambda: show_gpt_params(tab_train, preset_logger)
        )
        retrain_button.pack(pady=15)
