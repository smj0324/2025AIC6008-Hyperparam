import os
import glob
import time
import json
import base64
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
#from tuneparam.rag.search_faiss import faiss_search

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 저장하기 위한 설정
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────────────────
EPOCH_LOG_PATTERN = "epoch_log_*.jsonl"
INIT_INFO_PATTERN = "init_info_*.json"
TRAINING_GRAPH_DIRNAME = "training_graphs"
PNG_PATTERNS = [
    "loss_graph_epoch_*.png",
    "accuracy_graph_epoch_*.png",
    "*graph*.png",
    "*loss*.png",
    "*acc*.png"
]
DEFAULT_FIGSIZE = (6, 4)

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")

def faiss_search(query: str, top_k: int = 5):
    # 실제 RAG 검색을 구현할 때 대체
    return f"No specific evidence found for {query}. Using general best practices."

def extract_timestamp_from_filename(filepath: str) -> int:

    fname = os.path.basename(filepath)
    name_part = fname.rsplit(".", 1)[0]  
    parts = name_part.split("_")
    for part in reversed(parts):
        clean = part.strip()
        if clean.isdigit() and len(clean) >= 8:
            return int(clean)
    try:
        return int(os.path.getmtime(filepath))
    except Exception:
        return 0


def find_latest_epoch_log(log_dir: str) -> Optional[str]:

    if not log_dir or not os.path.isdir(log_dir):
        return None

    pattern = os.path.join(log_dir, EPOCH_LOG_PATTERN)
    epoch_logs = glob.glob(pattern)
    if not epoch_logs:
        return None

    epoch_logs.sort(key=extract_timestamp_from_filename, reverse=True)
    return epoch_logs[0]


def load_epoch_data(epoch_log_path: str):

    loss_list, val_loss_list = [], []
    acc_list, val_acc_list = [], []
    epoch_indices = []

    with open(epoch_log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            epoch = entry.get("epoch")
            if epoch is None:
                continue

            loss = entry.get("loss", entry.get("train_loss"))
            val_loss = entry.get("val_loss")
            acc = entry.get("accuracy", entry.get("acc"))
            val_acc = entry.get("val_accuracy", entry.get("val_acc"))

            epoch_indices.append(epoch + 1)  # 1-based index
            loss_list.append(loss if loss is not None else float("nan"))
            val_loss_list.append(val_loss if val_loss is not None else float("nan"))
            acc_list.append(acc if acc is not None else float("nan"))
            val_acc_list.append(val_acc if val_acc is not None else float("nan"))

    return epoch_indices, loss_list, val_loss_list, acc_list, val_acc_list


def plot_and_save_training_graphs(log_dir: str, save_dir: Optional[str] = None):

    latest_log = find_latest_epoch_log(log_dir)
    if not latest_log:
        return None, None

    epochs, loss_list, val_loss_list, acc_list, val_acc_list = load_epoch_data(latest_log)
    if not epochs:
        return None, None

    # 저장 경로 결정
    if save_dir is None:
        save_dir = os.path.join(log_dir, TRAINING_GRAPH_DIRNAME)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = int(time.time())
    loss_png = os.path.join(save_dir, f"loss_graph_epoch_{timestamp}.png")
    acc_png = os.path.join(save_dir, f"accuracy_graph_epoch_{timestamp}.png")

    # ─── 손실(Loss) 그래프 그리기 ─────────────────────────────────────────────────────
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(epochs, loss_list, label="Train Loss", marker="o")
    if any(not (val is None or (isinstance(val, float) and (val != val))) for val in val_loss_list):
        plt.plot(epochs, val_loss_list, label="Val Loss", marker="o", linestyle="--")
    plt.title("Training / Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_png)
    plt.close()

    # ─── 정확도(Accuracy) 그래프 그리기 ─────────────────────────────────────────────
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(epochs, acc_list, label="Train Acc", marker="o")
    if any(not (val is None or (isinstance(val, float) and (val != val))) for val in val_acc_list):
        plt.plot(epochs, val_acc_list, label="Val Acc", marker="o", linestyle="--")
    plt.title("Training / Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_png)
    plt.close()

    return loss_png, acc_png


def find_latest_graph_files(
    log_dir: str, max_retries: int = 5, retry_delay: float = 1.0
):
    """
    로그 디렉토리에서 가장 최근 PNG 그래프 파일들을 찾음 (5번 재시도 포함).
    Result: [('Loss Graph', path), ('Accuracy Graph', path)]
    """
    graph_folder = os.path.join(log_dir, TRAINING_GRAPH_DIRNAME)

    for attempt in range(max_retries):
        search_dirs = []
        if os.path.isdir(graph_folder):
            search_dirs.append(graph_folder)
        search_dirs.append(log_dir)

        found_files = []
        for base in search_dirs:
            for pat in PNG_PATTERNS:
                found_files.extend(glob.glob(os.path.join(base, pat)))
            if found_files:
                break

        found_files = list(set(found_files))
        if not found_files:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return []

        found_files.sort(key=extract_timestamp_from_filename, reverse=True)

        loss_graph, acc_graph = None, None
        for fpath in found_files:
            fn = os.path.basename(fpath).lower()
            if "loss" in fn and loss_graph is None:
                loss_graph = fpath
            elif ("accuracy" in fn or "acc" in fn) and acc_graph is None:
                acc_graph = fpath
            if loss_graph and acc_graph:
                break

        result_files = []
        if loss_graph:
            result_files.append(("Loss Graph", loss_graph))
        if acc_graph:
            result_files.append(("Accuracy Graph", acc_graph))

        if not result_files and len(found_files) >= 2:
            for idx, fpath in enumerate(found_files[:2]):
                label = "Loss Graph" if idx == 0 else "Accuracy Graph"
                result_files.append((label, fpath))

        return result_files

    return []


def get_latest_training_results(log_dir: str) -> Dict[str, float]:
    """
    로그 디렉토리에서 최신 epoch_log_*.jsonl 파일을 찾아
    마지막 줄(마지막 epoch)의 metrics을 딕셔너리로 반환.
    """
    if not log_dir or not os.path.isdir(log_dir):
        return {}

    # init_info에서 timestamp 추출
    init_pattern = os.path.join(log_dir, INIT_INFO_PATTERN)
    init_files = glob.glob(init_pattern)
    if init_files:
        latest_init = sorted(init_files, key=extract_timestamp_from_filename, reverse=True)[0]
        ts = os.path.basename(latest_init).replace("init_info_", "").replace(".json", "")
        epoch_pattern = os.path.join(log_dir, f"epoch_log_{ts}.jsonl")
        epoch_log_files = glob.glob(epoch_pattern)
        if not epoch_log_files:
            epoch_log_files = glob.glob(os.path.join(log_dir, EPOCH_LOG_PATTERN))
    else:
        epoch_log_files = glob.glob(os.path.join(log_dir, EPOCH_LOG_PATTERN))

    if not epoch_log_files:
        return {}

    latest_log = sorted(epoch_log_files, key=extract_timestamp_from_filename, reverse=True)[0]
    with open(latest_log, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            return {}
        last_entry = json.loads(lines[-1].strip())
        return {
            "accuracy": last_entry.get("accuracy", last_entry.get("acc", 0)),
            "val_accuracy": last_entry.get("val_accuracy", last_entry.get("val_acc", 0)),
            "loss": last_entry.get("loss", 0),
            "val_loss": last_entry.get("val_loss", 0),
        }


def get_current_params(log_dir: str) -> Dict[str, Any]:
    """
    로그 디렉토리에서 가장 최신 init_info_*.json을 로드하여 params_info 반환.
    """
    if not log_dir or not os.path.isdir(log_dir):
        return {}

    pattern = os.path.join(log_dir, INIT_INFO_PATTERN)
    info_files = glob.glob(pattern)
    if not info_files:
        return {}

    latest_file = sorted(info_files, key=extract_timestamp_from_filename, reverse=True)[0]
    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("params_info", {}) or {}


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as img_f:
        return base64.b64encode(img_f.read()).decode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────────────────
class HyperparameterOptimizer:

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def recommend_params(
        self,
        current_params: Dict[str, Any],
        training_results: Dict[str, float],
        model_name: str,
        dataset_type: str,
        goal: str = "Accuracy",
        rag_evidence: str = "",
        graph_analysis: str = ""
    ) -> Optional[Dict[str, Any]]:

        if not self.client:
            return None

        rag_evidence = faiss_search(model_name)
        print("rag_evidence:", rag_evidence)
        model_prompt = self._get_model_specific_prompt(
            model_name, current_params, dataset_type, goal, rag_evidence
        )
        model_params_schema = self._get_model_specific_params(model_name)

        response = self._query_llm(
            current_params=current_params,
            training_results=training_results,
            model_prompt=model_prompt,
            dataset_type=dataset_type,
            goal=goal,
            model_params_schema=model_params_schema,
            graph_analysis=graph_analysis
        )

        return response

    def _get_model_specific_params(self, model_name: str) -> Optional[Dict[str, Any]]:

        schemas = {
            "ResNet": {
                "recommendations": {
                    "optimizer": "",
                    "learning_rate": "",
                    "batch_size": "",
                    "epochs": "",
                    "weight_decay": "",
                    "momentum": "",
                    "dropout_rate": "",
                    "label_smoothing": "",
                    "scheduler": "",
                    "data_augmentation": "",
                    "batch_normalization": "",
                    "initialization": ""
                },
                "reasons": {
                    "optimizer": "",
                    "learning_rate": "",
                    "batch_size": "",
                    "epochs": "",
                    "weight_decay": "",
                    "momentum": "",
                    "dropout_rate": "",
                    "label_smoothing": "",
                    "scheduler": "",
                    "data_augmentation": "",
                    "batch_normalization": "",
                    "initialization": ""
                },
                "expected_improvement": ""
            },
            "MobilenetV3": {
                "recommendations": {
                    "optimizer": "",
                    "learning_rate": "",
                    "batch_size": "",
                    "epochs": "",
                    "verbose": "",
                    "callbacks": "",
                    "validation_split": "",
                    "validation_data": "",
                    "shuffle": "",
                    "class_weight": "",
                    "sample_weight": "",
                    "initial_epoch": "",
                    "steps_per_epoch": "",
                    "validation_steps": "",
                    "validation_batch_size": "",
                    "validation_freq": "",
                    "max_queue_size": "",
                    "workers": "",
                    "use_multiprocessing": "",
                    "alpha": "",
                    "minimalistic": "",
                    "include_top": "",
                    "weights": "",
                    "input_tensor": "",
                    "pooling": "",
                    "classifier_activation": "",
                    "include_preprocessing": "",
                },
                "reasons": {
                    "optimizer": "",
                    "learning_rate": "",
                    "batch_size": "",
                    "epochs": "",
                    "verbose": "",
                    "callbacks": "",
                    "validation_split": "",
                    "validation_data": "",
                    "shuffle": "",
                    "class_weight": "",
                    "sample_weight": "",
                    "initial_epoch": "",
                    "steps_per_epoch": "",
                    "validation_steps": "",
                    "validation_batch_size": "",
                    "validation_freq": "",
                    "max_queue_size": "",
                    "workers": "",
                    "use_multiprocessing": "",
                    "alpha": "",
                    "minimalistic": "",
                    "include_top": "",
                    "weights": "",
                    "input_tensor": "",
                    "pooling": "",
                    "classifier_activation": "",
                    "include_preprocessing": "",
                },
                "expected_improvement": ""
            },
            "LSTM": {
                "recommendations": {
                    "optimizer": "",
                    "learning_rate": "",
                    "batch_size": "",
                    "epochs": "",
                    "dropout": "",
                    "hidden_size": "",
                    "num_layers": "",
                    "sequence_length": "",
                    "bidirectional": "",
                    "gradient_clipping": "",
                    "embedding_dim": "",
                    "tagging_scheme": "",
                    "output_classifier": "",
                    "early_stopping_patience": "",
                    "recurrent_dropout": "",
                    "return_sequences": "",
                    "activation": "",
                    "recurrent_activation": "",
                    "use_bias": "",
                    "kernel_initializer": "",
                    "recurrent_initializer": "",
                    "bias_initializer": "",
                    "unit_forget_bias": "",
                    "kernel_regularizer": "",
                    "recurrent_regularizer": "",
                    "bias_regularizer": "",
                    "activity_regularizer": "",
                    "kernel_constraint": "",
                    "recurrent_constraint": "",
                    "bias_constraint": "",
                    "implementation": "",
                    "go_backwards": "",
                    "stateful": "",
                    "time_major": "",
                    "unroll": "",
                    "dense_dropout": "",
                    "verbose": "",
                    "callbacks": "",
                    "validation_split": "",
                    "validation_data": "",
                    "shuffle": "",
                    "class_weight": "",
                    "sample_weight": "",
                    "initial_epoch": "",
                    "steps_per_epoch": "",
                    "validation_steps": "",
                    "validation_batch_size": "",
                    "validation_freq": "",
                    "max_queue_size": "",
                    "workers": "",
                    "use_multiprocessing": ""
                },
                "reasons": {
                    "optimizer": "",
                    "learning_rate": "",
                    "batch_size": "",
                    "epochs": "",
                    "dropout": "",
                    "hidden_size": "",
                    "num_layers": "",
                    "sequence_length": "",
                    "bidirectional": "",
                    "gradient_clipping": "",
                    "embedding_dim": "",
                    "tagging_scheme": "",
                    "output_classifier": "",
                    "early_stopping_patience": "",
                    "recurrent_dropout": "",
                    "return_sequences": "",
                    "activation": "",
                    "recurrent_activation": "",
                    "use_bias": "",
                    "kernel_initializer": "",
                    "recurrent_initializer": "",
                    "bias_initializer": "",
                    "unit_forget_bias": "",
                    "kernel_regularizer": "",
                    "recurrent_regularizer": "",
                    "bias_regularizer": "",
                    "activity_regularizer": "",
                    "kernel_constraint": "",
                    "recurrent_constraint": "",
                    "bias_constraint": "",
                    "implementation": "",
                    "go_backwards": "",
                    "stateful": "",
                    "time_major": "",
                    "unroll": "",
                    "dense_dropout": "",
                    "verbose": "",
                    "callbacks": "",
                    "validation_split": "",
                    "validation_data": "",
                    "shuffle": "",
                    "class_weight": "",
                    "sample_weight": "",
                    "initial_epoch": "",
                    "steps_per_epoch": "",
                    "validation_steps": "",
                    "validation_batch_size": "",
                    "validation_freq": "",
                    "max_queue_size": "",
                    "workers": "",
                    "use_multiprocessing": ""
                },
                "expected_improvement": ""
            }
        }
        return schemas.get(model_name)

    def _get_model_specific_prompt(
        self,
        model_name: str,
        current_params: Dict[str, Any],
        dataset_type: str,
        goal: str,
        rag_evidence: Any = ""
    ) -> str:

        base_texts = {
            "MobilenetV3": f"""
            MobilenetV4 model optimization for {dataset_type} dataset.
            Key considerations:
            1. Balance between batch size and memory usage for mobile environments
            2. Learning rate's impact on convergence and accuracy
            3. Optimization goal: {goal}
            """,
            "LSTM": f"""
            LSTM model optimization for {dataset_type} dataset.
            Key considerations:
            1. Sequence length and batch size impact on memory
            2. Learning rate selection to prevent gradient issues
            3. Optimizer selection (especially Adam vs RMSprop)
            4. Optimization goal: {goal}
            """,
            "ResNet": f"""
            ResNet model optimization for {dataset_type} dataset.
            Key considerations:
            1. Batch normalization and learning rate scheduling
            2. Batch size impact on GPU memory and generalization
            3. Epoch count and optimizer selection for deep networks
            4. Optimization goal: {goal}
            """
        }
        base_prompt = base_texts.get(
            model_name,
            f"Optimize {model_name} model for {dataset_type} dataset with goal: {goal}"
        )

        if rag_evidence:
            base_prompt += f"\n\n[Reference Results from Similar Experiments or Papers]\n{rag_evidence.strip()}\n"

        return base_prompt

    def _query_llm(
        self,
        current_params: Dict[str, Any],
        training_results: Dict[str, float],
        model_prompt: str,
        dataset_type: str,
        goal: str,
        model_params_schema: Dict[str, Any],
        graph_analysis: str = ""
    ) -> Optional[Dict[str, Any]]:

        if not self.client:
            return None

        system_prompt = (
            "You are a machine learning hyperparameter optimization expert.\n"
            "Analyze the given model, dataset, current parameters, and training results to recommend optimal hyperparameters.\n"
            "Provide response in JSON format only, without additional explanation."
        )

        user_prompt = (
            f"Current hyperparameters:\n{json.dumps(current_params, indent=2)}\n\n"
            f"Training results:\n{json.dumps(training_results, indent=2)}\n\n"
            f"Dataset type: {dataset_type}\n"
            f"Optimization goal: {goal}\n\n"
            f"{model_prompt}\n\n"
            f"Respond in this JSON format only:\n{json.dumps(model_params_schema, ensure_ascii=False, indent=2)}"
        )

        print(f"\n\n\n\n\***************\nUser prompt: {user_prompt}")

        if graph_analysis:
            user_prompt += (
                "\n\nIMPORTANT: The following graph analysis provides critical insights about the training process.\n"
                "Use this information to make targeted recommendations:\n\n"
                f"{graph_analysis}\n\n"
                "Please ensure your recommendations directly address the issues identified in the graph analysis."
            )

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            content = response.choices[0].message.content.strip()
            if "json" in content:
                body = content.split("json")[1].split("```")[0].strip()[0].strip()
            else:
                body = content
                print(f"\n\n\n\n\***************\ncontent: {content}")
            return json.loads(body)
        except Exception as e:
            print(f"LLM query error: {e}")
            return None

    def _analyze_without_graphs(self, current_params, training_results, model_name):
        return ""

    def analyze_training_graphs(
        self,
        log_dir: str,
        current_params: Dict[str, Any],
        training_results: Dict[str, float],
        model_name: str
    ) -> str:

        if not self.client:
            return self._analyze_without_graphs(current_params, training_results, model_name)

        plot_and_save_training_graphs(
            log_dir,
            save_dir=os.path.join(log_dir, TRAINING_GRAPH_DIRNAME)
        )

        graph_files = find_latest_graph_files(log_dir, max_retries=5, retry_delay=1.0)
        if not graph_files:
            return self._analyze_without_graphs(current_params, training_results, model_name)

        image_contents = []
        for graph_name, graph_path in graph_files:
            try:
                img_b64 = encode_image(graph_path)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })
            except Exception:
                continue

        if not image_contents:
            return self._analyze_without_graphs(current_params, training_results, model_name)

        system_prompt = (
            "You are a deep learning expert. Analyze the training graphs and provide detailed diagnostic insights and actionable recommendations."
        )
        user_messages = [
            {
                "type": "text",
                "text": (
                    f"Here are the training graphs for the {model_name} model.\n"
                    f"Current hyperparameters:\n{json.dumps(current_params, indent=2, ensure_ascii=False)}\n\n"
                    f"Training results:\n{json.dumps(training_results, indent=2, ensure_ascii=False)}\n\n"
                    "Please analyze the graphs to diagnose the state of your training and suggest improvements:\n"
                    "- Training/validation loss interval (overfitting 여부)\n"
                    "- Stability of the learning curve\n"
                    "- Speed and pattern of convergence\n"
                    "- Need for early stopping\n"
                    "- Direction of hyperparameter tuning\n\n"
                    "Provide specific and actionable suggestions."
                )
            }
        ]
        user_messages.extend(image_contents)

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_messages}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception:
            return self._analyze_without_graphs(current_params, training_results, model_name)

    def analyze_and_recommend_from_log(
        self,
        log_dir: str,
        model_name: str = "ResNet",
        dataset_type: str = "Image",
        goal: str = "Accuracy"
    ) -> Optional[Dict[str, Any]]:

        if not log_dir or not os.path.isdir(log_dir):
            return None

        # 1) 최신 학습 결과
        training_results = get_latest_training_results(log_dir)
        if not training_results:
            training_results = {"accuracy": 0, "val_accuracy": 0, "loss": 0, "val_loss": 0}

        # 2) 현재 하이퍼파라미터
        current_params = get_current_params(log_dir)
        if not current_params:
            current_params = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100, "optimizer": "Adam"}

        # 3) 그래프 분석
        graph_analysis = self.analyze_training_graphs(
            log_dir, current_params, training_results, model_name
        )

        # 4) 하이퍼파라미터 추천
        recommendations = self.recommend_params(
            current_params=current_params,
            training_results=training_results,
            model_name=model_name,
            dataset_type=dataset_type,
            goal=goal,
            graph_analysis=graph_analysis
        )

        # 5) 결과 JSON 저장
        result = {
            "graph_analysis": graph_analysis,
            "recommendations": recommendations,
            "current_params": current_params,
            "training_results": training_results,
            "model_info": {
                "model_name": model_name,
                "dataset_type": dataset_type,
                "goal": goal
            }
        }
        timestamp = int(time.time())
        analysis_file = os.path.join(log_dir, f"hyperparameter_analysis_{timestamp}.json")
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result


# ─────────────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    optimizer = HyperparameterOptimizer()

    log_directory = os.environ.get("TP_LOG_DIR", "logs")

    if os.path.isdir(log_directory):
        subdirs = [d for d in os.listdir(log_directory) if os.path.isdir(os.path.join(log_directory, d))]
        if subdirs:
            def extract_ts_from_dir(dirname: str) -> str:
                parts = dirname.split("_")
                return "_".join(parts[-2:]) if len(parts) >= 3 else dirname

            sorted_subdirs = sorted(subdirs, key=extract_ts_from_dir, reverse=True)

            actual_log_dir = None
            for subdir in sorted_subdirs:
                test_dir = os.path.join(log_directory, subdir)
                graph_folder = os.path.join(test_dir, TRAINING_GRAPH_DIRNAME)
                if os.path.isdir(graph_folder):
                    png_files = glob.glob(os.path.join(graph_folder, "*graph*.png"))
                else:
                    png_files = glob.glob(os.path.join(test_dir, "*graph*.png"))
                if png_files:
                    actual_log_dir = test_dir
                    break

            if actual_log_dir is None:
                # PNG 그래프가 없으면 가장 최근 하위 디렉토리를 사용
                actual_log_dir = os.path.join(log_directory, sorted_subdirs[0])
        else:
            actual_log_dir = log_directory
    else:
        actual_log_dir = log_directory

    # 자동 분석 및 추천
    result = optimizer.analyze_and_recommend_from_log(
        log_dir=actual_log_dir,
        model_name="ResNet",      # "MobileNetV4", "LSTM" 등으로 변경 가능
        dataset_type="Image",     # "Text", "Tabular" 등
        goal="Accuracy"           # "Speed", "Memory" 등
    )

    if result:
        print("=== GRAPH ANALYSIS ===")
        print(result["graph_analysis"])
        print("=== RECOMMENDATIONS ===")
        print(json.dumps(result["recommendations"], indent=2, ensure_ascii=False))
    else:
        print("분석에 실패했습니다. 로그 디렉토리와 파일 구조를 확인해주세요.")
