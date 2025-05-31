import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
from datetime import datetime
import platform
import copy

def pretty_print_dict(title, d):
    print(f"\n{'='*30}\n{title}\n{'='*30}")
    print(json.dumps(d, ensure_ascii=False, indent=2))

def convert_json_serializable(obj):
    if isinstance(obj, dict):
        return {convert_json_serializable_key(k): convert_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_json_serializable(i) for i in obj]
    elif isinstance(obj, np.generic):  # numpy number (scalar)
        return obj.item()
    elif isinstance(obj, (np.ndarray, )):
        return obj.tolist()
    else:
        return obj
    

def convert_json_serializable_key(key):
    if isinstance(key, (int, float, bool, type(None), str)):
        return key
    elif isinstance(key, np.generic):
        return key.item()
    else:
        return str(key)
    



class TrainingLogger(Callback):
    def __init__(self, log_dir="logs", params=None, summary_params = None, X=None, y=None):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.epoch_log_path = os.path.join(self.log_dir, f"epoch_log_{now_str}.jsonl")
        self.summary_path = os.path.join(self.log_dir, f"exp_summary_{now_str}.json")
        self.logs = []
        self.start_time = None
        self.params_info = params if params else {}
        self.summary = summary_params
        self.params_key = copy.deepcopy(list(summary_params.keys()))
        self.user_data = {}

        if y is not None:
            labels, counts = np.unique(y, return_counts=True)
            y_class_dist = {str(label): int(count) for label, count in zip(labels, counts)}
        else:
            y_class_dist = None

        self.data_info = {
            "X_shape": X.shape if X is not None else None,
            "y_shape": y.shape if y is not None else None,
            "X_dtype": str(X.dtype) if X is not None else None,
            "y_dtype": str(y.dtype) if y is not None else None,
            "y_class_dist": y_class_dist,
        }
        print("🔎 [DEBUG] TrainingLogger init:")
        pretty_print_dict("data_info", self.data_info)
        pretty_print_dict("params_info", self.params_info)

        self.init_info_path = self.save_init_info()

    def save_init_info(self):
        init_info = {
            "data_info": self.data_info,
            "params_info": self.params_info
        }
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.log_dir, f"init_info_{now_str}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(init_info, f, ensure_ascii=False, indent=2)
        print(f"✅ 학습 전 초기 정보 저장 완료: {path}")
        return path
    

    def get_preset_data_for_main_tab(self):
        X_shape = self.data_info.get("X_shape", [None])
        hardware = self.params_info.get("hardware", {})
        gpu_list = hardware.get("gpu", [])
        return {
            "Version": self.params_info.get("tf_version", ""),
            "Hardware": gpu_list[0].name if gpu_list else "CPU",
            "Dataset Size": str(X_shape[0]) if X_shape and X_shape[0] is not None else "",
            "Model Type": "Classification" if "accuracy" in self.params_info.get("metrics", []) else "",
            "Dataset Type": "Tabular",
            "Goal": "Accuracy",
            "Model Size": "Base"
            # Username은 사용자가 직접 입력
        }

    def set_model(self, model):
        self.params_info["model_config"] = model.to_json()
        self.params_info["model_summary"] = []
        model.summary(print_fn=lambda x: self.params_info["model_summary"].append(x))
        self.params_info["model_summary"] = "\n".join(self.params_info["model_summary"])
        model.save_weights(os.path.join(self.log_dir, "final_weights.h5"))
        self.params_info["weights_path"] = os.path.join(self.log_dir, "final_weights.h5")
        if hasattr(model, "optimizer"):
            self.params_info["optimizer"] = str(model.optimizer)
            try:
                self.params_info["optimizer_config"] = model.optimizer.get_config()
            except Exception:
                pass
        self.params_info["loss"] = str(model.loss)
        self.params_info["metrics"] = model.metrics_names

    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
        self.params_info["random_seed"] = random.getstate()[1][0]
        self.params_info["np_seed"] = np.random.get_state()[1][0]
        self.params_info["tf_seed"] = getattr(tf.random, "get_seed", lambda: (None, None))()
        self.params_info["tf_version"] = tf.__version__
        self.params_info["python_version"] = platform.python_version()
        self.params_info["platform"] = platform.platform()
        self.params_info["hardware"] = {
            "cpu": platform.processor(),
            "gpu": tf.config.list_physical_devices('GPU'),
        }
        self.params_info["start_time"] = self.start_time.isoformat()

    def on_epoch_end(self, epoch, logs=None):
        log_entry = {"epoch": epoch, **(logs or {})}
        self.logs.append(log_entry)
        # todo DB 안에 log 저장하기
        
        # 매 에포크별로 jsonl(한 줄에 하나) 파일에 append
        with open(self.epoch_log_path, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")

    def on_train_end(self, logs=None):
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds() if self.start_time else None
        best_epoch = max(self.logs, key=lambda x: x.get("val_accuracy", x.get("accuracy", 0)))["epoch"] \
            if self.logs else None
        early_stopping_epoch = None
        for cb in getattr(self.model, 'callbacks', []):
            if isinstance(cb, EarlyStopping):
                early_stopping_epoch = cb.stopped_epoch
        print(self.summary)
        self.summary.update({
            "best_epoch": best_epoch,
            "best_accuracy": max((l.get("val_accuracy", l.get("accuracy", 0)) for l in self.logs), default=None),
            "train_end_time": end_time.isoformat(),
        })
        print(self.summary)

        sampled_logs = []

        for i in range(0, len(self.logs), 10):
            log = self.logs[i]
            # 소수점 둘째 자리까지 반올림
            rounded_log = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in log.items()
            }
            sampled_logs.append(rounded_log)

        self.summary["logs_every_10"] = sampled_logs


        summary = {
            "hyperparameters": self.params_info,
            "data_info": self.data_info,
            "history": self.logs,  # 전체 epoch log도 포함(원하면 제외 가능)
            "best_epoch": best_epoch,
            "best_accuracy": max((l.get("val_accuracy", l.get("accuracy", 0)) for l in self.logs), default=None),
            "train_end_time": end_time.isoformat(),
            "total_train_seconds": total_time,
            "early_stopping_epoch": early_stopping_epoch,
            "epoch_log_file": self.epoch_log_path,
        }
        serializable_summary = convert_json_serializable(summary)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(serializable_summary, f, ensure_ascii=False, indent=2)
        print(f"✅ 전체 요약 로그 저장 완료: {self.summary_path}")
        print(f"✅ 에포크별 로그: {self.epoch_log_path}")