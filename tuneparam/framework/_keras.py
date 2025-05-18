import json
import os
from tensorflow.keras.callbacks import Callback

class TrainingLogger(Callback):
    def __init__(self, log_path="training_log.json"):
        super().__init__()
        self.log_path = log_path
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        log_entry = {"epoch": epoch, **(logs or {})}
        self.logs.append(log_entry)

        with open(self.log_path, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")

    def on_train_end(self, logs=None):
        print(f"✅ 로그 저장 완료: {self.log_path}")