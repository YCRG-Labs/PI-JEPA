import os
import json
import time
import gzip
from datetime import datetime


class Logger:
    def __init__(
        self,
        log_dir,
        experiment_name="default",
        compress=True,
        float_precision=4,
        buffer_size=50
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.log_path, exist_ok=True)

        self.log_file = os.path.join(self.log_path, "logs.txt")

        self.compress = compress
        self.float_precision = float_precision
        self.buffer_size = buffer_size

        if compress:
            self.json_file = os.path.join(self.log_path, "metrics.jsonl.gz")
        else:
            self.json_file = os.path.join(self.log_path, "metrics.jsonl")

        self.buffer = []

    def _write(self, message):
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        self._write(full_message)

    def log_metrics(self, metrics: dict, step: int):
        entry = {"step": step}

        for k, v in metrics.items():
            entry[k] = round(float(v), self.float_precision)

        self.buffer.append(entry)

        if len(self.buffer) >= self.buffer_size:
            self._flush()

        msg = f"Step {step} | " + " | ".join(
            [f"{k}: {v:.{self.float_precision}f}" for k, v in metrics.items()]
        )
        self.log(msg)

    def _flush(self):
        if not self.buffer:
            return

        if self.compress:
            with gzip.open(self.json_file, "at") as f:
                for entry in self.buffer:
                    f.write(json.dumps(entry) + "\n")
        else:
            with open(self.json_file, "a") as f:
                for entry in self.buffer:
                    f.write(json.dumps(entry) + "\n")

        self.buffer = []

    def save_config(self, config):
        path = os.path.join(self.log_path, "config.yaml")
        with open(path, "w") as f:
            import yaml
            yaml.dump(config.as_dict(), f)

    def close(self):
        self._flush()

    def get_log_dir(self):
        return self.log_path
