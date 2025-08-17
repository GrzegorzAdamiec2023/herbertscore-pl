import json
import os

import torch
from .base import HerBERTScoreBase

class HerBERTScoreIO(HerBERTScoreBase):
    def load_idf(self, file_path: str = "herBERTScoreState/idf.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            idf_loaded = json.load(f)
        model_in_file = idf_loaded.get("_model_name", None)
        if model_in_file is not None and model_in_file != self.model_name:
            raise RuntimeError(f"IDF was computed for model '{model_in_file}', "
                               f"but current model is '{self.model_name}'.")
        self.idf = {int(k): v for k, v in idf_loaded.items() if k != "_model_name"}
        print(f"IDF loaded from {file_path}")

    def save_idf(self, file_path: str = "herBERTScoreState/idf.json"):
        idf_to_save = {str(k): v for k, v in self.idf.items()}
        idf_to_save["_model_name"] = self.model_name
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(idf_to_save, f)
        print(f"IDF saved to {file_path}")

    def save_baseline(self, file_path: str = "herBERTScoreState/baseline.pt"):
        torch.save({
            "b_acc": self.b_acc,
            "b_recall": self.b_recall,
            "_model_name": self.model_name
        }, file_path)
        print(f"Baseline saved to {file_path}")

    def load_baseline(self, file_path: str = "herBERTScoreState/baseline.pt"):
        baseline = torch.load(file_path)
        model_in_file = baseline.get("_model_name", None)
        if model_in_file is not None and model_in_file != self.model_name:
            raise RuntimeError(f"Baseline was computed for model '{model_in_file}', "
                               f"but current model is '{self.model_name}'.")
        self.b_acc = baseline["b_acc"]
        self.b_recall = baseline["b_recall"]
        print(f"Baseline loaded from {file_path}")

    def save_state(self, folder_path: str = "herBERTScoreState"):
        os.makedirs(folder_path, exist_ok=True)
        idf_path = os.path.join(folder_path, "idf.json")
        baseline_path = os.path.join(folder_path, "baseline.pt")
        self.save_idf(idf_path)
        self.save_baseline(baseline_path)

    def load_state(self, folder_path: str = "herBERTScoreState"):
        idf_path = os.path.join(folder_path, "idf.json")
        baseline_path = os.path.join(folder_path, "baseline.pt")
        self.load_idf(idf_path)
        self.load_baseline(baseline_path)
