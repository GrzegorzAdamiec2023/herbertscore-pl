import logging

from tqdm import tqdm
import math
import random
import warnings
import torch
from .core_logic import HerBERTScoreCoreLogic
from .warnings import HerbertScoreWarnings
from .exceptions import HerBERTScoreExceptions


class HerBERTScoreBaseline(HerBERTScoreCoreLogic, HerBERTScoreExceptions):
    def _precess_dataset_to_sentences(self, skip_dialogues: bool):
        with open(self.file_path, "r", encoding="utf-8") as file:
            texts = file.read()
        contexts = texts.split("\n")
        if skip_dialogues:
            contexts_clean = [i for i in contexts if i.find("–") == -1]
        else:
            contexts_clean = contexts
        sentences = []
        for context in contexts_clean:
            for i in context.split("."):
                if len(i.strip()) > 0:
                    sentences.append(i.strip())
        return sentences

    def compute_b(self, batch_size: int = 100):
        """
        Compute the BERTScore baseline (b) for precision and recall.
        """
        self.data_base_missing()
        self._ensure_idf("baseline computing")
        sentences = self._prepare_sentences_for_baseline()
        n_batches = math.ceil(len(sentences) / batch_size)

        acc_sum, recall_sum = self._compute_baseline_sums(sentences, batch_size, n_batches)

        # Compute number of pairs
        n = len(sentences) ** 2 - len(sentences)
        self.b_acc = acc_sum / n
        self.b_recall = recall_sum / n
        print("New baseline has been set.")

    def _prepare_sentences_for_baseline(self) -> list[str]:
        sentences = self._precess_dataset_to_sentences(self.skip_dialogues)
        random.shuffle(sentences)

        context_length = math.ceil(1 + math.sqrt(1 + 4 * self.n_example_pairs) / 2)
        if len(sentences) < context_length:
            warnings.warn(
                f"Only {len(sentences) ** 2 - len(sentences)} unique pairs available, "
                f"but {self.n_example_pairs} requested."
            )
            context_length = len(sentences)

        return sentences[:context_length]

    def _compute_baseline_sums(self, sentences: list[str], batch_size: int, n_batches: int):
        acc_sum = 0.0
        recall_sum = 0.0
        total_iterations = n_batches * n_batches

        with tqdm(total=total_iterations, desc="Computing baseline") as pbar:
            for batch_i in range(0, len(sentences), batch_size):
                for batch_j in range(0, len(sentences), batch_size):
                    sentences1 = sentences[batch_i: batch_i + batch_size]
                    sentences2 = sentences[batch_j: batch_j + batch_size]

                    out = self.__call__(sentences1, sentences2)

                    metric_sum_acc = torch.sum(out["accuracy"])
                    metric_sum_recall = torch.sum(out["recall"])

                    # Subtract diagonal values if same batch (not random)
                    if batch_j == batch_i:
                        metric_sum_acc -= torch.sum(out["accuracy"].diagonal())
                        metric_sum_recall -= torch.sum(out["recall"].diagonal())

                    acc_sum += metric_sum_acc.item()
                    recall_sum += metric_sum_recall.item()
                    pbar.update(1)

        return acc_sum, recall_sum
