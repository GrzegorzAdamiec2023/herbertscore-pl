from .warnings import HerbertScoreWarnings
from .idf import HerBERTScoreIdf
import torch
from transformers import logging

# It is fine that model head is not trained.
logging.set_verbosity_error()


def _build_similarity_matrix(context_embeddings: torch.Tensor, s_to_add_embeddings: torch.Tensor) -> torch.Tensor:
    context_shape = context_embeddings.shape
    s_to_add_shape = s_to_add_embeddings.shape

    context = torch.reshape(context_embeddings, (context_shape[0], 1, context_shape[1], context_shape[2]))
    s_to_add = torch.reshape(s_to_add_embeddings, (1, s_to_add_shape[0], s_to_add_shape[1], s_to_add_shape[2]))
    s_to_add = torch.transpose(s_to_add, -1, -2)

    matrix = context @ s_to_add
    assert matrix.shape == (context_shape[0], s_to_add_shape[0], context_shape[1], s_to_add_shape[1])
    return matrix


class HerBERTScoreCoreLogic(HerBERTScoreIdf, HerbertScoreWarnings):
    def __call__(self, candidate: list[str], reference: list[str]) -> dict:
        self._ensure_idf("sentence similarity scoring")
        tokenized_candidate = self._tokenize(candidate)
        tokenized_reference = self._tokenize(reference)

        embeddings_candidate = self._compute_embeddings(tokenized_candidate)
        embeddings_reference = self._compute_embeddings(tokenized_reference)

        matrix = _build_similarity_matrix(embeddings_candidate, embeddings_reference)

        acc = self._compute_weighted_accuracy(matrix, tokenized_reference)
        recall = self._compute_weighted_recall(matrix, tokenized_candidate)

        acc = (acc[0] - self.b_acc) / (1 - self.b_acc)
        recall = (recall[0] - self.b_recall) / (1 - self.b_recall)

        return {"accuracy": acc, "recall": recall, "f1": (2 * acc * recall / (acc + recall))}

    def _tokenize(self, sentences: list[str]) -> dict:
        tokenized = self.tokenizer(sentences, return_tensors="pt", padding=True, add_special_tokens=True)
        return {k: v.to(self.device) for k, v in tokenized.items()}

    def _compute_embeddings(self, tokenized: dict) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.model(**tokenized, output_hidden_states=True).last_hidden_state

        # L2-normalization
        lengths = torch.sqrt(torch.sum(embeddings**2, dim=-1))
        embeddings = embeddings / lengths.unsqueeze(-1)
        return embeddings

    def _compute_weighted_accuracy(self, matrix: torch.Tensor, tokenized_s_to_add: dict) -> torch.Tensor:
        acc = matrix.max(dim=2)[0]

        weights = tokenized_s_to_add["input_ids"].to("cpu").to(torch.float64).apply_(lambda x: self._get_idf(x)).to(self.device)
        weights = torch.unsqueeze(weights, 0)

        acc = acc * weights
        acc = acc.sum(dim=-1)
        acc = acc / torch.unsqueeze(weights.sum(dim=-1, keepdim=False), 0)
        return acc

    def _compute_weighted_recall(self, matrix: torch.Tensor, tokenized_context: dict) -> torch.Tensor:
        recall = matrix.max(dim=3)[0]

        weights = tokenized_context["input_ids"].to("cpu").to(torch.float64).apply_(lambda x: self._get_idf(x)).to(self.device)
        weights = torch.reshape(weights, (weights.shape[0], 1, weights.shape[1]))

        recall = recall * weights
        recall = recall.sum(dim=-1)
        recall = recall / torch.unsqueeze(weights.sum(dim=-1, keepdim=False), 0)
        return recall

