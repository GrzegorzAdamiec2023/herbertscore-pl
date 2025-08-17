from transformers import AutoTokenizer, AutoModel
import torch

class HerBERTScoreBase:
    """
    HerBERTScore - custom implementation of BERTScore using HerBERT embeddings.

    Features:
    - Computes precision, recall, and F1 metrics between sentences using transformer embeddings.
    - Computes a baseline 'b' from a subset of corpus sentences to rescale BERTScore (0-1 range).
    - Supports IDF weighting of tokens for more accurate similarity measures.

    Implementation Notes / Compromises:
    - The baseline 'b' is computed from all pairs in a limited shuffled subset of sentences
      rather than fully random pairs from a huge corpus (as in the original BERTScore paper).
      This makes computation fast and deterministic but slightly dataset-dependent.
    - Rescaling using this baseline may differ from the original paper’s universal baseline,
      but it is sufficient for practical ranking and evaluation of text generation systems.
    - Batching is used to accelerate computation of pairwise scores.
    """
    n_example_pairs = 1000000 # The value is based on the original article

    def __init__(self, model_name: str = "allegro/herbert-base-cased", file_path_to_texts: str = "data/NKJP_test_texts.txt",
                 skip_dialogues: bool = True):
        """

        :param model_name:
        :param file_path_to_texts: Path to a plain text file containing the corpus used for computing IDF and baseline metrics.
                                   Each sentence or paragraph should be separated by a newline. The text should be UTF-8 encoded.
                                   No additional formatting (like JSON or CSV) is required.
        :param skip_dialogues:
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.file_path = file_path_to_texts

        self.idf = None

        self.b_acc = 0
        self.b_recall = 0
        self.skip_dialogues = skip_dialogues
