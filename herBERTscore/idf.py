import math
from collections import defaultdict
from .exceptions import HerBERTScoreExceptions

def compute_idf(tokenized_dataset: list[list[int]]) -> dict[int, float]:
    """
    Computes the Inverse Document Frequency (IDF) for a tokenized dataset.

    Args:
        tokenized_dataset (list of list of ints): Dataset where each document is represented as a list of integer tokens.

    Returns:
        dict: A dictionary where keys are tokens and values are their IDF scores.
    """
    # Total number of documents
    num_documents = len(tokenized_dataset)

    # Dictionary to store Document Frequency (DF) for each token
    document_frequency = defaultdict(int)

    # Calculate DF for each token
    for document in tokenized_dataset:
        # Get unique tokens in the document
        unique_tokens = set(document)
        for token in unique_tokens:
            document_frequency[token] += 1

    # Calculate IDF for each token
    idf = {}
    for token, df in document_frequency.items():
        idf[token] = math.log(num_documents / (1 + df))

    return idf

class HerBERTScoreIdf(HerBERTScoreExceptions):


    def make_idf(self):
        """
        Generates a dictionary of token IDs and their corresponding IDF values.
        :return: A dictionary where keys are token IDs and values are computed IDF values.
        """
        self.data_base_missing()

        # Read the file and load the texts line by line
        with open(self.file_path, "r", encoding="utf-8") as file:
            texts = [line.strip() for line in file if line.strip()]  # Skip empty lines

        # Tokenize each line using self.tokenizer
        tokenized_dataset = [self.tokenizer.encode(text, add_special_tokens=False) for text in texts]

        # Compute IDF values using the `compute_idf` function
        idf_scores = compute_idf(tokenized_dataset)

        self.idf =  idf_scores
        print("New idf has been set.")

    def _get_idf(self, token_id: int) -> float:
        if token_id == self.tokenizer.pad_token_id:
            # Because of that we are processing tokenized sentences in batch there will be pad tokens, but we can easily
            # ignore them by setting their idf to 0.
            return 0.0
        return self.idf.get(token_id, 1.0)