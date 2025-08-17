from typing import Literal
from .base import HerBERTScoreBase

class HerbertScoreWarnings(HerBERTScoreBase):
    def _ensure_idf(self, attempted_operation: Literal["baseline computing", "sentence similarity scoring"]):
        if self.idf is None:
            raise RuntimeError(f"IDF must be computed or loaded before executing {attempted_operation} operation.")