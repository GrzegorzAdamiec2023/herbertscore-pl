from herBERTscore.base import HerBERTScoreBase


class HerBERTScoreExceptions(HerBERTScoreBase):
    def data_base_missing(self):
        if self.file_path is None:
            raise FileNotFoundError("File path not specified, you must specify a file path.")