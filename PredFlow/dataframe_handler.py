import pandas as pd
import os.path
from abc import ABCMeta
from .pipeline_worker import PipelineWorker, try_loading_from_saved_instance


class DataFrameHandler(PipelineWorker, metaclass=ABCMeta):
    @try_loading_from_saved_instance
    def __init__(self, sep='\t', encoding=None):
        super().__init__()
        self.sep = sep
        self.encoding = encoding
        self.df = pd.DataFrame()

    def to_csv(self):
        csv_filepath = self.build_filepath(extension=".csv")
        self.df.to_csv(path_or_buf=csv_filepath, sep=self.sep, encoding=self.encoding, index=False)

    def reset_csv(self):
        filepath = self.build_filepath(extension=".csv")
        if os.path.isfile(filepath):
            os.remove(filepath)

    def get_df(self):
        return self.df.copy()
