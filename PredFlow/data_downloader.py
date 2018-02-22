from abc import ABCMeta, abstractmethod
from .dataframe_handler import DataFrameHandler


class DataDownloader(DataFrameHandler, metaclass=ABCMeta):
    @abstractmethod
    def _download_to_df(self):
        pass

    def download(self):
        print("Downloading dataframe : %s" % self.get_name())
        self.df = self._download_to_df()

    def core_task(self, prod_or_dev):
        self.download()
