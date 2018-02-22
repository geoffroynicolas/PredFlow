from abc import ABCMeta, abstractmethod
from .dataframe_handler import DataFrameHandler


class DataCleaner(DataFrameHandler, metaclass=ABCMeta):

    @abstractmethod
    def _clean_to_df(self):
        pass

    def clean(self):
        print("Cleaning dataframes : %s" % self.get_name())
        self.df = self._clean_to_df()

    def core_task(self, prod_or_dev):
        self.clean()
