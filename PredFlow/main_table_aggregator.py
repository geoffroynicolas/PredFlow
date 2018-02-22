from abc import ABCMeta, abstractmethod
from PredFlow.dataframe_handler import DataFrameHandler
from PredFlow.pipeline_worker import WrongInputParameter, try_loading_from_saved_instance


class UnknownTargetColumn(Exception):
    pass


class MainTableAggregator(DataFrameHandler, metaclass=ABCMeta):
    @abstractmethod
    def _build_train_test_split(self):
        pass

    @abstractmethod
    def _aggregate_to_df(self):
        pass

    @abstractmethod
    def _set_target_column(self):
        pass

    @try_loading_from_saved_instance
    def __init__(self, sep='\t', encoding=None):
        super().__init__(sep, encoding)
        self.train_indexes = []
        self.test_indexes = []
        self.columns_without_target = []
        self.target_column = ""
        self.labels = []

    def _aggregate(self):
        print("Aggregating main table : %s" % self.get_name())
        self.df = self._aggregate_to_df()
        self.df.reset_index(inplace=True, drop=True)

    def get_data_set(self, set_type):
        df_X_raw = self.get_df().drop([self.target_column], axis=1)
        df_y = self.get_df()[self.target_column]

        if set_type == "full":
            # Do nothing
            pass
        elif set_type == "train":
            df_X_raw = df_X_raw.iloc[self.train_indexes]
            df_y = df_y.iloc[self.train_indexes]
        elif set_type == "test":
            df_X_raw = df_X_raw.iloc[self.test_indexes]
            df_y = df_y.iloc[self.test_indexes]
        else:
            raise WrongInputParameter('Invalid input parameter set_type. Must be in {"train", "test", "full"}')

        return df_X_raw.copy(), df_y.copy()

    def _check_target_column_existence(self):
        if self.target_column not in self.get_df().columns:
            raise UnknownTargetColumn("Target column not present in the main table.")

    def core_task(self, prod_or_dev):

        # Build modelisation
        self._aggregate()

        # Save informations
        self._set_target_column()
        self._check_target_column_existence()
        self.columns_without_target = [col for col in self.df.columns.tolist() if col != self.target_column]

        # Set train test split (default : full train - empty test)
        if prod_or_dev == "prod":
            self.train_indexes = list(range(self.get_df().shape[0]))
            self.test_indexes = []
        elif prod_or_dev == "dev":
            self._build_train_test_split()
        else:
            raise WrongInputParameter('Invalid input parameter prod_or_dev. Must be in {"dev", "prod"}')
        self.labels = self.get_df()[self.target_column].unique().tolist()
