from abc import ABCMeta, abstractmethod
from PredFlow.pipeline_worker import PipelineWorker, try_loading_from_saved_instance, WrongInputParameter
from PredFlow.main_table_aggregator import MainTableAggregator
import scipy.sparse as sp


class MainTableNotFound(Exception):
    pass


class FeatureBuilder(PipelineWorker, metaclass=ABCMeta):
    @abstractmethod
    def transform(self, df):
        pass

    # Leave the possibility to build features using the target.
    @abstractmethod
    def fit(self, df_X_raw, df_y):
        pass

    @try_loading_from_saved_instance
    def __init__(self):
        super().__init__()
        self.feature_array = sp.csr_matrix((0, 0))

    def get_feature_array(self, set_type):
        main_table = self.get_child_workers_of_type(MainTableAggregator)[0]
        train_indexes = main_table.train_indexes
        test_indexes = main_table.test_indexes

        if set_type == "full":
            return self.feature_array
        elif set_type == "train":
            return self.feature_array[train_indexes, :]
        elif set_type == "test":
            return self.feature_array[test_indexes, :]
        else:
            raise WrongInputParameter('Invalid input parameter array_type. Must be in {"train", "test", "full"}')

    def core_task(self, prod_or_dev):
        print("Fitting feature builder : %s" % self.get_name())

        # Get main table workers
        main_table_worker = self.get_child_workers_of_type(MainTableAggregator)[0]

        # Get fitting data
        data_set_type = "full" if prod_or_dev == "prod" else "train" if prod_or_dev == "dev" else None
        df_X_raw, df_y = main_table_worker.get_data_set(data_set_type)

        # Fit and build feature array
        self.fit(df_X_raw, df_y)
        df_X_raw_full, _ = main_table_worker.get_data_set("full")
        print("Applying feature builder : %s" % self.get_name())
        self.feature_array = sp.csr_matrix(self.transform(df_X_raw_full))
