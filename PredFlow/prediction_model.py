from abc import ABCMeta, abstractmethod
from PredFlow.pipeline_worker import PipelineWorker, try_loading_from_saved_instance
from PredFlow.main_table_aggregator import MainTableAggregator
from PredFlow.feature_builder import FeatureBuilder
from threading import Timer
from schedule import Scheduler
from datetime import datetime
import scipy.sparse as sp


# ---------- EXCEPTIONS ----------
class WrongPredictionDataFrame(Exception):
    pass


# ---------- DECORATORS ----------
def scheduled(method):
    def wrapper(self, *args, **kwargs):
        try:
            method(self, *args, **kwargs)
        finally:
            self._build_prod_thread().start()

    return wrapper


class PredictionModel(PipelineWorker, metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _optimize(self):
        pass

    @abstractmethod
    def score(self):
        pass

    @abstractmethod
    def predict_from_feature_array(self, feature_array):
        return None

    @try_loading_from_saved_instance
    def __init__(self):
        super().__init__()

        # Train/ test data
        self.X_train_array = sp.csr_matrix((0, 0))
        self.y_train_array = list()
        self.X_test_array = sp.csr_matrix((0, 0))
        self.y_test_array = list()

    def optimize(self, update=False):
        if update:
            self.update(prod_or_dev="dev", apply_to_pipeline=True, worker_types=[FeatureBuilder])
        self.X_train_array, self.y_train_array = self.build_data_set("train")
        self.X_test_array, self.y_test_array = self.build_data_set("test")
        self._optimize()

    def _build_scheduler(self):
        # Possibilities :
        #
        #  - Scheduler().every(10).minutes.do(job)
        #  - Scheduler().every(5).to(10).days.do(job)
        #  - Scheduler().every().hour.do(job, message='things')
        #  - Scheduler().every().day.at("10:30").do(job)

        sched = Scheduler().every().day.at("00:00").do(lambda x: x)

        return sched

    def _build_prod_thread(self):
        prod_script = self.__class__._prod_script
        interval = (self._build_scheduler().next_run - datetime.now()).seconds
        prod_thread = Timer(interval=interval, function=prod_script, args=(self,))
        prod_thread.daemon = True

        return prod_thread

    @scheduled
    def _prod_script(self):
        # Updating the pipeline in prod mod
        self.update(prod_or_dev="prod", apply_to_pipeline=True)

        # Saving model
        self.save(apply_to_pipeline=True)

    def start_prod(self):
        self._prod_script()

    def get_main_table(self):
        return self.get_child_workers_of_type(MainTableAggregator)[0]

    def build_data_set(self, set_type):
        # Build feature matrix
        X_array = sp.hstack([worker.get_feature_array(set_type=set_type) for worker in self.child_workers.values() if
                             isinstance(worker, FeatureBuilder)])

        # Get target array
        _, y_array = self.get_main_table().get_data_set(set_type=set_type)

        return X_array, y_array

    def predict(self, df):
        if self.lock:
            # Build feature array
            feature_array = sp.hstack([sp.csr_matrix(worker.transform(df)) for worker in self.child_workers.values() if
                                       isinstance(worker, FeatureBuilder)])

            # Return prediction
            return self.predict_from_feature_array(sp.csr_matrix(feature_array))

    def core_task(self, prod_or_dev):
        print("Building training data in prediction model %s" % self.get_name())
        set_type = "full" if prod_or_dev == "prod" else "train" if prod_or_dev == "dev" else None
        self.X_train_array, self.y_train_array = self.build_data_set(set_type=set_type)
        self.X_test_array, self.y_test_array = self.build_data_set("test")

        print("Training prediction model %s" % self.get_name())
        self.train()
