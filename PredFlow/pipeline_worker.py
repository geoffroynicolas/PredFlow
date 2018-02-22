import os
import threading
import pickle
from abc import ABCMeta, abstractmethod
from collections import OrderedDict


# ---------- EXCEPTIONS ----------
class NotExpectedWorker(Exception):
    pass


class MissingExpectedWorker(Exception):
    pass


class SingletonAlreadyInstantiated(Exception):
    pass


class WrongInputParameter(Exception):
    pass


# ---------- DECORATORS ----------
def pipeline_operation(method):
    def internal(self, method_to_call, apply_to_pipeline, updated_workers, worker_types, *args, **kwargs):

        # Depth first processing of the pipeline workers graph
        if apply_to_pipeline:
            for child_worker in self.child_workers.values():
                internal(child_worker, method_to_call, apply_to_pipeline, updated_workers, worker_types, *args,
                         **kwargs)

        if self not in updated_workers:
            if not worker_types:
                method_to_call(self, *args, **kwargs)
                updated_workers.add(self)
            else:
                if isinstance(self, tuple(worker_types)):
                    method_to_call(self, *args, **kwargs)
                    updated_workers.add(self)

    def wrapper(self, apply_to_pipeline=False, worker_types=[], *args, **kwargs):

        # First check pipeline config
        updated_workers = set()
        wtypes = []
        internal(self, method_to_call=self.__class__.check_worker_config, apply_to_pipeline=apply_to_pipeline,
                 updated_workers=updated_workers, worker_types=wtypes, *args, **kwargs)

        # Then apply method to pipeline
        updated_workers = set()
        internal(self, method_to_call=method, apply_to_pipeline=apply_to_pipeline, updated_workers=updated_workers,
                 worker_types=worker_types, *args, **kwargs)

    return wrapper


def try_loading_from_saved_instance(__init__):
    def wrapper(self, *args, **kwargs):
        print("Trying to load worker %s from saved instance." % self.get_name())
        instance = self.load()
        if instance is None:
            print("Load fail : Creating a new instance of class %s \n" % self.get_name())
            __init__(self, *args, **kwargs)
        else:
            print("Load success.\n")
            self.__dict__.update(instance.__dict__)

    return wrapper


# ---------- CORE CLASS ----------
class PipelineWorker(object, metaclass=ABCMeta):
    savedirpath = ""
    expected_child_workers_types = dict()

    @abstractmethod
    def core_task(self, prod_or_dev):
        pass

    def __init__(self, *args, **kwargs):
        self.child_workers = OrderedDict()
        self._set_internal_state()
        self.lock = threading.Lock()

    # To override if required
    def _set_internal_state(self):
        pass

    @classmethod
    def build_filepath(cls, extension=".pkl"):
        return os.path.normpath("/".join([cls.savedirpath, cls.__name__ + extension]))

    @classmethod
    def set_save_dir_path(cls, path):
        cls.savedirpath = path

    @classmethod
    def load(cls):
        try:
            filepath = cls.build_filepath()
            instance = None

            if os.path.isfile(filepath):
                print("Loading worker saved at path : %s" % filepath)
                with open(filepath, 'rb') as handle:
                    instance = pickle.load(handle)
                    instance.lock = threading.Lock()
            else:
                print("Worker loading impossible. File not found at path : %s" % filepath)

            return instance

        except:
            # Issue during loading. The pickle data may have been corrupted because of class code modification after
            #  pickling.
            return None

    def get_name(self):
        return self.__class__.__name__

    @classmethod
    def add_expected_worker_type(cls, worker_type):
        if cls not in cls.expected_child_workers_types:
            cls.expected_child_workers_types[cls] = set()
        cls.expected_child_workers_types[cls].add(worker_type)

    def register(self, *child_workers):
        cls = self.__class__
        for child_worker in child_workers:
            if cls in cls.expected_child_workers_types:
                if type(child_worker) in cls.expected_child_workers_types[cls]:
                    self.child_workers[child_worker.__class__] = child_worker
                else:
                    raise NotExpectedWorker("Incompatibles workers.")
            else:
                # No particular type of child worker expected. Registration is allowed.
                self.child_workers[child_worker.__class__] = child_worker

    def check_worker_config(self, *args, **kwargs):
        # Checking current worker config
        cls = self.__class__
        if cls in cls.expected_child_workers_types:
            for worker_type in cls.expected_child_workers_types[cls]:
                if worker_type not in [type(child_worker) for child_worker in self.child_workers.values()]:
                    raise MissingExpectedWorker(
                        "Missing not registered worker of type %s for worker %s" % self.get_name())

        print("Worker config is OK for %s" % self.get_name())

    def get_child_workers_of_type(self, cls):
        workers_list = list()

        def get_workers_of_type_internal(self_, cls_, workers_list_):
            for worker in self_.child_workers.values():
                get_workers_of_type_internal(worker, cls_, workers_list_)
                if isinstance(worker, cls) and worker not in workers_list_:
                    workers_list_.append(worker)

        get_workers_of_type_internal(self, cls, workers_list)

        return workers_list

    @pipeline_operation
    def erase_file(self):
        print("Erasing file for worker : %s" % self.get_name())
        filepath = self.build_filepath()
        if os.path.isfile(filepath):
            os.remove(filepath)

    @pipeline_operation
    def save(self):
        print("Saving worker : %s" % self.get_name())
        with open(self.build_filepath(), 'wb') as handle:
            self.lock = None
            child_workers_copy = self.child_workers.copy()
            self.child_workers.clear()

            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.lock = threading.Lock()
            self.child_workers = child_workers_copy

    @pipeline_operation
    def update(self, prod_or_dev="dev"):
        with self.lock:
            print("Updating in %s mode worker : %s " % (prod_or_dev, self.get_name()))
            self.core_task(prod_or_dev)
