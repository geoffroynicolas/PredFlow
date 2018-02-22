from PredFlow.prediction_model import PredictionModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from bs4 import BeautifulSoup
import pandas as pd
import lightgbm as lgb
import numpy as np
import cleaning_services as cs
import re


class LgbModel(PredictionModel):
    def _set_internal_state(self):
        self.params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclassova',
            'num_boost_round': 10000,
            'metric': {'multi_error'},
            'learning_rate': 0.01,
            'feature_fraction': 1,
            'bagging_fraction': 0.8,
            'max_depth': 7,
            'bagging_freq': 5,
            'verbose': 0
        }

        self.lb = LabelEncoder()

    def train(self):
        # Fit label encoder
        self.lb.fit(pd.concat([self.y_train_array, self.y_test_array], axis=0))
        labels_train = self.lb.transform(self.y_train_array)
        labels_test = self.lb.transform(self.y_test_array)

        # Set params
        self.params['num_class'] = len(self.lb.classes_)

        # Train model
        dtrain = lgb.Dataset(data=self.X_train_array, label=labels_train)
        self.bst = lgb.train(params=self.params, train_set=dtrain,
                             verbose_eval=True)

    def _optimize(self):
        # Fit label encoder
        self.lb.fit(pd.concat([self.y_train_array, self.y_test_array], axis=0))
        labels_train = self.lb.transform(self.y_train_array)

        # Define parameters search grid
        gridParams = {
            'task': ['train'],
            'learning_rate': [0.01],
            'boosting_type': ['gbdt'],
            'metric': ['multi_error'],
            'objective': ['multiclassova','multiclass','cross_entropy'],
            'feature_fraction': [0.8, 0.9, 1],
            'subsample': [0.8, 0.9, 1],
            'max_depth': [3, 4, 7],
            'min_data_in_leaf': [1, 2, 3, 5, 10],
            'num_class': [len(self.lb.classes_)],
        }
        grid_params = ParameterGrid(gridParams)

        # Perform grid search
        dtrain = lgb.Dataset(data=self.X_train_array, label=labels_train)
        min_error = np.inf
        for params in grid_params:
            print()
            print("TESTED PARAMS : %s" % params)
            cv_results = lgb.cv(params=params,
                                train_set=dtrain,
                                num_boost_round=10000,
                                verbose_eval=True,
                                nfold=2,
                                stratified=True,
                                early_stopping_rounds=30)

            params["num_boost_round"] = len(list(cv_results.values())[0])
            error_key = [key for key in cv_results.keys() if re.match(r'.*-mean$', key)][0]
            current_error = cv_results[error_key][-1]
            print("CURRENT ERROR : %s" % current_error)
            min_error = current_error if min_error is None else min_error
            if current_error < min_error:
                min_error = current_error
                print("ERROR DIMINUTION : %s" % current_error)
                print("MIN ERROR : %s" % min_error)
                self.params = params
                self.best_score = min_error
                print("BEST CURRENT PARAMETERS %s" % self.params)
                print()
                self.save()

    def score(self):
        dtest = self.X_test_array
        pred = self.lb.inverse_transform(np.argsort(self.bst.predict(dtest), axis=1)[:, ::-1][:, 0]).tolist()
        truth = self.y_test_array.tolist()

        print(accuracy_score(pred, truth))

        return pd.DataFrame({"pred": pred, "truth": truth})

    def predict_from_feature_array(self, feature_array):
        return self.bst.predict(feature_array)

    def get_tag(self, text):
        cleaned_text = cs.f_clean_text(BeautifulSoup(text, "lxml").get_text())
        df_query = pd.DataFrame({"text": [cleaned_text]})
        pred_tag = [(self.lb.inverse_transform(i), pred) for i, pred in enumerate(self.predict(df_query)[0, :])]

        return sorted(pred_tag, key=lambda tup: tup[1])[::-1]


lgb_model = LgbModel()
