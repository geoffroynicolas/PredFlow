from PredFlow.feature_builder import FeatureBuilder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np


class NmfFeature(FeatureBuilder):
    def _set_internal_state(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.8)
        self.nmf = NMF(n_components=150)

    def transform(self, df):
        return self.nmf.transform(self.vectorizer.transform(df["text"]))

    def fit(self, df_X_raw, df_y):
        vect_text = self.vectorizer.fit_transform(df_X_raw["text"])
        self.nmf.fit(vect_text)

    def print_topics(self, n_max_elements=10):
        feature_names = self.vectorizer.get_feature_names()
        for icomp in range(len(self.nmf.components_)):
            curcomp = self.nmf.components_[icomp, :]
            print("TOPIC n°%s" % icomp)
            print([feature_names[i] for i in np.argsort(curcomp)[::-1][0:10]])


# Singleton instance
nmf_feature = NmfFeature()
