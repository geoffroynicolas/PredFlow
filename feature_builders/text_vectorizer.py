from PredFlow.feature_builder import FeatureBuilder
from sklearn.feature_extraction.text import TfidfVectorizer
from dask.multiprocessing import get
import cleaning_services as cs
import dask.dataframe as dd


class TextVectorizer(FeatureBuilder):
    def _set_internal_state(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.8)

    def transform(self, df):
        return self.vectorizer.transform(df["text"])

    def fit(self, df_X_raw, df_y):
        self.vectorizer.fit(df_X_raw["text"])


# Singleton instance
text_vectorizer = TextVectorizer()
