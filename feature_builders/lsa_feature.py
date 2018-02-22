from PredFlow.feature_builder import FeatureBuilder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class LsaFeature(FeatureBuilder):
    def _set_internal_state(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.8)
        self.svd = TruncatedSVD(n_components=150)

    def transform(self, df):
        return self.svd.transform(self.vectorizer.transform(df["text"]))

    def fit(self, df_X_raw, df_y):
        vect_text = self.vectorizer.fit_transform(df_X_raw["text"])
        self.svd.fit(vect_text)


# Singleton instance
lsa_feature = LsaFeature()
