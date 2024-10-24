from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
class BinaryClassifierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, binary_model=None, probability=True):
        self.binary_model = binary_model
        self.probability = probability

    def fit(self, X, y):
        self.binary_model.fit(X, y)
        return self

    def transform(self, X):
        if self.probability:
            binary_feature = self.binary_model.predict_proba(X)[:, 1]
        else:
            binary_feature = self.binary_model.predict(X)
        binary_feature = binary_feature.reshape(-1, 1)
        return np.hstack([X, binary_feature])
