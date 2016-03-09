################
################   Clean Data
################


import warnings; warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from scipy import stats



class NullToNaNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return X.fillna(np.nan)


class nantovalue_trans(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def checkifnan(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        pass


class objtocat_strtobin_trans(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X, **transform_params):
        pass


class cattobin_trans(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X, **transform_params):
        pass


def distnormalvsnonnormal(dicti, dataframe):
    pass




