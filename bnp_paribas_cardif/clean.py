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


class ObjtoCatStrtoBinTrans(BaseEstimator, TransformerMixin):
    def __init__(self,columns=):
        # define columns
    
    def fit(self, X, y=None, **fit_params):
        # define mapping between strings and numbers

    def transform(self, X, **transform_params):
        # apply mapping from fit to the data
        return mapped_X

def objtocat_strtobin_trans(df):
    cols = df.columns
    for i in range(len(cols)):
        # convert to categorical
        df[cols[i]] = df[cols[i]].astype('category')
        # convert to integers (NaN == -1)
        df[cols[i]] = df[cols[i]].cat.codes
    return df


def ifNaN(df):
    n = df.isnull().sum().sum()
    if n == 0:
        return False
    else:
        return True

def matrixToDf(matrix, inidata):
    if inidata = None:
        newdf = pd.DataFrame(matrix)
    else:
        cols = inidata.columns
        newdf = pd.DataFrame(matrix)
        newdf.columns = cols
    return newdf
    

def distnormalvsnonnormal(dic, df):
    cols = df.columns[(df.dtypes == np.float64)]
    for i in range(len(cols)):
        df.update(df[cols[i]].notnull().apply(lambda x: (x - x.mean()) / (x.max() - x.min())).dropna())
        # check if normal distribution
        hyp = stats.shapiro(df[cols[i]])
        # if p-value < 0.05 is not normally distributed
        if hyp[1] < 0.05:
            dic[i] = "non-normal"
        else:
            dic[i] = "normal"
    return dic



def simple_classifier(Classifier):
    '''
    Returns an estimator that estimates the return probability of order
    positions which uses only the information available at the time shipping.
    '''
    pipeline = make_pipeline(
        NullToNaNTransformer(),
        make_union(
            make_pipeline(
                ColumnExtractor(columns=['a','b']),
                Imputer(strategy='median')
            ),
            make_pipeline(
                ColumnExtractor(columns=['c']),
                NaNToValueTransformer(value = 'unkown'),
                IntCategorizer(),
                Imputer(strategy='most_frequent'),
                OneHotEncoder()
            )
        ),
        Classifier()
        )
    return pipeline
