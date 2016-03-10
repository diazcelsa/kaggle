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
from collections import defaultdict



class NullToNaNTrans(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return X.fillna(np.nan)


class ObjtoCatStrtoIntTrans(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        if columns == None:
            self.cols = X.columns
        else:
            self.cols = columns 

    def fit(self, X, y=None, **fit_params):
        # define mapping between strings and numbers
        self.mapping = []
        m = 0
        while m < len(self.cols):
            X[self.cols[i]] = X[self.cols[i]].astype('category')
            cats = X[self.cols[i]].dropna().unique()
            ncat = len(cats)
            d = {}
            a = 0
            while a < len(cats):
                d[cats[a]] = a+1 
                a += 1
            self.mapping[self.cols[i]] = d
            m += 1
        
    def transform(self, X, **transform_params):
        # apply mapping from fit to the data
        X_ = X.copy()
        m = 0
        while m < len(self.cols):
            X_[self.cols[i]] = X_[self.cols[i]].astype('category')
            val = X_[cols[m]]
            X_.ix[val, cols[m]] = X_.ix[val, cols[m]].map(lambda x: self.mapping[cols[m]][x])
            m += 1
        return X_


class DataSpliterTrans(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, dtype=None):
        self.dtype = dtype
        self.cols = columns

    def fit(self, X, y=None, **fit_params):
        # select data by datatype (np.int, np.float64, np.object)
        if self.dtype != None:
            self.cols = X.loc[:, X.dtypes == self.dtype].columns 

    def transform(self, X, **transform_params):
        return X.self.cols 


def simple_classifier(Classifier):
    '''
    Returns an estimator that estimates the return probability of order
    positions which uses only the information available at the time shipping.
    '''
    pipeline = make_pipeline(
        NullToNaNTrans(),
        make_union(
            make_pipeline(
                DataSpliterTrans(dtype='np.float64'),
                Imputer(strategy='median')
            ),
            make_pipeline(
                DataSpliterTrans(dtype='np.int'),
                Imputer(strategy='most_frequent'),
                preprocessing.OneHotEncoder()
            ),
            make_pipeline(
                DataSpliterTrans(dtype='np.object'),
                ObjtoCatStrtoIntTrans(),
                Imputer(strategy='most_frequent'),
                preprocessing.OneHotEncoder()
        ),
        Classifier()
        )
    return pipeline



# Simple functions

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

