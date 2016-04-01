################
################   Clean Data
################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from scipy import stats
from collections import defaultdict



class NullToNaNTrans(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        print('NullToNaNTrans fit done.')
        return self

    def transform(self, X, **transform_params):
        print('NullToNaNTrans transform done.')
        return X.fillna(np.nan)


class ObjtoCatStrtoIntTrans(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
            self.cols = columns 

    def fit(self, X, y=None, **fit_params):
        # define mapping between strings and numbers
        if self.cols == None:
            self.cols = X.columns
        self.mapping = defaultdict(list)
        X_ = X.copy()
        m = 0
        while m < len(self.cols):
            X_[self.cols[m]] = X_[self.cols[m]].astype('category')
            cats = X_[self.cols[m]].dropna().unique()
            ncat = len(cats)
            d = defaultdict(lambda : np.nan)
            a = 0
            while a < len(cats):
                d[cats[a]] = a+1 
                a += 1
            self.mapping[self.cols[m]] = d
            m += 1
        print('ObjtoCatStrtoIntTrans fit done.')
        return self
        
    def transform(self, X, **transform_params):
        # apply mapping from fit to the data
        X_ = X.copy()
        m = 0
        while m < len(self.cols):
            val = X_[self.cols[m]].isnull()
            X_.loc[~val,self.cols[m]] = X_.loc[~val,self.cols[m]].map(lambda x: self.mapping[self.cols[m]][x])
            m += 1
        print(X_.shape)
        print('ObjtoCatStrtoIntTrans transform done.')
        return X_


class DataSpliterTrans(BaseEstimator, TransformerMixin):
    def __init__(self, data=None, columns=None, transp=False):
        self.dtype = data
        self.cols = columns
        self.transp = transp

    def fit(self, X, y=None, **fit_params):
        # select data by datatype (np.int, np.float64, np.object)
        if self.dtype != None:
            self.cols = X.loc[:, X.dtypes == self.dtype].columns
        print(self.cols)
        print('DataSpliterTrans fit done.')
        return self

    def transform(self, X, **transform_params):
        print('one')
        #X_ = [X[i] for i in self.cols]
        X_ = X[self.cols]
        print('two')
        X_ = DataFrame(X_)
        print('three')
        if self.transp == True:
            X_ = X_.transpose()
        print(X_)
        print(X_.shape)
        print('DataSpliterTrans transform done.')
        return X_

class Debugger(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        from IPython.core.debugger import Tracer
        Tracer()()
        return self

    def transform(self, X, **transform_params):
        return X

def PipelineBNP(Classifier):
    pipeline = make_pipeline(
        NullToNaNTrans(),
        make_union(
            make_pipeline(
                DataSpliterTrans(data=np.float64),
                Imputer(strategy='median')
            ),
            make_pipeline(
                DataSpliterTrans(data=np.int),
                Imputer(strategy='most_frequent'),
                preprocessing.OneHotEncoder()
            ),
            make_pipeline(
                DataSpliterTrans(data=np.object),
                ObjtoCatStrtoIntTrans(),
                Imputer(strategy='most_frequent'),
                preprocessing.OneHotEncoder()
            ),
        ),
        Classifier()
        )
    print('pipeline done.')
    return pipeline


def PipelineTelstra(Classifier):
    pipeline = make_pipeline(
        make_union(
            make_pipeline(
                DataSpliterTrans(columns='event_type'),
                DictVectorizer()
            ),
            make_pipeline(
                DataSpliterTrans(columns='severity_type'),
                DictVectorizer()
            ),
            make_pipeline(
                DataSpliterTrans(columns='resource_type'),
                DictVectorizer()
            ),
            make_pipeline(
                DataSpliterTrans(columns='volume'),
                DictVectorizer()
            ),
            make_pipeline(
                DataSpliterTrans(columns='log_feature'),
                DictVectorizer()
            ),
        ),
        Classifier()
        )
    print('pipeline done.')
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

def unique_column(df,colname):
    nouni = len(df[colname])
    uni = len(df[colname].unique())
    if nouni == uni:
        return True
    else:
        return False

def debugger(x):
    from IPython.core.debugger import Tracer
    return Tracer()()

def ifNaN(df):
    n = df.isnull().sum().sum()
    if n == 0:
        return False
    else:
        return True

def matrixToDf(matrix, inidata):
    if inidata == None:
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

