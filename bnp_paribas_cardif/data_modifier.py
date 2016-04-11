################
################   Structure Data
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



class NulltoNanTrans(BaseEstimator, TransformerMixin):
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
    def __init__(self, dtype=None, cols=None, transp=False, matrix=False):
        self.dtype = dtype
        self.cols = cols
        self.transp = transp
        self.matrix = matrix

    def fit(self, X, y=None, **fit_params):
        # select data by datatype (np.int, np.float64, np.object)
        if self.dtype != None:
            self.cols = X.loc[:, X.dtypes == self.dtype].columns
        print('DataSpliterTrans fit done.')
        return self

    def transform(self, X, **transform_params):
        #X_ = X.copy()
        if len([self.cols]) > 1:
            X_ = pd.DataFrame(X, columns=self.cols)
        elif len([self.cols]) == 1:
            X_ = pd.DataFrame(X, columns=self.cols)
        if self.matrix == True:
            X_ = X_.as_matrix()
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
        NulltoNanTrans(),
        make_union(
            make_pipeline(
                DataSpliterTrans(dtype=np.float64),
                Imputer(strategy='median')
            ),
            make_pipeline(
                DataSpliterTrans(dtype=np.int),
                Imputer(strategy='most_frequent'),
                preprocessing.OneHotEncoder(handle_unknown='ignore')
            ),
            make_pipeline(
                DataSpliterTrans(dtype=np.object),
                ObjtoCatStrtoIntTrans(),
                Imputer(strategy='most_frequent'),
                preprocessing.OneHotEncoder(handle_unknown='ignore')
            )
        ),
        Classifier()
        )
    print('pipeline done.')
    return pipeline


def PipelineTelstra(Classifier):
    pipeline = make_pipeline(
        make_union(
            make_pipeline(
                DataSpliterTrans(cols='event_type',matrix=True),
                DictVectorizer()
            ),
            make_pipeline(
                DataSpliterTrans(cols='severity_type',matrix=True),
                DictVectorizer()
            ),
            make_pipeline(
                DataSpliterTrans(cols='resource_type',matrix=True),
                DictVectorizer()
            ),
            make_pipeline(
                DataSpliterTrans(cols='volume',matrix=True),
                DictVectorizer()
            ),
            make_pipeline(
                DataSpliterTrans(cols='log_feature',matrix=True),
                DictVectorizer()
            )
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

def if_nan(df):
    n = df.isnull().sum().sum()
    if n == 0:
        return False
    else:
        return True

def distnormal_vs_nonnormal(dic, df):
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

