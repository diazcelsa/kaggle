###############
################   Clean Data + Generate Model
################   BNP Paribas Cardif Claims Management Kaggle Competition
################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from scipy import stats

# import other methods
from clean import *

##### upload data
train = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/train.csv")
test = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/test.csv")
sample = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/sample_submission.csv")
trains = train[:5000]
tests = test[:5000]


##### split and clean data
## define variables
y = train.target
columns = train.columns
x_train = train[columns[2:]]
x_test = train[columns[1:]]

## change null to nan
nton = NullToNaNTransformer()
ntonr = nton.fit(x_train)
x_tr_ntont = ntonr.transform(x_train)
x_te_ntont = ntonr.transform(x_test)

# include data into df
x_train = matrixToDf(x_tr_ntont, x_train)
x_test = matrixToDf(x_te_ntont, x_test)


#### continous data
x_train_a = x_train.loc[:, trains.dtypes == np.float64]
x_test_a = x_test.loc[:, trains.dtypes == np.float64]

## if nan change nan to median
nan = ifNaN(x_train_a)
if nan == True:
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    ntovr = imp.fit(x_train_a)
    x_tr_ntovt = ntovr.transform(x_train_a)
    x_te_ntovt = ntovr.transform(x_test_a)

    # add column names into df
    x_train_a = matrixToDf(x_tr_ntovt, x_train_a)
    x_test_a = matrixToDf(x_te_ntovt, x_test_a)


#### categorical data

### int categorical data
x_train_b = x_train.loc[:, trains.dtypes == np.int]
x_test_b = x_test.loc[:, trains.dtypes == np.int]

## if nan change nan to most frequent
nan = ifNaN(x_train_b)
if nan == True:
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    ntovr = imp.fit(x_train_b)
    x_tr_ntovt = ntovr.transform(x_train_b)
    x_te_ntovt = ntovr.transform(x_test_b)

    # add column names into df
    x_train_b = matrixToDf(x_tr_ntovt, x_train_b)
    x_test_b = matrixToDf(x_te_ntovt, x_test_b)

## transform each category to binary
enc = preprocessing.OneHotEncoder()
catobi = enc.fit(x_test_b)
x_tr_catobit = catobi.transform(x_train_b).toarray()
x_te_catobit = catobi.transform(x_test_b).toarray()

# include data into df
x_train_b = matrixToDf(x_tr_catobit)
x_test_b = matrixToDf(x_te_catobit)


### object categorical data
x_train_c = x_train.loc[:, trains.dtypes == np.object]
x_test_c = x_test.loc[:, trains.dtypes == np.object]

## change object to categorical & convert categories str to int (NaN = -1!!!)
x_tr_obtcatin = objtocat_strtobin_trans(x_train_c)
x_te_obtcatin = objtocat_strtobin_trans(x_test_c)

## if nan (-1) change nan (-1) to most frequent
nan = ifNaN(x_train_c)
if nan == True:
    imp = Imputer(missing_values='-1', strategy='most_frequent', axis=0)
    ntovr = imp.fit(x_train_c)
    x_tr_ntovt = ntovr.transform(x_train_c)
    x_te_ntovt = ntovr.transform(x_test_c)

    # add column names into df
    x_train_c = matrixToDf(x_tr_ntovt, x_train_c)
    x_test_c = matrixToDf(x_te_ntovt, x_test_c)

## transform each category to binary
enc = preprocessing.OneHotEncoder()
catobi = enc.fit(x_test_c)
x_tr_catobit = catobi.transform(x_train_c).toarray()
x_te_catobit = catobi.transform(x_test_c).toarray()

# include data into df
x_train_c = matrixToDf(x_tr_catobit)
x_test_c = matrixToDf(x_te_catobit)




##### generate pipeline
#mypipeline = make_pipeline(NullToNaNTransformer())
#x_train_afterothershit = mypipeline.fit(x_test)


##### predict the model


#est = pipeline()
#est.set_parameters(dfsdfsd)

#est.fit(x,y)


