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
nton = NullToNaNTrans()
ntonr = nton.fit(x_train)
x_tr_ntont = ntonr.transform(x_train)
x_te_ntont = ntonr.transform(x_test)


#### continous data
dat = DataSpliterTrans()
datc = dat.fit(x_tr_ntont, dtype='np.float64')
x_tr_datct = datc.transform(x_tr_ntont)
x_te_datct = datc.transform(x_te_ntont)

## if nan change nan to median
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
ntovr = imp.fit(x_tr_datct)
x_tr_ntovt = ntovr.transform(x_tr_datct)
x_te_ntovt = ntovr.transform(x_te_datct)


#### categorical data
### int categorical data
dat = DataSpliterTrans()
dati = dat.fit(x_tr_ntont, dtype='np.int')
x_tr_datit = dati.transform(x_tr_ntont)
x_te_datit = dati.transform(x_te_ntont)

## if nan change nan to most frequent
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
ntovr = imp.fit(x_tr_datit)
x_tr_ntovt = ntovr.transform(x_tr_datit)
x_te_ntovt = ntovr.transform(x_te_datit)

## transform each category to binary
enc = preprocessing.OneHotEncoder()
catobi = enc.fit(x_tr_ntovt)
x_tr_catobit = catobi.transform(x_tr_ntovt)
x_te_catobit = catobi.transform(x_te_ntovt)


### object categorical data
dat = DataSpliterTrans()
dato = dat.fit(x_tr_ntont, dtype='np.object')
x_tr_datot = dato.transform(x_tr_ntont,)
x_te_datot = dato.transform(x_te_ntont)

## change object to categorical & convert categories str to int
cat = ObjtoCatStrtoIntTrans()
cato = cat.fit(x_tr_datot)
x_tr_catot = cato.transform(x_tr_datot)
x_tr_catot = cato.transform(x_te_datot)

## if nan change nan to most frequent
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
ntovr = imp.fit(x_tr_catot)
x_tr_ntovt = ntovr.transform(x_tr_catot)
x_te_ntovt = ntovr.transform(x_tr_catot)

## transform each category to binary
enc = preprocessing.OneHotEncoder()
catobi = enc.fit(x_tr_ntovt)
x_tr_catobit = catobi.transform(x_tr_ntovt)
x_te_catobit = catobi.transform(x_te_ntovt)



##### generate pipeline
mypipeline = make_pipeline(NullToNaNTrans(),make_union(make_pipeline(DataSpliterTrans(dtype='np.float64'),Imputer(strategy='median')),make_pipeline(DataSpliterTrans(dtype='np.int'),Imputer(strategy='most_frequent'),preprocessing.OneHotEncoder()),make_pipeline(DataSpliterTrans(dtype='np.object'),ObjtoCatStrtoIntTrans(),Imputer(strategy='most_frequent'),preprocessing.OneHotEncoder()),Classifier()))
x_train_afterothershit = mypipeline.fit(x_test)


##### predict the model


#est = pipeline()
#est.set_parameters(dfsdfsd)

#est.fit(x,y)


