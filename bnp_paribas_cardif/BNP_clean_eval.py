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
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

# import other methods
from clean import *

##### upload data
train = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/train.csv")
test = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/test.csv")
sample = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/sample_submission.csv")
trains = train.iloc[:5000,2:]
tests = test[:5000]


##### split and clean data
## define variables
y = train.target
columns = train.columns
x_train = train[columns[2:]]
x_test = train[columns[1:]]


##### generate pipeline
call = mypipeline(RandomForestClassifier)
call = call.fit(x_train,y)
y_predict = call.predict(x_test)


##### predict the model


#est = pipeline()
#est.set_parameters(dfsdfsd)

#est.fit(x,y)


