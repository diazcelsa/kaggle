################
################   Clean Data + Generate Model
################   BNP Paribas Cardif Claims Management Kaggle Competition
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
from scipy import stats


##### upload data
train = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/train.csv")
test = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/test.csv")
sample = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/sample_submission.csv")
trains = train[:5000]
tests = test[:5000]


##### clean data

#### continous data
## change null to nan
## if nan change nan to median

#### categorical data
### int categorical data
## change null to nan
## if nan change nan to most frequent
## transform each category to binary

### object categorical data
## change null to nan
## change object to categorical 
## convert categories str to int (NaN = -1!!!)
## if nan (-1) change nan (-1) to most frequent
## transform each category to binary


##### predict the model


est = pipeline()
est.set_parameters(dfsdfsd)

est.fit(x,y)


