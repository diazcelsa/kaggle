################
################   Clean Data + Generate Model + Evaluate
################   BNPlean Data + Generate Model + Evaluate
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
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from scipy import stats

# import other methods
from data_modifier import *

##### upload data
train = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/train.csv")
test = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/test.csv")

print(train.shape,test.shape)

## define variables 
y_trainf = train.target
columns = train.columns
x_trainf = train[columns[2:]]
y_testf = test[columns[2:]]
print(y_trainf.shape,x_trainf.shape,y_testf.shape)

##### generate pipeline 
## X.shape (91456, 108)
parames = {'randomforestclassifier__max_depth': None,
               'randomforestclassifier__criterion': 'gini', 
                              'randomforestclassifier__n_estimators': 100, 
                                             'randomforestclassifier__max_leaf_nodes': None, 
                                                            'randomforestclassifier__min_samples_split': 2,
                                                                           'randomforestclassifier__min_samples_leaf': 1, 
                                                                                          'randomforestclassifier__min_weight_fraction_leaf': 0.0,
                                                                                                         'randomforestclassifier__n_jobs': 1
}
call = PipelineBNP(RandomForestClassifier)
call.set_params(**parames)
call = call.fit(x_trainf,y_trainf)

##### generate y_predict
y_predict = call.predict_proba(y_testf)

results = pd.DataFrame({'ID':test.ID,'PredictedProb':y_predict[:,1]})
results.to_csv('results.csv')


##### generate pipeline
#call = Pipeline(
#           NulltoNanTrans(),
#           make_union(
#               make_pipeline(
#                   DataSpliterTrans(dtype=np.float64),
#                   Imputer(strategy='median')
#               ),
#               make_pipeline(
#                   DataSpliterTrans(dtype=np.int),
#                   Imputer(strategy='most_frequent'),
#                   preprocessing.OneHotEncoder(handle_unknown='ignore')
#               ),
#               make_pipeline(
#                   DataSpliterTrans(dtype=np.object),
#                   ObjtoCatStrtoIntTrans(),
#                   Imputer(strategy='most_frequent'),                                                                                                                                                                        preprocessing.OneHotEncoder(handle_unknown='ignore')                                                                                                                                                  )
#           ),
#           RandomForestClassifier()
#           )


##### predict y
#call.set_parameters({'randomforestclassifier__max_depth': [None],
#                    'randomforestclassifier__criterion': ['gini'], 
#                    'randomforestclassifier__n_estimators':[2], 
#                    'randomforestclassifier__max_leaf_nodes':[None], 
#                    'randomforestclassifier__min_samples_split':[2],
#                    'randomforestclassifier__min_samples_leaf':[1], 
#                    'randomforestclassifier__min_weight_fraction_leaf':[0.0],
#                    'randomforestclassifier__n_jobs':[1]})
#call = call.fit(x_train,y_train)
#y_predict = call.predict_proba(x_test)







