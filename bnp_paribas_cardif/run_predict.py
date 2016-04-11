################
################   Clean Data + Generate Model + Evaluate
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

%matplotlib inline

##### upload data
train = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/train.csv")
test = pd.read_csv("../../../github_data/bnp_paribas_cardif_data/test.csv")


##### split and clean data  
tr_a, te_a = train_test_split(train, train_size = 0.8)

## define variables 
y_train = tr_a.target
y_test = te_a.target
columns = train.columns
x_train = tr_a[columns[2:]]
x_test = te_a[columns[2:]]
y_testf = test[columns[2:]]


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
call = call.fit(x_train,y_train)

##### generate y_predict
y_predict = call.predict_proba(y_testf)

results = pd.DataFrame({'ID':test.ID,'PredictedProb':y_predict[:,1]})
results.to_csv('results.csv')


##### Compute ROC curve and ROC area for each fold
fpr0, tpr0, thresholds = metrics.roc_curve(y_test,y_predict[:,1])
a0 = metrics.auc(fpr0, tpr0)
ax = sns.set_style("whitegrid")
ax = sns.set_context("notebook", font_scale=1.1, rc={"lines.linewidth": 1.5})
plt.plot(fpr0, tpr0, label='ROC curve (area = %0.2f)' % a0)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for 3-fold CV before Grid Search')
plt.legend(loc="lower right")
plt.show()
plt.savefig("ROC_Curve")


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







