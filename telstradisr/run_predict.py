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


# read the data
event = pd.read_csv("../../../github_data/telstradisr_data/event_type.csv")
log = pd.read_csv("../../../github_data/telstradisr_data/log_feature.csv")
sample = pd.read_csv("../../../github_data/telstradisr_data/sample_submission.csv")
severity = pd.read_csv("../../../github_data/telstradisr_data/severity_type.csv")
resource = pd.read_csv("../../../github_data/telstradisr_data/resource_type.csv")
train = pd.read_csv("../../../github_data/telstradisr_data/train.csv")
test = pd.read_csv("../../../github_data/telstradisr_data/test.csv")


# obtain integers out of strings with redundant data
event['event_type'] = event['event_type'].str.split(' ').str[1]
log['log_feature'] = log['log_feature'].str.split(' ').str[1]
severity['severity_type'] = severity['severity_type'].str.split(' ').str[1]
resource['resource_type'] = resource['resource_type'].str.split(' ').str[1]
test['location'] = test['location'].str.split(' ').str[1]
train['location'] = train['location'].str.split(' ').str[1]


# check in each log table if ids repeated
tables = [log,event,severity,resource] 
names = ['log','event','severity','resource'] 
for i in range(len(tables)):
    check = unique_column(tables[i],tables[i].columns[-1])
    # AGGREGATE CATEGORICAL VALUES INTO A DICTIO
    if check == False:
        if len(tables[i].columns) == 2:
            a = tables[i].groupby([tables[i].columns[-2],tables[i].columns[-1]]).agg({tables[i].columns[-1]:'count'})
            a.index.names = ['id','cat']
            a = a.reset_index()
            a = a.set_index('cat')
            a = a.groupby('id').apply(lambda x: {int(k):int(v) for k,v in x.to_dict()[tables[i].columns[-1]].items()})
            tables[i] = pd.DataFrame(a,columns=[tables[i].columns[-1]])
        elif len(tables[i].columns) == 3:
            a = tables[i].groupby([tables[i].columns[-3],tables[i].columns[-2],tables[i].columns[-1]]).agg({tables[i].columns[-1]:'count',tables[i].columns[-2]:'count'})
            a.index.names = ['id','cat1','cat2']
            a = a.reset_index()
            a = a.set_index('cat1')
            b = a.set_index('cat2')
            a = a.groupby('id').apply(lambda x: {int(k):int(v) for k,v in x.to_dict()[tables[i].columns[-1]].items()})
            b = b.groupby('id').apply(lambda x: {int(k):int(v) for k,v in x.to_dict()[tables[i].columns[-2]].items()})
            log1 = pd.DataFrame(a,columns=[tables[i].columns[-1]])
            log2 = pd.DataFrame(b,columns=[tables[i].columns[-2]])
event = tables[1]
severity = tables[2]
resource = tables[3]

# join tables with train data 
train = train.merge(event, left_on='id', right_index=True, how='left')
train = train.merge(severity, left_on='id', right_index=True, how='left')
train = train.merge(resource, left_on='id', right_index=True, how='left')
train = train.merge(log1, left_on='id', right_index=True, how='left')
train = train.merge(log2, left_on='id', right_index=True, how='left')

# join tables with test data
test = test.merge(event, left_on='id', right_index=True, how='left')
test = test.merge(severity, left_on='id', right_index=True, how='left')
test = test.merge(resource, left_on='id', right_index=True, how='left')
test = test.merge(log1, left_on='id', right_index=True, how='left')
test = test.merge(log2, left_on='id', right_index=True, how='left')


## define variables 
y_train = train.fault_severity
columns = train.columns
x_train = train[columns[3:]]
x_test = test[columns[3:]]


##### generate pipeline 
## shape 5904x615
parames = {'randomforestclassifier__max_depth': 8,
                       'randomforestclassifier__criterion': 'gini', 
                                      'randomforestclassifier__n_estimators': 500, 
                                                     'randomforestclassifier__max_leaf_nodes': None, 
                                                                    'randomforestclassifier__min_samples_split': 2,
                                                                                   'randomforestclassifier__min_samples_leaf': 4, 
                                                                                                  'randomforestclassifier__min_weight_fraction_leaf': 0.0,
                                                                                                                 'randomforestclassifier__n_jobs': 1
                                                                                                                 }

call = PipelineTelstra(RandomForestClassifier)
call.set_params(**parames)
call = call.fit(x_train,y_train)

##### generate y_predict
y_predict = call.predict(x_test)
y_predict = pd.get_dummies(y_predict)

results = pd.DataFrame({'id':test.id,'predict_0':y_predict[0],'predict_1':y_predict[1],'predict_2':y_predict[2]})
results.to_csv('results.csv',index=False)


