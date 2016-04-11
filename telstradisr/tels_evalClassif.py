{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/utils/fixes.py:64: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  if 'order' in inspect.getargspec(np.copy)[0]:\n"
     ]
    }
   ],
   "source": [
    "################\n",
    "################   Clean Data + join Data + Generate Model + Evaluate\n",
    "################   Telstra Network Disruptions Kaggle Competition\n",
    "################\n",
    "\n",
    "import warnings; warnings.simplefilter('ignore')\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from pandas import Series, DataFrame\n",
    "import seaborn as sns\n",
    "from sklearn import preprocessing\n",
    "from sklearn.feature_extraction import DictVectorizer\n",
    "from sklearn.preprocessing import Imputer\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.cross_validation import KFold\n",
    "from sklearn.cross_validation import train_test_split\n",
    "from sklearn import metrics\n",
    "from sklearn import grid_search\n",
    "from scipy import stats\n",
    "\n",
    "# import other methods\n",
    "from clean import *\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# read the data\n",
    "event = pd.read_csv(\"../../../github_data/telstradisr_data/event_type.csv\")\n",
    "log = pd.read_csv(\"../../../github_data/telstradisr_data/log_feature.csv\")\n",
    "sample = pd.read_csv(\"../../../github_data/telstradisr_data/sample_submission.csv\")\n",
    "severity = pd.read_csv(\"../../../github_data/telstradisr_data/severity_type.csv\")\n",
    "resource = pd.read_csv(\"../../../github_data/telstradisr_data/resource_type.csv\")\n",
    "train = pd.read_csv(\"../../../github_data/telstradisr_data/train.csv\")\n",
    "test = pd.read_csv(\"../../../github_data/telstradisr_data/test.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# obtain integers out of strings with redundant data\n",
    "event['event_type'] = event['event_type'].str.split(' ').str[1]\n",
    "log['log_feature'] = log['log_feature'].str.split(' ').str[1]\n",
    "severity['severity_type'] = severity['severity_type'].str.split(' ').str[1]\n",
    "resource['resource_type'] = resource['resource_type'].str.split(' ').str[1]\n",
    "test['location'] = test['location'].str.split(' ').str[1]\n",
    "train['location'] = train['location'].str.split(' ').str[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# check in each log table if ids repeated\n",
    "tables = [log,event,severity,resource] \n",
    "names = ['log','event','severity','resource'] \n",
    "for i in range(len(tables)):\n",
    "    check = unique_column(tables[i],tables[i].columns[-1])\n",
    "    \n",
    "    # AGGREGATE CATEGORICAL VALUES INTO A DICTIO\n",
    "    if check == False:\n",
    "        if len(tables[i].columns) == 2:\n",
    "            a = tables[i].groupby([tables[i].columns[-2],tables[i].columns[-1]]).agg({tables[i].columns[-1]:'count'})\n",
    "            a.index.names = ['id','cat']\n",
    "            a = a.reset_index()\n",
    "            a = a.set_index('cat')\n",
    "                    \n",
    "            # take care that keys are still integers in the dict            \n",
    "            a = a.groupby('id').apply(lambda x: {int(k):int(v) for k,v in x.to_dict()[tables[i].columns[-1]].items()})\n",
    "            tables[i] = pd.DataFrame(a,columns=[tables[i].columns[-1]])\n",
    "            \n",
    "        elif len(tables[i].columns) == 3:\n",
    "            a = tables[i].groupby([tables[i].columns[-3],tables[i].columns[-2],tables[i].columns[-1]]).agg({tables[i].columns[-1]:'count',tables[i].columns[-2]:'count'})\n",
    "            a.index.names = ['id','cat1','cat2']\n",
    "            a = a.reset_index()\n",
    "            a = a.set_index('cat1')\n",
    "            b = a.set_index('cat2')\n",
    "            \n",
    "            # take care that keys are still integers in the dict\n",
    "            a = a.groupby('id').apply(lambda x: {int(k):int(v) for k,v in x.to_dict()[tables[i].columns[-1]].items()})\n",
    "            b = b.groupby('id').apply(lambda x: {int(k):int(v) for k,v in x.to_dict()[tables[i].columns[-2]].items()})\n",
    "            log1 = pd.DataFrame(a,columns=[tables[i].columns[-1]])\n",
    "            log2 = pd.DataFrame(b,columns=[tables[i].columns[-2]])\n",
    "event = tables[1]\n",
    "severity = tables[2]\n",
    "resource = tables[3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# join tables with train data \n",
    "train = train.merge(event, left_on='id', right_index=True, how='left')\n",
    "train = train.merge(severity, left_on='id', right_index=True, how='left')\n",
    "train = train.merge(resource, left_on='id', right_index=True, how='left')\n",
    "train = train.merge(log1, left_on='id', right_index=True, how='left')\n",
    "train = train.merge(log2, left_on='id', right_index=True, how='left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# join tables with test data\n",
    "test = test.merge(event, left_on='id', right_index=True, how='left')\n",
    "test = test.merge(severity, left_on='id', right_index=True, how='left')\n",
    "test = test.merge(resource, left_on='id', right_index=True, how='left')\n",
    "test = test.merge(log1, left_on='id', right_index=True, how='left')\n",
    "test = test.merge(log2, left_on='id', right_index=True, how='left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "##### split data  \n",
    "tr_a, te_a = train_test_split(train, train_size = 0.8)\n",
    "\n",
    "## define variables \n",
    "y_train = tr_a.fault_severity\n",
    "y_test = te_a.fault_severity\n",
    "columns = train.columns\n",
    "x_train = tr_a[columns[3:]]\n",
    "x_test = te_a[columns[3:]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "## define parameters\n",
    "param_grid_1 =  {'randomforestclassifier__max_depth': [8,16],\n",
    "               'randomforestclassifier__criterion': ['gini'], \n",
    "               'randomforestclassifier__n_estimators':[5,20], \n",
    "               'randomforestclassifier__max_leaf_nodes':[None], \n",
    "               'randomforestclassifier__min_samples_split':[2],\n",
    "               'randomforestclassifier__min_samples_leaf':[10], \n",
    "               'randomforestclassifier__min_weight_fraction_leaf':[0.0],\n",
    "               'randomforestclassifier__n_jobs':[1]\n",
    "}\n",
    "param_grid_2 =  {'randomforestclassifier__max_depth': [4,8],\n",
    "               'randomforestclassifier__criterion': ['gini'], \n",
    "               'randomforestclassifier__n_estimators':[5], \n",
    "               'randomforestclassifier__max_leaf_nodes':[None], \n",
    "               'randomforestclassifier__min_samples_split':[2],\n",
    "               'randomforestclassifier__min_samples_leaf':[1,10], \n",
    "               'randomforestclassifier__min_weight_fraction_leaf':[0.0],\n",
    "               'randomforestclassifier__n_jobs':[1]\n",
    "}\n",
    "param_grid_3 =  {'randomforestclassifier__max_depth': [8],\n",
    "               'randomforestclassifier__criterion': ['gini'], \n",
    "               'randomforestclassifier__n_estimators':[10,100], \n",
    "               'randomforestclassifier__max_leaf_nodes':[None], \n",
    "               'randomforestclassifier__min_samples_split':[2],\n",
    "               'randomforestclassifier__min_samples_leaf':[1], \n",
    "               'randomforestclassifier__min_weight_fraction_leaf':[0.0],\n",
    "               'randomforestclassifier__n_jobs':[1]\n",
    "}\n",
    "param_grid_4 =  {'randomforestclassifier__max_depth': [8],\n",
    "               'randomforestclassifier__criterion': ['gini'], \n",
    "               'randomforestclassifier__n_estimators':[5,30], \n",
    "               'randomforestclassifier__max_leaf_nodes':[None], \n",
    "               'randomforestclassifier__min_samples_split':[2],\n",
    "               'randomforestclassifier__min_samples_leaf':[1,4], \n",
    "               'randomforestclassifier__min_weight_fraction_leaf':[0.0],\n",
    "               'randomforestclassifier__n_jobs':[1]\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "init event_type\n",
      "init severity_type\n",
      "init resource_type\n",
      "init volume\n",
      "init log_feature\n",
      "pipeline done.\n",
      "init event_type\n",
      "init severity_type\n",
      "init resource_type\n",
      "init volume\n",
      "init log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "init event_type\n",
      "init severity_type\n",
      "init resource_type\n",
      "init volume\n",
      "init log_feature\n",
      "init event_type\n",
      "init severity_type\n",
      "init resource_type\n",
      "init volume\n",
      "init log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before event_type\n",
      "DataSpliterTrans fit done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit after event_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  event_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before severity_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after severity_type\n",
      "fit before event_type\n",
      "columns:  severity_type\n",
      "DataSpliterTrans fit done.\n",
      "DataSpliterTrans transform done.\n",
      "fit after event_type\n",
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before resource_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after resource_type\n",
      "columns:  resource_type\n",
      "fit before severity_type\n",
      "DataSpliterTrans fit done.\n",
      "DataSpliterTrans transform done.\n",
      "fit after severity_type\n",
      "columns:  severity_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before volume\n",
      "DataSpliterTrans fit done.\n",
      "fit after volume\n",
      "columns:  volume\n",
      "DataSpliterTrans transform done.\n",
      "fit before resource_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after resource_type\n",
      "columns:  resource_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before log_feature\n",
      "DataSpliterTrans fit done.\n",
      "fit after log_feature\n",
      "columns:  log_feature\n",
      "DataSpliterTrans transform done.\n",
      "fit before volume\n",
      "DataSpliterTrans fit done.\n",
      "fit after volume\n",
      "columns:  volume\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans fit done.\n",
      "fit after log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  severity_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  resource_type\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  volume\n",
      "DataSpliterTrans transform done.\n",
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n",
      "columns:  log_feature\n",
      "DataSpliterTrans transform done.\n",
      "columns:  severity_type\n",
      "DataSpliterTrans transform done.\n",
      "columns:  resource_type\n",
      "DataSpliterTrans transform done.\n",
      "DataSpliterTrans transform done.\n",
      "columns:  volume\n",
      "columns:  log_feature\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before event_type\n",
      "DataSpliterTrans transform done.\n",
      "DataSpliterTrans fit done.\n",
      "fit after event_type\n",
      "columns:  event_type\n",
      "fit before severity_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after severity_type\n",
      "columns:  severity_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before resource_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans fit done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit after resource_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  resource_type\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "init event_type\n",
      "init severity_type\n",
      "init resource_type\n",
      "init volume\n",
      "init log_feature\n",
      "init event_type\n",
      "init severity_type\n",
      "init resource_type\n",
      "init volume\n",
      "init log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before volume\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans fit done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit after volume\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  volume\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before event_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after event_type\n",
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before severity_type\n",
      "fit before log_feature\n",
      "DataSpliterTrans fit done.\n",
      "DataSpliterTrans fit done.\n",
      "fit after log_feature\n",
      "fit after severity_type\n",
      "columns:  severity_type\n",
      "columns:  log_feature\n",
      "DataSpliterTrans transform done.\n",
      "DataSpliterTrans transform done.\n",
      "fit before resource_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after resource_type\n",
      "columns:  resource_type\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before volume\n",
      "DataSpliterTrans fit done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit after volume\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  volume\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans fit done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit after log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  severity_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  resource_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  volume\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  log_feature\n",
      "DataSpliterTrans transform done.\n",
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n",
      "columns:  severity_type\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  resource_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  volume\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before event_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after event_type\n",
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n",
      "columns:  log_feature\n",
      "DataSpliterTrans transform done.\n",
      "fit before severity_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after severity_type\n",
      "columns:  severity_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before resource_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after resource_type\n",
      "columns:  resource_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before volume\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "init event_type\n",
      "init severity_type\n",
      "init resource_type\n",
      "init volume\n",
      "init log_feature\n",
      "init event_type\n",
      "init severity_type\n",
      "init resource_type\n",
      "init volume\n",
      "init log_feature\n",
      "DataSpliterTrans transform done.\n",
      "DataSpliterTrans fit done.\n",
      "fit after volume\n",
      "columns:  volume\n",
      "fit before log_feature\n",
      "DataSpliterTrans fit done.\n",
      "fit after log_feature\n",
      "columns:  log_feature\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before event_type\n",
      "DataSpliterTrans fit done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit after event_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  event_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before severity_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after severity_type\n",
      "columns:  severity_type\n",
      "DataSpliterTrans transform done.\n",
      "columns:  severity_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before resource_type\n",
      "DataSpliterTrans fit done.\n",
      "columns:  resource_type\n",
      "fit after resource_type\n",
      "DataSpliterTrans transform done.\n",
      "columns:  resource_type\n",
      "DataSpliterTrans transform done.\n",
      "columns:  volume\n",
      "DataSpliterTrans transform done.\n",
      "fit before volume\n",
      "DataSpliterTrans fit done.\n",
      "fit after volume\n",
      "columns:  volume\n",
      "DataSpliterTrans transform done.\n",
      "columns:  log_feature\n",
      "DataSpliterTrans transform done.\n",
      "fit before log_feature\n",
      "DataSpliterTrans fit done.\n",
      "fit after log_feature\n",
      "columns:  log_feature\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  severity_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before event_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after event_type\n",
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n",
      "columns:  resource_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before severity_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after severity_type\n",
      "columns:  severity_type\n",
      "DataSpliterTrans transform done.\n",
      "columns:  volume\n",
      "DataSpliterTrans transform done.\n",
      "fit before resource_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after resource_type\n",
      "columns:  resource_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before volume\n",
      "DataSpliterTrans fit done.\n",
      "fit after volume\n",
      "columns:  volume\n",
      "DataSpliterTrans transform done.\n",
      "columns:  log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "init event_type\n",
      "init severity_type\n",
      "init resource_type\n",
      "init volume\n",
      "init log_feature\n",
      "DataSpliterTrans transform done.\n",
      "fit before log_feature\n",
      "DataSpliterTrans fit done.\n",
      "DataSpliterTrans transform done.\n",
      "fit after log_feature\n",
      "columns:  log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before event_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans fit done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit after event_type\n",
      "columns:  event_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before severity_type\n",
      "DataSpliterTrans fit done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit after severity_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  severity_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "init event_type\n",
      "init severity_type\n",
      "init resource_type\n",
      "init volume\n",
      "init log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before resource_type\n",
      "DataSpliterTrans fit done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit after resource_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  resource_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before volume\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans fit done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit after volume\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  volume\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit before log_feature\n",
      "DataSpliterTrans fit done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fit after log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  severity_type\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  resource_type\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  volume\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  log_feature\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n",
      "columns:  severity_type\n",
      "DataSpliterTrans transform done.\n",
      "columns:  resource_type\n",
      "DataSpliterTrans transform done.\n",
      "columns:  volume\n",
      "DataSpliterTrans transform done.\n",
      "columns:  log_feature\n",
      "DataSpliterTrans transform done.\n",
      "init event_type\n",
      "init severity_type\n",
      "init resource_type\n",
      "init volume\n",
      "init log_feature\n",
      "fit before event_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after event_type\n",
      "columns:  event_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before severity_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after severity_type\n",
      "columns:  severity_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before resource_type\n",
      "DataSpliterTrans fit done.\n",
      "fit after resource_type\n",
      "columns:  resource_type\n",
      "DataSpliterTrans transform done.\n",
      "fit before volume\n",
      "DataSpliterTrans fit done.\n",
      "fit after volume\n",
      "columns:  volume\n",
      "DataSpliterTrans transform done.\n",
      "fit before log_feature\n",
      "DataSpliterTrans fit done.\n",
      "fit after log_feature\n",
      "columns:  log_feature\n",
      "DataSpliterTrans transform done.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n",
      "/usr/local/lib/python3.5/site-packages/sklearn/base.py:175: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead\n",
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    }
   ],
   "source": [
    "##### generate pipeline \n",
    "## shape 5904x615\n",
    "call = PipelineTelstra(RandomForestClassifier)\n",
    "# check that the name of the attributes are the same as the parameter's name\n",
    "gs = grid_search.GridSearchCV(call, param_grid_4, cv=2, n_jobs=2, pre_dispatch='n_jobs') # check scoring function\n",
    "gs = gs.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[mean: 0.65650, std: 0.00418, params: {'randomforestclassifier__criterion': 'gini', 'randomforestclassifier__n_jobs': 1, 'randomforestclassifier__n_estimators': 5, 'randomforestclassifier__min_samples_split': 2, 'randomforestclassifier__min_weight_fraction_leaf': 0.0, 'randomforestclassifier__min_samples_leaf': 1, 'randomforestclassifier__max_depth': 8, 'randomforestclassifier__max_leaf_nodes': None},\n",
       " mean: 0.67818, std: 0.01039, params: {'randomforestclassifier__criterion': 'gini', 'randomforestclassifier__n_jobs': 1, 'randomforestclassifier__n_estimators': 30, 'randomforestclassifier__min_samples_split': 2, 'randomforestclassifier__min_weight_fraction_leaf': 0.0, 'randomforestclassifier__min_samples_leaf': 1, 'randomforestclassifier__max_depth': 8, 'randomforestclassifier__max_leaf_nodes': None},\n",
       " mean: 0.68005, std: 0.01654, params: {'randomforestclassifier__criterion': 'gini', 'randomforestclassifier__n_jobs': 1, 'randomforestclassifier__n_estimators': 5, 'randomforestclassifier__min_samples_split': 2, 'randomforestclassifier__min_weight_fraction_leaf': 0.0, 'randomforestclassifier__min_samples_leaf': 4, 'randomforestclassifier__max_depth': 8, 'randomforestclassifier__max_leaf_nodes': None},\n",
       " mean: 0.67293, std: 0.00616, params: {'randomforestclassifier__criterion': 'gini', 'randomforestclassifier__n_jobs': 1, 'randomforestclassifier__n_estimators': 30, 'randomforestclassifier__min_samples_split': 2, 'randomforestclassifier__min_weight_fraction_leaf': 0.0, 'randomforestclassifier__min_samples_leaf': 4, 'randomforestclassifier__max_depth': 8, 'randomforestclassifier__max_leaf_nodes': None}]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##### plot the roc_auc to compare parameters\n",
    "sco = gs.grid_scores_\n",
    "sco"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# extract data from scoring function\n",
    "meanv = np.zeros(len(sco))\n",
    "deptv = np.zeros(len(sco),dtype=np.int)\n",
    "estiv = np.zeros(len(sco),dtype=np.int)\n",
    "deptk = [i for i in sco[0].parameters.keys()][5]\n",
    "estik = [i for i in sco[0].parameters.keys()][2]\n",
    "\n",
    "l = 0\n",
    "while l < len(sco):\n",
    "    mean = sco[l].mean_validation_score\n",
    "    par = sco[l].parameters.values()\n",
    "    values = [v for v in par]\n",
    "    meanv[l] = mean\n",
    "    estiv[l] = values[2]\n",
    "    deptv[l] = values[5]\n",
    "    l += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAWYAAAEKCAYAAAAhEP83AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAGLdJREFUeJzt3X+0XWV95/H3udFmkJ+11IqdqYLCx3ENEE1iCIYfYpAC\nola7rKhFUhALtDMOC6aCdSq21bEKdhgEhVCQVqmiBpCaiKLQGErQCEosfgKG/tKxVkaIFSOG3Pnj\neY4cDjc5F3Mvd+fJ57XWWevss389+5x9vvt7vvvZ+/TGx8eJiIjuGJvpBkRExKMlMEdEdEwCc0RE\nxyQwR0R0TAJzRETHJDBHRHRMAvM2kjRP0sUz3Y7JkPRFSa/ahvmPkXRufX6cpD+fwra9RtLXJN0h\n6fOSnjOJeeZKuncb13uJpOfX59v0/kyXwX2sbvPHp3DZb5d03FQtL6ZGAvO2+y/Ar850I54g84Ff\nBLD9adtvmYqFSnoacBFwtO05wDLgwknOvq0d8Y8Eetu4jOn2s33M9hrbr5nCZR8BPHkKlxdToNfl\nC0wkHQacB/wj8BzgQeBE25a0L/ABYGfgGcAdwG/ZfkjSRuBa4ADg9cAc4BTKDvhU4H/Z/pCkNwKv\nBnYCngX8U13m7wH7Au+3fX5ty+8Ap1G+xPfVaR4EVgG7AZ+yfVLNPt5W1/UgcKbt1ZL+CFgI7AV8\nDfhT4DJgdl3mZbYfk3lLWgi8B3gKsBl4h+3PSFoFnGf7U3W6d9dZ/hi4uLb/qcAPgdfZvlvSF4H/\nA6wB1tretc77zP6wpKdMND8lIF9LOZhfAtwD/Kbt4yT9ap3nWbUNV9p+X13ujcBngAV1GW+zffUE\n2znL9sOSngS8C3iG7TdMMN2pwFuA+4FvAIfb3qeOOwd4VW3jPwCn2f5u3e67gLnALwF/aftcSX8C\nnAWsB04A/gz4Z0DA04AbbZ88tP59gVuAvWxvkjRG2T+PBJ5H+ewfro+zbH9pgm0Y3pd+v+7Tiyj7\n+xjlgPNu4MsM7GPAlcCFtveXdDnwY8oB81eAq4F/A46rwyfbvmlL3xXgZMq+9T3gDOCLdbo5lH1t\nBXC27c0TfKdeUR8P1W040fa/Dm9r/Hy2h4x5DvDntg8ErgD+qr7+JuAK2y+iBJF9gGPruF8ArrX9\nnwEDJ1GysbnAa4H3Dix/EfBG2/tSdubfsn1EXdafwM8OEG8EFtVlvBdYZvtfgP8JrKxB+TmUgNtf\n15uBZZJ2quv6NWCO7RMoAeE62/Prug4Z3nBJewCXA2+wPY/yRfigpP8IXAosqdONAW8AlgJHAz+w\nfbDt5wJfoRxEhg0fkfvDE85v+zbgg8DHbL99aJ6PUILYAfX9fIOkfla3D7Dc9gLgrUPv/c/UoDyX\nEhjfRDmADL8fBwJ/RPkcFlAOfP1xJwD7Ay+0/QJgOeXA1/cs4GDgBcBrJR1j+w+B71AOXF+u0+1S\nl/084GhJLxpq593AWuDl9aWjgHttf5MS2E+1/ULg7cDhE2zDoTx2X/pUHf0OysF2PmWfPWJ4H6vT\nDX52cygHvfnAfwc21O/EBZT3G7bwXbF9EeXzPdP2tXWe79veH5gHHAicWZcx+J36HvDfgPl1W2+o\nbYgp8qSZbsAkrLW9sj7/C+ADkn4R+APgSElnAftRMtFdBub7EoDtH9Us9mU1c5hDyRz6vmz7O/X5\nvZSdDOBbwOyaQR4DPBu4RVL/Z+8eNXAOOhJ4OnDjwHSbKNk+wK22+1+qZcCHJS0APg/81wm2vZ9h\nXzOwvM2UrOXjwHtrGWAecLftbwHfkrRe0u/V9R5OyfAmxfYnH8/89f15Ud12bG+QdAUlwK8GHrK9\nvE7+VWopZAvrXgPsJeko4DOS9ra9YWCSlwCftf1vdfhDlM8GysFtPrBGEpSkY6eBeS+x/TCwQdLV\nlID6mTpusJTxsdqWH0u6m5I5D1sKnEgJqCdSDpIAV1E+q78BPkcJ1MOOZcv70scp+/fLKfvEORPM\nP+zTtjcD/yrpR8Bn6+vfovzigdHflX47jqYcvLD9U0kfpATg/nb0s/9vU7Lu2yUtpxx4vzCJtsYk\nbQ8Z86aB5/32Pgz8NSUT+AfgfOB2Hv0F+3eA+jP7Dkq2uhL4w6Hl/2Ro+KcTtGEW5efvC2w/3/bz\ngYW2759guhuHpltE+cn9szYB2P4bSvbyMcrBYq2kvSdY3t8PLe9FlOD0IOWn6+sZCA71p/5lwI8o\nmexVPLaGOs6jP/tf6D+Z5PyDJtqHxnikbvnQ0HofsyxJe0l6aX/Y9meBDZQANtzuwfkH941ZwHsG\n3qd5wKFbmHaMsg9NZPDzn7C9wCeABZKeW9dxdW332ymB7cuUz+TWCebd4r5k+xJK1n8D5cBxp6Rd\nt9DOvsnsv6O+K33Dn+Xg5wh1/7U9bvtwSub/feD9U3kiOLaPwHyApAPq81OAL9Us6qXAO2u9skf5\nKTVrgvnnAd+z/ae2P0epvzGQrWxNf5obgOMlPb3OewqPZCabeGTn/QLwUtWUrWZ+t1PqyI8i6SPA\na21/HDgdeAD4T0OT3QrsW3/+Iml/4JuUOiE8krktBD5ZX3spcLnty4G76/YOvy/3A0+ugQVKXZZJ\nzD+4rQDY/vfaztNrG3en1Gv7vzyG3+eJ3vf/AHxMUr9W/OK6zruGpvsc5f3tb/+SgXGfBU4eCGTn\nAh8eGP96Sb36a+s1wHVb2qZRbP+EckC9Avik7Y2SZtUeIrvUAHsa8FxJw8ve4r5Uzxu8wPaVlDLY\n7pRfGI+7jUO29l0ZXPYKHvkcZ1O+bzcwRNIBktYCd9l+D/B+yq+4mCLbQ2D+HnCupK9TaqxvrK+f\nTfnZuIpSz/skj5QMBmtwNwD/IsmS/hbYCHx3YNpBE9Zdbd9AOUnyOUl3UOq5v1Gn+TvKF/CTtv+e\nsjP/taTbKTXq42z/eIJ1vZMSLG6nBLZP2f7bwQlsf59ycvLP6nr/ilJv/uc6/quUDOkTtvuZ6fuA\n35X0ZUqmdM3w+1IPbP8DWCFpNY/OHrc2/43AyyX976FteQOwuH5GtwJX1+Cyxfd0aDvvBX4H+JSk\nr1I+z5fZ3jg03dra7i9Iuo1yQrRvKXA9cKukOyn10TcOjJ8N3Eb5vD5g+6b6+jWUg8LiybR1wKWU\n0smltW0PU372f1TSGkpZYontR2WwI/als4B31vlvpJzo/ScG9rGttGdr7T2HLX9XPg28T9JvU8pp\nv1Lfv69RkoB3DS/b9tcpB6Y1dT9ZQqlvxxTZHnplXGz7eTPdlth+1V4ZF9dfJxGdtz2c/IvYVt3N\nPqKTaqnzIsovr42UrofrB8bPp3RthHIy9ARKWWgppbvlw8CbbK+T9GxK2WszpTPD6aPW3+lShu2b\nky3HtrJ9RLLleJxeCcy2fTClbHr+0PhLKH23D6WUnfam1PJ3tr2Icj1Bvwx0PnCO7cOAMUmvGLXy\nTgfmiIgZsohyMhTbqymdCACQtB/lopozJN0E7GF7HSWz3r1m27vzSI+kuQNdfpcDi0etPKWMiIjH\n2o3SU6pvk6Sx2md8T0pPqNMoV41eL+krlO64O1FOmv4S8LIJlvtDStDeqmkNzA9tuC+1vXiMYxa+\neaabEB30+W98YpvvWXLAMw+bdMz5+j/evLX1bQAG+5D3gzKUbPmemiUjaQUlo14IrLL9tnp17hdq\nF9fBNu1K6a66VSllREQzer3epB8jrKJeVSrpIODOgXHrgV36/e4pt1P4BuWK4n6W/QNK4jsGfLV/\nLQLl6sqVjJBSRkQ0o9ebslxzGeUy9lV1eImk4ykn95ZKOgm4ql5Ldovt5ZL+Drhc0kpKbD27Xtp/\nJnBpvdjoLsqVo1vfjunsx5xSRkwkpYyYyFSUMubsfcSkY84d936hs7d7TcYcEc2YNXUZ84xKYI6I\nZowlMEdEdMskTuptF9o4vERENCQZc0Q0o9f5v2+cnATmiGhGaswRER3TSo05gTkimjGWwBwR0S29\nRvozJDBHRDNSyoiI6JiUMiIiOqaV7nJtFGQiIhqSjDkimpF+zBERHTNrLIE5IqJTUmOOiIhpkYw5\nIpqRGnNERMfkApOIiI7JBSYRER3Tysm/BOaIaEZKGRERHZNSRkREx6SUERHRMa10l2tjKyIiGpKM\nOSKakZN/EREdM6uRUkYCc0Q0o5VeGW0cXiIiGpKMOSKakRpzRETHtFLKSGCOiGbkApOIiI5JxhwR\n0TGpMUdEdEwy5oiIjkmNOSKiY1rJmHOBSURExyRjjohm5ORfRETHTFUpQ1IPuAg4ENgInGx7/cD4\n+cB5dfDbwAnA64ATgXFgpzrv04F9gOuBdXX6i21fvbX1JzBHRDOm8Eb5rwRm2z5Y0gLg/Ppa3yXA\nq22vl3QysLftDwMfBpB0IbDU9gZJc4HzbL9/0tsxVVsREdGQRcAKANurgXn9EZL2A+4DzpB0E7CH\n7XUD4+cBz7N9WX1pLnCspJslLZW086iVJzBHRDPGepN/jLAb8MDA8CZJ/Xi5J7AQuABYDCyWdPjA\ntGcD5w4MrwbOsn0YsB54x8jtGNm8iIjtRK/Xm/RjhA3ArgPDY7Y31+f3AffYXmd7EyWzngcgaXdg\nP9s3D8x7je3b6/NlwJxRK09gjohmjPV6k36MsAo4BkDSQcCdA+PWA7tI2qcOHwJ8oz4/FLhxaFkr\nankD4CXAmlErz8m/iGjGFHaXWwYcKWlVHV4i6XhgZ9tLJZ0EXCUJ4Bbby+t0ogTuQW8GLpL0EPBd\n4JRRK09gjogYYnscOHXo5XUD428CFkww3/smeO3rlJOJk5bAHBHNyJ+xRkR0TCv3ykhgjohmNBKX\n0ysjIqJrkjFHRDN2iFKGpC8Cs4de7gHjtg+etlZFRPwcdpQb5b8VuBT4DWDT9DcnIuLnt0Pc9tP2\nakl/CRxge9kT1KaIiJ/LrEncBGN7MLLGbPu9T0RDIiKiyMm/iGjGDnHyLyJie7KjnPyLiNhuJGOO\niOiYRuJyAnNEtGOH6C4XEbE9SSkjIqJjGonLCcwR0Y5WMubcXS4iomOSMUdEM9KPOSKiY9IrIyKi\nY1q5iVFqzBERHZOMOSKakVJGRETHNFLJSGCOiHYkY46I6JhG4nJO/kVEdE0y5ohoxqxeG7lmAnNE\nNKOVUkYCc0Q0IzcxioiIaZGMOSKake5yEREd00hcTmCOiHYkY46I6Jhckh0R0THJmCMiOqaRuJzA\nHBHtaKUfcwJzRDQjpYyIiEZJ6gEXAQcCG4GTba8fGD8fOK8Ofhs4AXgdcCIwDuxU53068MvAFcBm\nYK3t00etP1f+RUQzer3JP0Z4JTDb9sHA2cD5Q+MvAU60fShwI7C37Q/bfrHtI4A1wO/b3lDnPcf2\nYcCYpFeMWnkCc0Q0Y2ysN+nHCIuAFQC2VwPz+iMk7QfcB5wh6SZgD9vrBsbPA55n+7L60lzbK+vz\n5cDikdsxuc2NiOi+sV5v0o8RdgMeGBjeJKkfL/cEFgIXUILsYkmHD0x7NnDuFpb7Q2D3kdsxaoKI\niB3QBmDXgeEx25vr8/uAe2yvs72JklnPA5C0O7Cf7ZsH5t088HxX4P5RK09gjohmTGGNeRVwDICk\ng4A7B8atB3aRtE8dPgT4Rn3erzkPul3SofX50cBKRkivjIhoxhR2l1sGHClpVR1eIul4YGfbSyWd\nBFwlCeAW28vrdKIE7kFnApdKejJwF/CJUStPYI6IZkxVXLY9Dpw69PK6gfE3AQsmmO99E7x2N3D4\n41l/AnNENCMXmEREdEwjcTmBOSLakXtlRER0TCNxOYE5ItrRSo05/ZgjIjomGXNENKORhDmBOSLa\nMYmbE20XEpgjohmpMUdExLRIxhwRzWgkYU5gjoh2tFLKSGCOiGY0EpenNzDP2/9V07n42E4dP+fw\nmW5CNCqXZEdEdEwjcTmBOSLakRpzRETHNBKXE5gjoh29XPkXEdEtrWTMufIvIqJjkjFHRDNy8i8i\nomNyd7mIiI5pJGFOjTkiomuSMUdEOxpJmROYI6IZOfkXEdExjcTlBOaIaEeu/IuI6JhkzBERHZMa\nc0RExzQSlxOYI6IdrWTMucAkIqJjkjFHRDMaSZgTmCOiHb1ZbUTmBOaIaEZqzBERMS2SMUdEMxpJ\nmBOYI6IdrZQyEpgjohlTFZcl9YCLgAOBjcDJttcPjJ8PnFcHvw2cYPunkt4KvJwSWy+0faWkOcD1\nwLo6/cW2r97a+hOYI6IdU5cxvxKYbftgSQuA8+trfZcAr7a9XtLJwN6S9gIW1nl2Bs6q084FzrP9\n/smuPCf/IqIZvbHepB8jLAJWANheDczrj5C0H3AfcIakm4A9bK8DjgLWSroGuK4+oATmYyXdLGlp\nDdpblcAcEc3o9Sb/GGE34IGB4U2S+vFyT2AhcAGwGFgs6cX19bnAbwKnAh+t068GzrJ9GLAeeMeo\nlaeUERHNmMKTfxuAXQeGx2xvrs/vA+6pWTKSVlAy6u8Dd9neBKyTtFHSnsA1tvtBfhkloG9VMuaI\naMYUZsyrgGMAJB0E3Dkwbj2wi6R96vAhwNo6z6/XeZ4BPIUSxFdI6pdCXgKsGbXyZMwREY+1DDhS\n0qo6vETS8cDOtpdKOgm4ShLALbaXA0g6RNJtQA84zfa4pDcDF0l6CPgucMqolScwR0Q7pqiUYXuc\nUicetG5g/E3Aggnme+sEr32dcjJx0hKYI6IZ+c+/iIiOaSUw5+RfRETHJGOOiGY0cquMBOaIaEcr\npYwE5ohoRu4uFxHRNW3E5Zz8i4jommTMEdGMsbE2cs0E5ohoRxtxOYE5ItrRysm/Ro4vERHtSMYc\nEc1oJWNOYI6IdrQRlxOYI6IdufIvIqJrUsqIiOiWRuJyAnNEtCMn/yIiuqaRGvOk+zFLetp0NiQi\nYlv1er1JP7psixmzpP2GXrpS0gkAttdNMEtEREyBrZUyPg88CHyH0jtQwIeAceCI6W9aRMTjsyN0\nl5sHfBC42PbnJH3R9oufoHZFRDxurQTmLdaYbX8PeA1wrKRznrgmRUT8nHq9yT86bKsn/2xvsv0W\nSjkjNzyKiE5r/uTfINtXAFdMa0siIgJIP+aIaEm3E+FJS2COiGa0cvIvgTkimtFr5D//2tiKiIiG\nJGOOiHaklBER0S1d7wY3WQnMEdGONuJyAnNEtKOVjDkn/yIiOiYZc0Q0ozerjVwzgTki2tFIKSOB\nOSKakRpzRERMi2TMEdGOXGASEdEtU1XKkNQDLgIOBDYCJ9tePzB+PnBeHfw2cILtn0p6K/BySmy9\n0PaVkp5NuW3yZmCt7dNHrT+ljIhox9T9g8krgdm2DwbOBs4fGn8JcKLtQ4Ebgb0lHQYsrPO8GNin\nTns+cI7tw4AxSa8YtfIE5ohoRm+sN+nHCIuAFQC2V1P+AxUASfsB9wFnSLoJ2MP2OuAoYK2ka4Dr\n6gNgru2V9flyYPGolaeUERHtmLpeGbsBDwwMb5I0ZnszsCewEDgNWA9cL2lNff3XgJdRsuXrgOfy\n6AvFfwjsPmrlCcwR0Ywp7C63Adh1YLgflKFky/fULBlJKygZ9feBu2xvAtZJ+rGkXwYeHljOrsD9\no1aeUkZEtGPqasyrgGMAJB0E3Dkwbj2wi6R+DfkQYG2d59frPM8AdqYE69slHVqnPRpYyQjJmCOi\nGVP411LLgCMlrarDSyQdD+xse6mkk4CrJAHcYns5gKRDJN1GKV+cZntc0pnApZKeDNwFfGLkdoyP\nj0/VhjzGAc88bPoWHtut4+ccPtNNiA46+9pztzmq/r87bpt0zHnqnBd2ttNzMuaIaEcjl2QnMEdE\nM1r5M9YE5ohoRyOXZLdxeImIaEgy5ohoRq/XRq6ZwBwR7cjJv4iIbmnlRvkJzBHRjkZO/iUwR0Qz\nkjFHRHRNAnNERMekV0ZERLdM4U2MZlQbh5eIiIYkY46IdqTGHBHRLb2xWTPdhCmRwBwRzUiNOSIi\npkUy5ohoR2rMERHdkiv/IiK6JheYRER0TCMn/xKYI6IZKWVERHRNShkREd2SjDkiomsayZjb2IqI\niIYkY46IZrRySXYCc0S0IzXmiIhuaeXucr3x8fGZbkNERAzIyb+IiI5JYI6I6JgE5oiIjklgjojo\nmATmiIiOSWCOiOiY9GN+AkhaAzxQB++1fdJMtidmhqQx4FJAwGbgd4GfAFfU4bW2T5+xBkZnJDBP\nM0mzAWwfMdNtiRl3HDBue5Gkw4B3AT3gHNsrJV0s6RW2r53ZZsZMSylj+h0I7Czps5I+L2nBTDco\nZkYNuKfUwWcCPwBeYHtlfW05sHgm2hbdksA8/R4E3mv7KOBU4CP1J23sgGxvlnQ5cAHwUUrG3PdD\nYPcZaVh0SgLE9FsHfATA9t3AfcBeM9qimFG2lwD7AUuBnQZG7QrcPyONik5JYJ5+S4DzACQ9g/Ll\n+78z2qKYEZJ+W9LZdXAj8DDwlVpvBjgaWDnhzLFDyU2MppmkJwF/ATwLGAf+wPatM9qomBGSdqL0\nwHg65cT7u4FvUjLnJwN3AW+ynS/lDi6BOSKiY1LKiIjomATmiIiOSWCOiOiYBOaIiI5JYI6I6JgE\n5oiIjklgjojomATmiIiO+f9qtR60nYgySgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x119669160>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot scoring function data\n",
    "ma = np.matrix([[meanv[0], meanv[1]], [meanv[2], meanv[3]]], dtype=np.float64)\n",
    "scores = pd.DataFrame(ma,columns=np.unique(estiv),index=np.unique(deptv))\n",
    "ax = sns.heatmap(scores)\n",
    "ax.set_title('parameters evaluation 3 depth vs estimators')\n",
    "ax.set_title\n",
    "fig = ax.get_figure()\n",
    "fig.savefig(\"par_eval_3_depth8_esti10_100.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "##### generate y_predict\n",
    "y_predict = call.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "##### first evaluation confusion matrix\n",
    "cm1 = confusion_matrix(y_test,y_predict)\n",
    "cm1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "ax = sns.heatmap(cm1)\n",
    "ax.set_title('confusion matrix 1')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}