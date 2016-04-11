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
    "collapsed": true
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
    "            \n",
    "            # check if any value is greater than 1\n",
    "            #l1 = a[tables[i].columns[-1]]\n",
    "            #l3 = a[tables[i].columns[-2]]\n",
    "            #for i in range(len(l1)):\n",
    "            #    if l3.iloc[i] != 1:\n",
    "            #        print('more than once',l3.iloc[i])\n",
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
    "            # check if any value is greater than 1\n",
    "            #l1 = a[tables[i].columns[-1]]\n",
    "            #l2 = a[tables[i].columns[-3]]\n",
    "            #l3 = a[tables[i].columns[-2]]\n",
    "            #print(l1,l2,l3)\n",
    "            #for i in range(len(l1)):\n",
    "            #    if l1.iloc[i] != 1:\n",
    "            #        print('more than once',l2.iloc[i])\n",
    "            #for i in range(len(l3)):\n",
    "            #    if l3.iloc[i] != 1:\n",
    "            #        print('more than once',l2.iloc[i])\n",
    "        \n",
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
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>volume</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>id</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>{345: 1, 179: 1, 68: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>{312: 1, 313: 1, 315: 1, 235: 1, 233: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>{171: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>{370: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>{232: 1, 312: 1}</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                      volume\n",
       "id                                          \n",
       "1                    {345: 1, 179: 1, 68: 1}\n",
       "2   {312: 1, 313: 1, 315: 1, 235: 1, 233: 1}\n",
       "3                                   {171: 1}\n",
       "4                                   {370: 1}\n",
       "5                           {232: 1, 312: 1}"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>event_type</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>id</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>{11: 1, 13: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>{34: 1, 35: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>{11: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>{47: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>{34: 1, 35: 1}</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        event_type\n",
       "id                \n",
       "1   {11: 1, 13: 1}\n",
       "2   {34: 1, 35: 1}\n",
       "3          {11: 1}\n",
       "4          {47: 1}\n",
       "5   {34: 1, 35: 1}"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "event.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>location</th>\n",
       "      <th>fault_severity</th>\n",
       "      <th>event_type</th>\n",
       "      <th>severity_type</th>\n",
       "      <th>resource_type</th>\n",
       "      <th>volume</th>\n",
       "      <th>log_feature</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>14121</td>\n",
       "      <td>118</td>\n",
       "      <td>1</td>\n",
       "      <td>{34: 1, 35: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{232: 1, 312: 1}</td>\n",
       "      <td>{19: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9320</td>\n",
       "      <td>91</td>\n",
       "      <td>0</td>\n",
       "      <td>{34: 1, 35: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{315: 1, 235: 1}</td>\n",
       "      <td>{200: 1, 116: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>14394</td>\n",
       "      <td>152</td>\n",
       "      <td>1</td>\n",
       "      <td>{34: 1, 35: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{301: 1, 221: 1}</td>\n",
       "      <td>{1: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8218</td>\n",
       "      <td>931</td>\n",
       "      <td>1</td>\n",
       "      <td>{11: 1, 15: 1}</td>\n",
       "      <td>{1: 1}</td>\n",
       "      <td>{8: 1}</td>\n",
       "      <td>{80: 1, 82: 1, 203: 1}</td>\n",
       "      <td>{1: 1, 12: 1, 9: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>14804</td>\n",
       "      <td>120</td>\n",
       "      <td>0</td>\n",
       "      <td>{36: 1, 34: 1, 11: 1, 20: 1}</td>\n",
       "      <td>{1: 1}</td>\n",
       "      <td>{8: 1, 2: 1}</td>\n",
       "      <td>{160: 1, 181: 1, 227: 1, 117: 1, 134: 1, 232: ...</td>\n",
       "      <td>{1: 1, 2: 1}</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      id location  fault_severity                    event_type severity_type  \\\n",
       "0  14121      118               1                {34: 1, 35: 1}        {2: 1}   \n",
       "1   9320       91               0                {34: 1, 35: 1}        {2: 1}   \n",
       "2  14394      152               1                {34: 1, 35: 1}        {2: 1}   \n",
       "3   8218      931               1                {11: 1, 15: 1}        {1: 1}   \n",
       "4  14804      120               0  {36: 1, 34: 1, 11: 1, 20: 1}        {1: 1}   \n",
       "\n",
       "  resource_type                                             volume  \\\n",
       "0        {2: 1}                                   {232: 1, 312: 1}   \n",
       "1        {2: 1}                                   {315: 1, 235: 1}   \n",
       "2        {2: 1}                                   {301: 1, 221: 1}   \n",
       "3        {8: 1}                             {80: 1, 82: 1, 203: 1}   \n",
       "4  {8: 1, 2: 1}  {160: 1, 181: 1, 227: 1, 117: 1, 134: 1, 232: ...   \n",
       "\n",
       "           log_feature  \n",
       "0              {19: 1}  \n",
       "1     {200: 1, 116: 1}  \n",
       "2               {1: 1}  \n",
       "3  {1: 1, 12: 1, 9: 1}  \n",
       "4         {1: 1, 2: 1}  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# join tables with train data \n",
    "train = train.merge(event, left_on='id', right_index=True, how='left')\n",
    "train = train.merge(severity, left_on='id', right_index=True, how='left')\n",
    "train = train.merge(resource, left_on='id', right_index=True, how='left')\n",
    "train = train.merge(log1, left_on='id', right_index=True, how='left')\n",
    "train = train.merge(log2, left_on='id', right_index=True, how='left')\n",
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>location</th>\n",
       "      <th>event_type</th>\n",
       "      <th>severity_type</th>\n",
       "      <th>resource_type</th>\n",
       "      <th>volume</th>\n",
       "      <th>log_feature</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>11066</td>\n",
       "      <td>481</td>\n",
       "      <td>{34: 1, 35: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{308: 1, 228: 1, 310: 1, 230: 1}</td>\n",
       "      <td>{24: 1, 20: 1, 26: 1, 28: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>18000</td>\n",
       "      <td>962</td>\n",
       "      <td>{11: 1, 15: 1}</td>\n",
       "      <td>{1: 1}</td>\n",
       "      <td>{8: 1}</td>\n",
       "      <td>{82: 1, 203: 1}</td>\n",
       "      <td>{9: 1, 20: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>16964</td>\n",
       "      <td>491</td>\n",
       "      <td>{34: 1, 35: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{315: 1, 235: 1}</td>\n",
       "      <td>{10: 1, 11: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4795</td>\n",
       "      <td>532</td>\n",
       "      <td>{10: 1, 27: 1}</td>\n",
       "      <td>{5: 1}</td>\n",
       "      <td>{9: 1, 3: 1}</td>\n",
       "      <td>{240: 1, 37: 1, 38: 1}</td>\n",
       "      <td>{1: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3392</td>\n",
       "      <td>600</td>\n",
       "      <td>{15: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{8: 1}</td>\n",
       "      <td>{82: 1, 203: 1}</td>\n",
       "      <td>{2: 1, 6: 1}</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      id location      event_type severity_type resource_type  \\\n",
       "0  11066      481  {34: 1, 35: 1}        {2: 1}        {2: 1}   \n",
       "1  18000      962  {11: 1, 15: 1}        {1: 1}        {8: 1}   \n",
       "2  16964      491  {34: 1, 35: 1}        {2: 1}        {2: 1}   \n",
       "3   4795      532  {10: 1, 27: 1}        {5: 1}  {9: 1, 3: 1}   \n",
       "4   3392      600         {15: 1}        {2: 1}        {8: 1}   \n",
       "\n",
       "                             volume                   log_feature  \n",
       "0  {308: 1, 228: 1, 310: 1, 230: 1}  {24: 1, 20: 1, 26: 1, 28: 1}  \n",
       "1                   {82: 1, 203: 1}                 {9: 1, 20: 1}  \n",
       "2                  {315: 1, 235: 1}                {10: 1, 11: 1}  \n",
       "3            {240: 1, 37: 1, 38: 1}                        {1: 1}  \n",
       "4                   {82: 1, 203: 1}                  {2: 1, 6: 1}  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# join tables with test data\n",
    "test = test.merge(event, left_on='id', right_index=True, how='left')\n",
    "test = test.merge(severity, left_on='id', right_index=True, how='left')\n",
    "test = test.merge(resource, left_on='id', right_index=True, how='left')\n",
    "test = test.merge(log1, left_on='id', right_index=True, how='left')\n",
    "test = test.merge(log2, left_on='id', right_index=True, how='left')\n",
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
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
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "id               0\n",
       "location         0\n",
       "event_type       0\n",
       "severity_type    0\n",
       "resource_type    0\n",
       "volume           0\n",
       "log_feature      0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## check if null values\n",
    "tr_a.isnull().sum()\n",
    "te_a.isnull().sum()\n",
    "test.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "False False\n"
     ]
    }
   ],
   "source": [
    "## check if nan values\n",
    "nantr = ifNaN(x_train)\n",
    "nante = ifNaN(x_test)\n",
    "print(nantr,nante)"
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
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>event_type</th>\n",
       "      <th>severity_type</th>\n",
       "      <th>resource_type</th>\n",
       "      <th>volume</th>\n",
       "      <th>log_feature</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>642</th>\n",
       "      <td>{11: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{8: 1}</td>\n",
       "      <td>{160: 1, 44: 1}</td>\n",
       "      <td>{1: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6340</th>\n",
       "      <td>{34: 1, 35: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{232: 1, 312: 1}</td>\n",
       "      <td>{20: 1, 28: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>942</th>\n",
       "      <td>{11: 1, 15: 1}</td>\n",
       "      <td>{1: 1}</td>\n",
       "      <td>{8: 1}</td>\n",
       "      <td>{193: 1, 82: 1, 203: 1, 71: 1}</td>\n",
       "      <td>{1: 1, 2: 1, 12: 1, 20: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5296</th>\n",
       "      <td>{34: 1, 35: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{232: 1, 312: 1}</td>\n",
       "      <td>{1: 1, 2: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5720</th>\n",
       "      <td>{34: 1, 35: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{232: 1, 233: 1, 313: 1, 312: 1}</td>\n",
       "      <td>{1: 1, 116: 1, 65: 1}</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          event_type severity_type resource_type  \\\n",
       "642          {11: 1}        {2: 1}        {8: 1}   \n",
       "6340  {34: 1, 35: 1}        {2: 1}        {2: 1}   \n",
       "942   {11: 1, 15: 1}        {1: 1}        {8: 1}   \n",
       "5296  {34: 1, 35: 1}        {2: 1}        {2: 1}   \n",
       "5720  {34: 1, 35: 1}        {2: 1}        {2: 1}   \n",
       "\n",
       "                                volume                 log_feature  \n",
       "642                    {160: 1, 44: 1}                      {1: 1}  \n",
       "6340                  {232: 1, 312: 1}              {20: 1, 28: 1}  \n",
       "942     {193: 1, 82: 1, 203: 1, 71: 1}  {1: 1, 2: 1, 12: 1, 20: 1}  \n",
       "5296                  {232: 1, 312: 1}                {1: 1, 2: 1}  \n",
       "5720  {232: 1, 233: 1, 313: 1, 312: 1}       {1: 1, 116: 1, 65: 1}  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>event_type</th>\n",
       "      <th>severity_type</th>\n",
       "      <th>resource_type</th>\n",
       "      <th>volume</th>\n",
       "      <th>log_feature</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>4924</th>\n",
       "      <td>{11: 1}</td>\n",
       "      <td>{1: 1}</td>\n",
       "      <td>{8: 1}</td>\n",
       "      <td>{87: 1}</td>\n",
       "      <td>{3: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4335</th>\n",
       "      <td>{34: 1, 35: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{233: 1, 315: 1, 313: 1}</td>\n",
       "      <td>{1: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>823</th>\n",
       "      <td>{35: 1}</td>\n",
       "      <td>{4: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{313: 1, 315: 1}</td>\n",
       "      <td>{1: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5752</th>\n",
       "      <td>{11: 1}</td>\n",
       "      <td>{1: 1}</td>\n",
       "      <td>{8: 1}</td>\n",
       "      <td>{171: 1, 55: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2726</th>\n",
       "      <td>{34: 1, 35: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{2: 1}</td>\n",
       "      <td>{232: 1, 307: 1, 227: 1, 312: 1, 233: 1, 235: ...</td>\n",
       "      <td>{20: 1, 5: 1, 6: 1, 7: 1, 10: 1, 15: 1}</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          event_type severity_type resource_type  \\\n",
       "4924         {11: 1}        {1: 1}        {8: 1}   \n",
       "4335  {34: 1, 35: 1}        {2: 1}        {2: 1}   \n",
       "823          {35: 1}        {4: 1}        {2: 1}   \n",
       "5752         {11: 1}        {1: 1}        {8: 1}   \n",
       "2726  {34: 1, 35: 1}        {2: 1}        {2: 1}   \n",
       "\n",
       "                                                 volume  \\\n",
       "4924                                            {87: 1}   \n",
       "4335                           {233: 1, 315: 1, 313: 1}   \n",
       "823                                    {313: 1, 315: 1}   \n",
       "5752                                    {171: 1, 55: 1}   \n",
       "2726  {232: 1, 307: 1, 227: 1, 312: 1, 233: 1, 235: ...   \n",
       "\n",
       "                                  log_feature  \n",
       "4924                                   {3: 1}  \n",
       "4335                                   {1: 1}  \n",
       "823                                    {1: 1}  \n",
       "5752                                   {2: 1}  \n",
       "2726  {20: 1, 5: 1, 6: 1, 7: 1, 10: 1, 15: 1}  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5904, 5)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "pipeline done.\n",
      "DataSpliterTrans fit done.\n",
      "DataSpliterTrans transform done.\n",
      "DataSpliterTrans fit done.\n",
      "DataSpliterTrans transform done.\n",
      "DataSpliterTrans fit done.\n",
      "DataSpliterTrans transform done.\n",
      "DataSpliterTrans fit done.\n",
      "DataSpliterTrans transform done.\n",
      "DataSpliterTrans fit done.\n",
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
      "  args, varargs, kw, default = inspect.getargspec(init)\n"
     ]
    }
   ],
   "source": [
    "##### generate pipeline \n",
    "## shape 5904x615\n",
    "call = PipelineTelstra(RandomForestClassifier)\n",
    "call.set_params()\n",
    "call = call.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DataSpliterTrans transform done.\n",
      "DataSpliterTrans transform done.\n",
      "DataSpliterTrans transform done.\n",
      "DataSpliterTrans transform done.\n",
      "DataSpliterTrans transform done.\n"
     ]
    }
   ],
   "source": [
    "##### generate y_predict\n",
    "y_predict = call.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[819, 110,  25],\n",
       "       [176, 183,  30],\n",
       "       [ 29,  42,  63]])"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##### first evaluation confusion matrix\n",
    "cm1 = confusion_matrix(y_test,y_predict)\n",
    "cm1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.text.Text at 0x109f6fa20>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x10a6644e0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "ax = sns.heatmap(cm1)\n",
    "ax.set_title('confusion matrix 1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "## save confusion matrix 1\n",
    "fig = ax.get_figure()\n",
    "fig.savefig(\"cm1_preCV.png\")"
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
    "##### cross validation 3 folds\n",
    "kf = KFold(5904, n_folds=3)\n",
    "itr = defaultdict(list)\n",
    "ite = defaultdict(list)\n",
    "c = 0\n",
    "for trai, tes in kf:\n",
    "    print(\"%s %s\" % (trai, tes))\n",
    "    itr[c] = trai\n",
    "    ite[c] = tes\n",
    "    c += 1"
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
    "##### redefine my samples\n",
    "\n",
    "#### 2.1 samples\n",
    "x_train1 = x_train.iloc[itr[0],:]\n",
    "x_test1 = x_train.iloc[ite[0],:]\n",
    "y_train1 = tr_a.fault_severity.iloc[itr[0]]\n",
    "y_test1 = tr_a.fault_severity.iloc[ite[0]]"
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
    "#### 2.1 make prediction\n",
    "call = PipelineTelstra(RandomForestClassifier)\n",
    "call = call.fit(x_train1,y_train1)"
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
    "y_predict1 = call.predict(x_test1)"
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
    "#### 2.1 confusion matrix\n",
    "cm21 = confusion_matrix(y_test1,y_predict1)\n",
    "cm21"
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
    "ax = sns.heatmap(cm21)\n",
    "ax.set_title('confusion matrix 2.1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fig = ax.get_figure()\n",
    "fig.savefig(\"cm21_posCV.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#### 2.2 samples\n",
    "x_train2 = x_train.iloc[itr[1],:]\n",
    "x_test2 = x_train.iloc[ite[1],:]\n",
    "y_train2 = tr_a.fault_severity.iloc[itr[1]]\n",
    "y_test2 = tr_a.fault_severity.iloc[ite[1]]"
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
    "#### 2.2 make prediction\n",
    "call = PipelineTelstra(RandomForestClassifier)\n",
    "call = call.fit(x_train2,y_train2)"
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
    "y_predict2 = call.predict(x_test2)"
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
    "#### 2.2 confusion matrix\n",
    "cm22 = confusion_matrix(y_test2,y_predict2)\n",
    "cm22"
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
    "ax = sns.heatmap(cm22)\n",
    "ax.set_title('confusion matrix 2.2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fig = ax.get_figure()\n",
    "fig.savefig(\"cm22_posCV.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#### 2.3 samples\n",
    "x_train3 = x_train.iloc[itr[2],:]\n",
    "x_test3 = x_train.iloc[ite[2],:]\n",
    "y_train3 = tr_a.fault_severity.iloc[itr[2]]\n",
    "y_test3 = tr_a.fault_severity.iloc[ite[2]]"
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
    "#### 2.3 make prediction\n",
    "call = PipelineTelstra(RandomForestClassifier)\n",
    "call = call.fit(x_train3,y_train3)"
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
    "y_predict3 = call.predict(x_test3)"
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
    "#### 2.3 confusion matrix\n",
    "cm23 = confusion_matrix(y_test3,y_predict3)\n",
    "cm23"
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
    "ax = sns.heatmap(cm23)\n",
    "ax.set_title('confusion matrix 2.3')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fig = ax.get_figure()\n",
    "fig.savefig(\"cm23_posCV.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "##"
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