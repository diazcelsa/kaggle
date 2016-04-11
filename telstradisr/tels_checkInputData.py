{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# first view into the data"
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
    "import warnings; warnings.simplefilter('ignore')\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from pandas import Series, DataFrame\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# insert data into a dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
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
       "      <th>id</th>\n",
       "      <th>location</th>\n",
       "      <th>fault_severity</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>14121</td>\n",
       "      <td>location 118</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9320</td>\n",
       "      <td>location 91</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>14394</td>\n",
       "      <td>location 152</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8218</td>\n",
       "      <td>location 931</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>14804</td>\n",
       "      <td>location 120</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      id      location  fault_severity\n",
       "0  14121  location 118               1\n",
       "1   9320   location 91               0\n",
       "2  14394  location 152               1\n",
       "3   8218  location 931               1\n",
       "4  14804  location 120               0"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# obtain integers out of strings with redundant data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
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
   "execution_count": 17,
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
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>14121</td>\n",
       "      <td>118</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>94</th>\n",
       "      <td>6821</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>103</th>\n",
       "      <td>12008</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>228</th>\n",
       "      <td>18441</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>897</th>\n",
       "      <td>9479</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1018</th>\n",
       "      <td>2627</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1201</th>\n",
       "      <td>3072</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1434</th>\n",
       "      <td>8714</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1678</th>\n",
       "      <td>14167</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1910</th>\n",
       "      <td>8676</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1930</th>\n",
       "      <td>14538</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2110</th>\n",
       "      <td>1805</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2571</th>\n",
       "      <td>15576</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2835</th>\n",
       "      <td>6206</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2894</th>\n",
       "      <td>11697</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3731</th>\n",
       "      <td>2364</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4089</th>\n",
       "      <td>17252</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4301</th>\n",
       "      <td>16023</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4662</th>\n",
       "      <td>8002</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4767</th>\n",
       "      <td>1014</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4835</th>\n",
       "      <td>474</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5106</th>\n",
       "      <td>727</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5120</th>\n",
       "      <td>4384</td>\n",
       "      <td>118</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5504</th>\n",
       "      <td>731</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5742</th>\n",
       "      <td>14559</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5966</th>\n",
       "      <td>10964</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6098</th>\n",
       "      <td>17307</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6230</th>\n",
       "      <td>107</td>\n",
       "      <td>118</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6242</th>\n",
       "      <td>15725</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6584</th>\n",
       "      <td>7495</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6712</th>\n",
       "      <td>9463</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6971</th>\n",
       "      <td>18104</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6982</th>\n",
       "      <td>2387</td>\n",
       "      <td>118</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         id  location  fault_severity\n",
       "0     14121       118               1\n",
       "94     6821       118               0\n",
       "103   12008       118               0\n",
       "228   18441       118               0\n",
       "897    9479       118               0\n",
       "1018   2627       118               0\n",
       "1201   3072       118               0\n",
       "1434   8714       118               0\n",
       "1678  14167       118               0\n",
       "1910   8676       118               0\n",
       "1930  14538       118               0\n",
       "2110   1805       118               0\n",
       "2571  15576       118               0\n",
       "2835   6206       118               0\n",
       "2894  11697       118               0\n",
       "3731   2364       118               0\n",
       "4089  17252       118               0\n",
       "4301  16023       118               0\n",
       "4662   8002       118               0\n",
       "4767   1014       118               0\n",
       "4835    474       118               0\n",
       "5106    727       118               0\n",
       "5120   4384       118               1\n",
       "5504    731       118               0\n",
       "5742  14559       118               0\n",
       "5966  10964       118               0\n",
       "6098  17307       118               0\n",
       "6230    107       118               1\n",
       "6242  15725       118               0\n",
       "6584   7495       118               0\n",
       "6712   9463       118               0\n",
       "6971  18104       118               0\n",
       "6982   2387       118               0"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.loc[train['location'] == 118]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# count events by type"
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
       "<seaborn.axisgrid.FacetGrid at 0x7f3f911c4be0>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nJKnw/wNotSpkI8FrGAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f3f911c4c88>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('event_type',data=event,kind='count',size=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# locate train data by severity type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f3f91031f98>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f3f91031fd0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('fault_severity',data=train,kind='count',size=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f3f9104fb70>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nNLt8RnNHv6Zp5XXZSy6SVOg/+t4Xr9OL9xUuA0wvPadT37yuptpnl1bLxjqaXVpVXtJwhUs9s0ur\nmrvvI9Kr/qxs+cLymhYf/4ym+7p1ziv/TBdccEHptlOnZvSDx27WJfv6fPc5OTmlbz9wjf7jK/68\nbLvNBkE8dWpG9lP/rIsvvqX0mEnSXHFwSu++gmhWn9NyZk0PnTwgSdqR6FZfA5fRanVmdmXT0Ly6\nsKKewR0Vb0+lJvSxxz6j//XS15WWpdNprS6cltSl6Xz0k88Gtba0rBt/8FjFc2N1YVHv/uKt6hkc\n1Lte9d833d/qwoLe/cXb1TM4oBv+8C1l5+Fm21WbZ62e/9BZdEb1CSepCy64QNu2naNrv3mvticG\ntLowr/OGyy+prCzM6v/c97Be/5IX+u7r1KkZ3f7kM+pOFGpcXVrQvZkFDVw0UNpeku7NLGrvJsPo\nuEHr4otf4Lvc77nw60V69NGjGh8f08UXv6DUJ7Rnz/ml29PpdKnXc8+e8zes4x7Pu81m3Drcbb2h\n0a1vdPR46Xk3e8paZWrhpNaml3WxCpdfT2VmtLuvtsto05lp6XNTG54nP+l0WrPLcxo5co6m/m9p\nLTOjqeNJDe6VdMdMoH3MZBrre5tdXtQ5Ty1oyng+Ty6c0vZ0WmcXTml++nm5F5hnTy8qN7Uk7bu4\n7mOm02mdXZhT9/hY4Nd6UB37DZLkPzjkSH+/z5rl3HAUxHDfeRruK+83Guor7xnZ3df8D99KPUkD\nfd0a7vfv4RnYWb0/ZrDOQQuHB9uj92lnX7d29HXHctDIIHoG/Pvc2lHPcPVB5noGB2qatLZnsF+9\nw9EPtieV9yNtT1TvnzpvuHo/SbexvRuWvHp8lqF97R2obcDJfQPrExUPFwd73DcQPIA2andfbf11\nQ32bfyZHpaMDEgAAQD3a9hKbe3119+6Ruib3K137LD4C2WxWqdSELrzwouLthb6ewu95jY4e1yWX\n/GhpPb9rrEEmvHT7gtwxUtzfu7q61Mj3GO7YSLt3b/wK171tZGSkpsfJ3e788/d6lmU1OVlYtm3b\nORvWn52d1WB3Xo6T1NraWW3bti2UPhlvLeZxN9suih6jIGMguRPUVu69KTye3v+MyVe4P/ni47PZ\n8dzHsKtr8/V9a9l0vWCvxUrrVevDcft51vt6Cj0VhcsnXbr00hdt2MYdw+iSS350w74mJiaUy+U2\nneg6lar9/PH2ArnHMo/j9kpls7mKf6Yd5PEM+pgjGt4xjur9rBofH9PevRdUHF8rl8v5Dith7iOb\nzRUnll3QoGr7psftb6pWR7XjT09PaaD4uhgK+buZnOc9Ydu2baHtt20D0uxyRvre0zr1kz+u7HJG\n+t6/Vl3/5MKCzvWEl9nl08pNTunkBeeru9jnMvalzyj9n15Zun3P4weVKl6Lf+Lmj+jnfr/Q4+N8\n8ZPqfumvbzhGKjWh733xeg319Wgm85x29/XoVGZFuz1/6j13ek3zRz+r5/q2a+dP/baWT69p7uht\nkqQXVegHCmJheU35pz8n/fjv+N6mpz8n6Xf0zKM365ILqh/n1PyKVqantLi8psXvf7bUHzA7v6Lv\nfe97Gvv+bfq5V7x1Q5/C0vKaMlN3S/v69MXb367M8pr6+rr1opf8bt33y7WcOat/e+ZW7dlz/qZ9\nTJnFVSUGepRZLPRuffnpD2hHf+VLaZnFVfUVLwsuLxbGgVqaW9Gjjx4tO9bSXPA+qsnJKX31qZvU\nU3zun1so1NLrufy4urSmj9//QWUl7bh449fMa0sr+nrmfiUuGvIse07X3HeDpC4lLtpVerNdOjGl\nWzNj6k5U7ltaWzqtO5YeKj2ftz51UImLgn31vra0rHuWp9T3I/vW61/IbFhvdX5Jt03Y2t5X+Rxb\nXViSJH1h6Smf2xb1zjtv1msv/YkNty0547rOKYw5c93R+0rLrzt6vyTp/Zev92OtLiyqd3iXnnji\ncV139EG95RcuM/bl6KPfe1rqkvovukhveMlP+taaSk3or2+6UYmLLtZ0V1fF+2RaW1rUwUxG2xMJ\nrS0t6uP33le8PLa+j5WFOb3j5tuKv3WVLp9Nd53x7GdBt6fGlbioPOB5rS7N60tLwWrz+3B2x0Dy\n6+Xx/lypfyjIMczlfuv49RV5+5ZqdejQvXryye9qz57zS+NMXXDBBaX7E3SflcaG8jO7NKOu4r+7\n+wv/sTqzNKXZg6fUJWnu9KwWHp7T0I7yy7+nMjNSXlLee4YUxkSayUxr5HO7lf6ltFL/MqrtP1t4\nbZ1cnFR3uvB+derUjKYOnyhdUitsu/4fNFOZGemOGXX/4vmaXZ5XdmpBMzsrBxy/XqTZ5QXteWRM\n35GU/uZ3dO5PX1hx+w3bnl7U/MPP6vm+hHJTS5rpCzbUxkxmQSObXJYu1JbRnse+K+fnfyHUPqS2\nvsTmnaQ2yIS1pqG+8g+Tvcb4KnsH1z+0Rjx9SXsHK7/5D/dtPmbN0M7t2u0Z22eob7uGQhgvp9rY\nSEPFPqTBGo8z5NOHNFjlOH3FvqbBRLcSO7eHOoFtos5G6h2JaPqMegI81tsT29WdqLxed2Lj+dSd\n6K2wfPOepB7PuVstTPnZHnCMsWrhyKt70P9NsqfC8sJtA2X/uj9XGxupZ9C/x6E7kVB3IrFpr1I9\n4yZJ0nbPdtsT/b5jH3UnBtSdGNzQW2Sus5newXj0W8Hfbs9YRWY4CsLtQxrpq9yjNryz+jdC3t6k\nIBPR+tYxWAh9tfYZFY5ZeB02q+doXxPGFmvrgAQAANAMbXuJTVrvIzKzsHu9c1fx2r93mZndvT0q\nu4rrDHr2s779+nrDPv0etY535Pbr1PK9Ri6X11yxr8Hbu+S9faFCn0jO00Ni/jw5eVLPP/+8ZmdP\nVby9q6urtPwczzaSSttVO24uwDbmcc1eG79am8UdJ8l7nT3vc3y3J+js2bM6efJkad1q9QXpFwpa\no19fULV+obzPeRO0v6jWdQvr+/ff+K3nvn7c/p1qvTX5XK7sfuQ9r1XvOtPTUxvW9duXexx33dHR\nH0qSpqbWl3t7ivwf9+rHCaLavmt93GvtE0Ft3D7DbfL/htH9fLpA9bdOePc1PT2l/ny3Zmdn1a/t\npTGH+vPbisu2la3rv4/G3jfX68gVj1l5nSGtPwaNjH2fq/C6MpfnPD19IyM/28AR13Xl8/lQdhQG\ny7KulvTzknKS/ty27W9VWvevf/3yvJTX3HJGw30JFS7gSu5F3LnlZQ337dDc8rKkwhQkhXV3umsV\nDyPNF9cZ3Hme5pdPey695TW3fFpSTsN952kuc7q4r17NLT+nob4elS4cK6e55ec03NetfPF35XPF\nf7PKK1dYs8vdc17zp89qsG+7vM+AuzdJUlfZvVK+q9BPNFC8PLewvKa81i+b5bukxeU19fd1l++n\n+NPi8lkNFI+3ePqs+ouXwxZPny3WLCWK+146vabEzvWfpbz6+7pL+19cXivbe2G7vMyzKS8pc/qs\n+nZuV+b02VIteZVPIeJu5/YtSdJy8f719XUrX3wsTmfWSuvu9Fy+y3fJ99h+y7zrex9rc7szy2vK\nSTqveJy8pDOZs6Wfz0tsV76rsCxvbJvvWr/ElvMcq3hm6LnMWW3v367iGaJs1znq0jlSfpsKX+ye\nUyzK/ZLX/b2reBIV/reWWVVP6dKad/lz6k54L4l1lf5dyzynQs/LDs/6p4315TlO+fZnl08bl9u8\n62nD8rOZwutre6KvynrS2cxpdScKHyZrxZ/XMsulZea2a8X9upfAzN8L9yuj7kTC57bi/Sv8X2k9\n737cfRSWd2kts1S63OX92etsJlO8r/2ePXgfQ+99cG8x79ei0a/kPkfe5V2evZg/dZX2/RuXMg1F\nMzmPntRg31DhkS9+lhZ+Lvw7n5nXUPH20nLPm1KXCr+7t7m3e/7zRe671NyZeQ3vGCz+O1D4nDs9\nr+Gdg4V/+wbX1z09r+G+IZnvgu76ZfvuMt8lvcfe+O/86QUN7ezX/JlFDe3Y+Bro6soX1xkorr+o\noZ3F112Xuc8gx8xrfrnwuhoqu3xfvrxLec2dzmi4r0//+NU76/+vFO99iUtAsizrlyS9zbbt/25Z\n1r+T9Gnbtl9WaX03IK0rD0gbPx7zZcu8Aam0LJ+rsE1xeemxKvye3/Cx512+eUByA1AtAcnv2XJj\nyfq+zf3mjfU3/usNDhu3KC7pqr5tpYBUtj+f++y7nvlzhQC02e2NBCSp+Iz6PCbeIFT2+KnsjCns\nwycg5YrLGw1I0jnGB6T5P/MOmrdVCjdSpYC0UfWAtHGZ33reIFFp3SD78i7fZH1PQKq8XrX7X+09\n2Pus1BaQ/I9tHnfzgNQ7uEu/uiuUzwlUsPD0iiT3I2djQHLfBMIISOW/e1dyD2KsW/Fz0NhXjQHJ\n/53Vc1izDu/6dQakjbdvXLereF9G+gf0tts+HcqJH6cepFdJ+pIk2bb9jKRBy7Ia/24SAACgRnEK\nSHsleQdyOFVcBgAA0FJxbtKu+hXZ3HJGFb+m6/L7em7jhSdzncIlNpWtU7pcVnaJzb1Ml1P5fr0X\nV7y3Zdcvg5X+V/57eT3F/+9ziU3G+nnvEuMSm7lu2e9d5n7yFZb7Hsl/nYr1bdxDtS9Ofev1OfZm\nl9hk3F62rc/lQr/1y8+ILt9Lht7Lad5Lb2W/e/ZVuhjb5Z4dUk5dkrZJel7ll9jWL6WVDlh2ic28\nTOP9b55Kl2X8Lp2ZW1W6tBT0EpPnOHlzmflz0Etnfvv3+z3g/jZcYqt8+c//d78LZNVu33xflW/b\nrDbz0l3xvNhFD1IzLSzPS1q/xGZe9dpwiU3Fd5G8cdnNXde8EuV7ycxYueolto0Fbbh8V7Zfn+P6\n7KPaNl2+2292mcznGPWsU/FyYX3iFJDSKv/G6AJJJyusqxse+CYX1wEAQFPE6RLbNyW9VpIsy/q/\nJH4CPNEAAATuSURBVKVs2z4dbUkAAGAris1fsUmSZVn/R9JlKlx1+FPbtqvPHwIAANAEsQpIAAAA\ncRCnS2wAAACxQEACAAAwEJAAAAAMcfoz/8Asy/o5SY+oTesHAACRyBb/vdG27TdVW7HtvkGyLGuH\npNsjLIGudgAA2sdZFYLRqqS54rJbN9uo7QKSpBVJ/17SIRXu7PNaH1b0bHHZZiEmSMhxBzn2Dtu5\nIv8hbTcbFLrqIM8Vbl/x2W+t+662Xq54DKz/F0WY2jVIt6LuRh/vs8V/m1Wr+9oPYz/m76sh7LeW\nY7bLvlux/2Zr9/o7xZqkMyp/PhYlHVNhuoJ+rU9sUFXbBSTbtnO2ba9I+mmpND+DO2r7GUnbFWwO\nhM149+3+3htgf0HmKwiix2e/jezbXO8cVb4/W822JuyzXUd6b0XdjT7e24v/NqtW97Ufxn7M33tC\n2G8tx2yXfbdi/83W7vV3im4VPtu8z8eQpBep0JbTo0KIetFmO2q7gCRJlmVdIWlBhf+S9H6z0lfn\nLqtNXxaVzb6p2kxc7gfaB+cMgHbmfvNrZpu8Cp+pOa1/w/SOzXbWlgFJ0utUmKtth8qDjXdWz1pU\nn70zHmr98Irr/WgmPuAbsxXPGQCdw3wPy3mWb9N6Ptgt6WLLsl5SbWftGpDuV6H2MyrvFajn/ngn\nXJfq6+lptOcp6CW/MD7AGuljijs+4AvqeT5rPe875ZwB0Dm8GWBV6728blB6TtI9KvQkjdm2/YNq\nO2u7P5O3LOulkv63pPNC2qVfn0Ct2zXS89TqD/WweqQQX/U8n7We95wzAOLM2+/n9i2eJ+m/qdCe\n8/bNdsBcbAAAAIZ2vcQGAADQNAQkAAAAAwEJAADAQEACAAAwEJAAAAAMBCQAAAADAQlAZCzLusyy\nrIdC3N9/sSxrsPjzbZZl7Qtr3wC2lrYbKBJAxwlzMLa3qjBr94Jt278b4n4BbDEEJACRsyzrxyT9\nswrfam+T9De2bT9sWdaIpP2SBiQ9L+lPbdv+vmVZ75X0ahWmGkqpMD/jlZJeLukzlmW9QdK/SHqV\nJEfSNZL+owpTDhyybfvdlmVdpsJouhOS/oMKk1j+qm3bK6251wDijEtsAKLWJelaSdfZtv3Lkt4i\n6ebibX8v6Wu2bb9c0rslXWFZ1jmSTkt6uW3bvyRpSNKv2Lb9z5ImJf1ucY4l95up35Z0iW3bvyjp\nMkmvsSzr5cXbfl7S223bfpkK4elXmnxfAbQJAhKAOPhPkg5Kkm3bT0tKWJa1S9JLJT1QXP6Qbdt/\nY9t2ToUwc9iyrAck/ZQKs3O7zPniXirp3uI+cpIekvRzxdt+YNv2bPHnMUnDod8zAG2JgAQgDsw+\npC4VQlBexvuUZVkvk/SHkl5t2/YrJB3ZZJ9++3aXPe9zGwAQkADEwqOSflWSLMv6GUmztm3PS3rE\ns/zllmXdJOl8SY5t2yuWZb1A0i9ofebunNZn7u7y7Ps/F/dxrgqX2R5t9h0C0N4ISACilpf0/0q6\n0rKs+yV9RNIVxdveJemXLct6UNL7JX1Q0jclDViWdUTSO1XoTXqHZVkvknSPpLsty/oFrX9LdIek\nHxbXPyzpLtu2j1aoAwAkSV35PO8JAAAAXnyDBAAAYCAgAQAAGAhIAAAABgISAACAgYAEAABgICAB\nAAAYCEgAAAAGAhIAAIDh/wcZ8JAwyzqqfwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f3f9104f5f8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('location',data=test,kind='count',size=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f3f8f462748>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nNDn5x7ojeaveuu/PJeWuhWsfvVZ/+KI/WHd/x8bGNDY2pivv+rz6zhtcie2Kw5/VF97yr+uuX6vF\n+bRuST2s3bvPXn9hw/j4hP7+hk+rMxhQZ75/0eJ8Wtc9/eDKua0msVlIzqk7Ur6vyJEjh/W+W29S\nV7hvpeN3taanp/S5h+7Xu/7Lr0mSPvPgfWs6YJ+endHjj/94zbrVPpwXkglJHfrG/Jx+Kf6czj//\nhVqcn9PnHnxUk5MT+urjT6kzGFZXOKK3vuSCNesvzid1KDVftm9SpWN4JplLrCY7pDPJWcnVCXth\nfla3p6TuMp2nx8cn9O0nRtUT7F95feNdj6lD0tnnWSvTbj/6rPrCu0u2Uc74+ITuOTqin39xlxIz\nZ9SjlKTqRzOYn59Rel7qD+WGFzl2bE4jP1tUf1/xvkSPP6XYSFZSrkN/Ym7KdQSKVXM+zSRsfHxC\nD9/+uMKBfnU81SH94pTmRkc0863d0uuK15k9OSM90SG9fGp1/bkTSo31VvWsi8dH9ZNbf6xIb1hn\nvSLX/3AqNb2mH1KtplIzGuwrfX+VSjqnUrMb2l49pqenNPXAY+oPBNXxxLwm8s+jqQceVX9gtS/m\n1PycOpTVWfl7YnDwlxuy/Zb4BEla/2fEzfmRwNqOrHvynT73hNfv0DXYF9BgX29++eK2IoH1OzAP\nVBgCZDOSo0YKG8lMvR2s1+ucHayzQ3SwQqftajpn9wQ7iyrcdgZLx9HrSojWS45MXfl922kMU9JV\n47A0nUaRgJcdtjtD1VfrmM5yJUcFXR50pi7Xwbu6dcMl/240s2N2V/51Ljnq184Kw5M0suO223od\ntLuCxTF3ByPaaXRsDtQ4pEhBcIOdw8PGdsN9peMwk6ZGC+c7Jg/0re7P7nDpjtW7Ahvb50hPvwY3\n2Ear6g/knkPuTtuFaV5rmQQJAABgs5AgAQAAGEiQAAAADC2TIK2tXiiu8Cj85Lh7CAv3T5YXqgwK\nbbirEEZGjmtxcVGOE9PExLirzWy++iBbU7VRcdzV9bjPZLKuWHOVcqvbX+/n1rNrtlHYbqGSa2lp\nSaOjoxoZGdHo6OjKUCaV2ioXuzvWtfOq+wl7dxur5yu37aWlpZXYN/pz/9kSVYflKrTcsWZLHNN6\ntl04f2YcpeKqFHup4zo2Nqasqwqv2phqqYxqZNVloYKrcC0WV0Vl5DgxRaM/K7qHs67KoMIQGiMj\n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\n1GKAJEmS1PI//xtMLuCbrzMAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f3f910ad3c8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('location',data=train,kind='count',size=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# group train data by severity and location"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "train0 = train.groupby('fault_severity').get_group(0)\n",
    "train1 = train.groupby('fault_severity').get_group(1)\n",
    "train2 = train.groupby('fault_severity').get_group(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f3f903c5978>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nUC9XP0GyLGuLpH+QdH/J7KskXWvb9iWSQpLe62YMAAAAjXL7K7Y5Sb8lqfRXAd8s6WD+8UFJb3U5\nBgAAgIa4miDZtp2xbXvBmL215Cu1CUk73IwBAACgUe3+I+3Vl8tg3WlkrBnGpalPOp3W8PCJjhuP\nyc3rVzjm1e6HPtZ6pdcOaJd2lPknLcvqtm17XtIuSVVHlOvp2ayNGzcsmRcMetXf72tqUNGoV5Lk\n92/RRH5aa3/RqFfx+BYFg7lt4/HFbfz+3Px4fItiFdoobD9ZsrwQR2kbtY61sI0b56WSwcFBxeOT\nxfNU6fiqxVVYbi4bHBzU9Te8W5/8m+9qz549VbcZHBzU1V9/tz7z8eXrFpZLKrusHis9t6XXUar/\nWjZDuZgHBwf1iQPX6eNve9eSvubzbarZ3tat3dKscutOLH1tJJMzNbedG44pmZxRMPhvl/Rxv3+L\nUqkZfeTAbbrrw59c8TUyFdpPJCb0ybtul//ii/L7uUt3ffgjxf2U9iXzepnn7k/3fkN3fviviq/1\n0uWV+nE9MZbbX6V1Gm2/cG8yj69c7AUr6ae5/pDMT1PFcyQtvSeW7r/c/goxS6eVSs3ozrueUm+g\nf1nMtZTr/9WudTw+qWjUW4yr2n5K2y49nnpiK7et37+loXtjNaXvXaUxlc4vt7ye/daa3+x7W7Vr\n2Mr3uXYkSA9KukzS/vy06ohyicTpZfNmZlKanEw2NaiZmZQkKR4/tWRabX8zMynF46eWbVt4XFhe\nqY1yywttlbZR61gL27hxXqrts9Y5qhVXYXm57QK93VXbLMyfmUnJ37up4X3Ua6XntvQ6SvVfy2Yo\nF/PMTErdAa+SybklfS2ZnKvZ3uzsvCQV1y297rW2L93W7OOFdjYFe5t6bkrPfXd+PKZ4/JQ2BYPL\nzomkZa+7wrLS9TYH+5asU6mdlcRotldpnUbbL+1ztWIvWEk/Ne+Z5j2sXGzl9ld6T4nHTymQT47M\nmGup1P+l8te69B5eWKeethuNrdy28fippt23Ss9daUzm9al2javdk6vNb/a9rdo1rGdfzUqgXE2Q\nLMv6vyXdJKlf0quWZX1A0m9KutmyrPdLOiHpZjdjAAAAaJSrCZJt209J+oUyiy51c78AAACr0e4/\n0gYAAOg4JEgAAACGNZkgUXaLtaATSpUXXyvNK+/PZNIaHa1afFpTNpPR6Ohoxw07cDZxY+iHTCbT\n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\n1o3n24inFp9LuYxzJrGgHZLis2eU9XiK08Rsrv3kbG6/ifzUJymanM/FlphXj6REvv1Evt1k6dQj\n9UmKJReK022SkrNnJI+n2H7K2HZglxRPzGvHLimeWNDOXYvrFKazhWkyPz+Zi0M7pER8XtqRm1dY\nVrruqWRuv7P586gBKZmYl7ZLp/Jxz5Zsk9Xi/FOFbfql2fhCbhpbkLZJp5O545pLnlFWHp3OTwvP\n1Sedii9IfdLp2IIUlOaTZ5TxeLSQPKPMOfmpx6OFxBllNa+FxKuSzpF6pfn4aalXWkjOSZ5ztJCc\nlzye3FTn5Od7dCY5l3vVBKX5+CkpKC0kT+fXPS0p/1yeZVMFpflYasl0ITkrjzw6kzyVm6ZOyePx\n6ExqVh6PpD5pIZ7MT1NSn/LLPDqTSsnjUcm00Fbudac+aSGWyG2TTOWXzcqjc/JTjxaSqXyMsyp0\nqvlYPLdtMpk/nsI0ZTxPlmyTkPp25bfdWdJuybrZXBu549qu+VisZHpesb3FaaJkKqmvL79uUAux\nqNTXqzPJRP64ksXpOfKUzM9N1efLb+PTQjImKVtsdyEZz08L+4vnj2un5mMzUt8OzcVmpL6B/LbS\nQjImjzyaz687n4zn97NN87HpfKwzUl9Q88moJGk+GZPH49F8Kip5PJpPxSTPOVKfV6fjU1LfVp2K\nTUnBrZpLRrVB0ulkVBvk0enkjM6RdCo/zT33SMHNSsYmpeBmzeaXzeaXpZJReeTRbCoqj+ccpVJR\nnePxSH3dSsQntauvW/H4pHb0dSuZ3yaZnJFHWjYN9nUpHptU77YuxeKT8vd3KZFflkgZ26Rm5PFk\ntXV7t6KJSW0e6FY0PqkdA92KJ2dy98rCtkYb8fx0265uzSSn1KduxVK544jPLp0W5hemPdqk6cSU\nfCXTaH5ZdDaaew+azbUfm52RJytt0mZNJSfVrc2aTE2qS5s1MzsjKbs4PRWVPFnNnIpJHulc+TSZ\nnNK56slP/ZqZjea3iUoeaWY2109Kp7n3vG2aSk5ro/oVnY3L45Gis7F8bLl1FqdxeSRt1A5NJWe0\nUTs1lZrRRu1S9FR+2+I0Js85+W08WW2QT1OpqDaoJz/1K5pK5t+HC9NEyTSrDerTVDKWn8a1Qf3G\nNtkybeTuNRu0U5OJhJYPOrJynmw2W3utJrIs61OSRm3bvjH/PCTpDbZtz7Y0EAAAgAra8RXb/ZLe\n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\nRK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f3f906b60b8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('location',data=train0,kind='count',size=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f3f8f06ea58>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nnjbWAgAAsCY0+h0kAECB53lKpcblee3/dXCd7Gs17DrXS91rFePXfQQkAGiS66b19tFRuW66I329\nY/RrHelrNVw3rX/adnexTtdN6zPbdq75utcq101r96fuYvy6iIAEACvQlxjqWF8nJ5Id62s1YkNb\nyl7Hh0a6VMnxYSTG+HUTAQkAAMBCQAIAALAQkAAAACwEJAAAAAsBCQAAwEJAAgAAsBCQAAAALAQk\nAAAACwEJAADAQkACAACwEJAAAAAsBCQAAAALAQkAAMBCQAIAALAQkACsC57nKZUal+d53S4FbdKq\nfcyx0jjGqjoCEoB1wXXTOmf0SrluutuloE1cN63rRneteh+7blpfue4ejpUGuG5a9156I2MVgoAE\nYN3oSwx2uwS0WSKxpSXtDA+2pp0TwUjslG6XsCYRkAAAACwEJAAAAAsBCQAAwEJAAgAAsBCQAAAA\nLAQkAAAACwEJAADAQkACAACwEJAAAAAsBCQAAAALAQkAAMBCQAIAALAQkAAAACwEJAAAAAsBCQAA\nwEJAAoB1zvM8TUzsk+dlu9o/ynmep1RqXJ7ndbsUrAABCQDWOddN6x9uulOZzKNd6/+6HT9oaNlU\navyECVOum9adl+2U66a7XQpWgIAEAMeBvvhQV/vvjye72v9adWpspNslYIUISAAAABYCEgAAgIWA\nBAAAYCEgAQAAWAhIAAAAFgISAACAhYAEAABgISABAABYCEgAAAAWAhIAAICFgAQAAGAhIAEAAFgI\nSAAAABYCEgAAgIWABACr4HmeUqlxeZ7X7VJW7XjalvVoPY//eq69GgISgLpSqXGlUuPdLmNNct20\n3j66Ta6b7nYpq+a6af3ttjuOi21Zj1w3re9/6u51Of6um9a9//zFdVl7NQQkAFilvsRQt0tomU2J\nkW6XcEIbia3f8R+JJbtdQksRkAAAACwEJAAAAAsBCQAAwEJAAgAAsBCQAAAALAQkAAAACwEJAADA\nQkACAACwEJAAAAAsBCQAAAALAQkAAMCysd0dOI7zXEk7JF1qjLnKcZynSrpB+XD2S0lnGWOOtbsO\nAACARrX1DpLjOJskXSLpzsDkiyVdYYw5Q1JK0pvbWQMAAECz2v0R21FJfyTpYGDaSyTdVnh+m6RX\ntLkGAACAprQ1IBljssaYZWvy5sBHaockbWlnDQCOb57nKZUal+d53S6lY07EbcaJZS0c423/DlId\n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\nwcewbfBrta8ffpthY2O3HTa91vFea7yDdfn7Lqyvauv4/LGvtp9rtecf7/nzOls47zYHzuesdR5l\nK+bZ1zj/mM9moxXLZrOxkP5iFcs95SmnhdSarWg3bH611/WWrzXd3u7g8WUfRwPZ/prrN8vub6Xb\nUKv9RsYpUaOO1dRSa342m9XExD694AW/XWcrGhPJ5XItaahdvv71m3MTt9wqSZqcn1MyGi08DkgR\nScpJymlyfr6wRi7wWNq2LS//Qz35yU+WJB04cEAH7v5mcf5wtL+wfmGdXP4xGd0sKSspp1xxelaT\nC7+SlNVTX/nGsjYfv39UynmaXDyqof4e+cdFLpLT1MKyhgZ6lJNKzyPBCqWphSVtfcV7yto89NDV\nZcvkCq9yEWl6fkmD0V7lVHoeXGZmflk5Sb/5sr8ua9P90ec0vbCkeGH5mYWlspGTpHi0p/h6dn5J\nOUmxaK/mCs+f/5L3lrX5yL9+VjlJswvLihW2c67QbrSwnv8Yi/ZqtvCYU05z88uSpDPOeF9Zmz/6\n989KkubnlhSN9ZaPQ6R8Ty/MLak/1qvFuSVtjvUWl3vRc99V1uZ37rtU/dFeLcznl/fXz0lanF/S\n5sJ45iLS4txysY/N8R5lJb3mWaX2/Da/ev8/F9fJSeqL5fdtttButvDv8bllZSPSk2I9+ec6Sb3x\nk3XO1reV1Xjlg59Xb2yT8gf4SaUN1klamntcfbHNkiJamjui3sLzwsmgpbnH1Rvr11v/S/nx/tkH\nbiouE3zsjfdraXZREUk98QH5B+3y3KJ6YtHC8wX1xKJ666+fUdbmdY/cW5wvqbB8RMtz81Iuop54\nVMuz85KCz0v998Sjeuuvv6Cszavuv1u98WhIrSq+XpqdU288lt/e2bnCdkQL8+b1zhe+xGrz+4Xl\niyekludm1RuPa2nWf5yTJPXG44V258t6fccLf7eszc/df79644Nanp1RT3ywrL7l2Vn1xOM6Njtb\nmBcpTM8vuzw7Iymit7/wd8ra3PbI3kLfM+qNJ8r2q//fo7PTxVd98YSWZqfVFx/S0dlp9cWH9Ve/\nvqWszesfOail2Zmy+iKKqC8+nG9vbkp9saROUkTLc1PaFEvqJElLc1PaHB/W//wv0Yrj/ZYH9ioi\naSCe1K9mpxSNJ7U4O6mBeH7dxdlJReNJSfmjd2F2SrHYsCKSFuYO62UvOL2izXsfmFAslswvW1h3\nfnZSg/GkZmcnFZEUjyc1NzOZf5yd1GAsv9zc3KR+50WnlW13+j+OKqKcZucmlYgmNTM3qUQsWX40\n5VQ2PSJpZjb/WrmctvzXTeXX4oePaHphSkP9w4E9k388PDepoehw/nXOnzal4YH8socXpiRJT3/l\n08ranLjdVXIgqan5SSX7k4oULjiRnDS5eEjJgaTK38sKzyOlq2HPiwfL2lzeM6nJxcOSpORA8NjM\nrz+1cFjDA4lSe4W2gtN7XvTU8jZ/MFFWx9TitIb7B1X+rlGoPZIrzp9anJGU03D/oJ70wl8ra/PY\nAz/X1MKMhgeC7fi1zGh4IG5tu1+vNDU/q+FoXE96/nPL2nziwYclSe//8hfCU1mT1s0dJElKRmPK\nB5eo7B2TnyZVHEh+uLEMR/vLlk9G+1V8O8sF1w+pY+DkqvMkabi/p2Lu8EBp2tBAT+h6Q9He0OnV\nJKKl0BB87huMhvcjSYMDpeWDz33BMYv7oUEqhJrqYoFtiw30FkNitLBtMesx/7wUJsNEY/XHZaAQ\noPpjtevrL/TbH9Jmv7Vtm2OFbYnU2tvSpkJ48pcJvzeQD07ZiORJ6on3KFvlE+7eWF/VvvLBqfJ5\nadrm8PXi/Sq/pEeqzMvriQ2EPg+Tnx8JvC4FnJ64/by23oaWiYU+r7a+vUx+Wtx6DASosnXCD8ze\nQigqD0cqTIuHzusJrBOp0m6+7UTVeX3xodDX9vTyZYJhK1LWd19suPj85MDzTfHSc9vmeClo9BfC\njP8o5YNTUDQ+XDw5orHwdmOFsBMNrBsvPI8HpsWC0wptxmPl/QUNRvPzBqssk7Cm268rlh8YDr0Y\n+OGoYlph2eEq6w0PJEuP1vzkQO1aakkODKnae9hwcV5j08Pkw1H9+dVCVKnP8Hby4ahG+9Hq87fU\nOH+axXeQAAAALAQkAAAACwEJAADAQkACAACwEJAAAAAsBCQAAAALAQkAAMBCQAIAALAQkAAAACwE\nJAAAAEtX/tSI4ziXSnqB8n+V4W+MMT/qRh0AAABhOn4HyXGc35f0n40xL5L0VkmXd7oGAACAWrrx\nEdvLJd0iScaYn0uKO47TX3sVAACAzulGQBqRNBl4PVWYBgAAsCZ05TtIlki9BSbn5wKvcoE1c4Fp\nOWt+rmzZLVabU/OLZcvmisvnpJzdXk45ZQNt5iRl9VS7zcWjUs4rrpmNlFqvrKg0p9hTRBqy2jw8\nv2Rtea64bFib5e3nX59mTZtZWMovF6m1bq7qPLsPSZpdWA7pOxd4Ht5GWFu+hbml6jVEqq9b7Cvk\nyFqcD28zbDxzitTdbkk6Mpvfdr+m8qOkdBz4z3OSPElZnSRpQ0V7S3NHC8UH/uVKzyPFDQsuI+u5\n1ebsorWcypaNWK8r26psd3luIaQnfyBrrVvZf6nO+SrLVq+jVnv5NufKJ+QiipQtGlZr7cvS0uyM\ntURlG+WjW387lmanQ9qpXVukYnr5lS6szYgihUn+9JN0klT4F9FJhTn5n56jFXX+anYycORF7CO1\nYp6sadLpFW3OzU0qklNxjWpHan6ZsHnlV7nZ+UlJudLyIev5J3RZjYGTfItV5/TCVHG9ijpyubIa\n7fb9eQPaXNbm1MJksT5V1BrSUMj721M0WNbm5MJhayNLGxWx2wx2GvBk691tamHaKrJyHf91pKK9\n/POK9+CFGVUKe2cIe29XaJuT8/b1Y3UiuVytt6jWcxznI5IOGGOuLbxOSXquMeZXHS0EAACgim58\nxHanpD+VJMdxfltShnAEAADWko7fQZIkx3H+UdIZyn/S8C5jzH90vAgAAIAquhKQAAAA1jJ+kzYA\nAICFgAQAAGAhIAEAAFjWwu9BqslxnOdK+r7yvzBmoDA5p/xvY/iVpJNF0AMAAOH8L1tHJD0i6TFJ\ndxljPlFrpTX9JW3HcTZJ2i3pFEkJ5UNSr6QnlN/gJxUWXS687gus7oco/zc8nqTy3z4VKazXW1gm\nImmpsJz/+iTlQ+TRQl9+oPQK8/x1nqTykOb3rUBbdX8h5goE+/Fla9TyhNZBKG6hsPFpd9v29OCJ\neTzwFPw9go3xf2fmBq398bDPH18jx1JwmUaPj2Xlz8mIStcp35ykWJU2jknqser1Co/272f0+/QU\n9ptJ22Ml514j17O1oNltC1u+k/viRGP/XuJfKX8cLSn/m08fNMac0UhDa+3Asx2V9LvKJ74jkr6l\n/IbOSfpXSTOS7ii8vkulQVkurOtfcH5ZmH5YpWAUUf7C9LikfcpfcJYKj8uF555KB/eGQt8q1HKs\nMM8/yIO/QNlfz5/uh7RjyoeUbGCZnPI7cMpqx6vSlt+ePz5eod1j8n9Bc2nZJUnpwvb5y/tteiHt\nBVVLzp71+liV5RpN3k+ETMtKWlD59jTbrn2S+O1KpX33eOG1PR7B5ZertH3UWu5oYJ7/6Cl/bIb9\n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\nAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f3f8f06e278>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('location',data=train1,kind='count',size=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f3f8d0aeda0>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\npQ0t9HSvPqttbOhsecB1Ojpb6Nq4+qTHC6nTuXH+vrWXHrj+6axdRP3Ws2xzwe10rV+Wds4lLaUS\nbT0bmRoKcktAAAAgAElEQVQZXVT91lKJtp5NC1ivm7aeTUyNDNNa6qKtZ3Msd9PWs3lRbT9wmz2s\n6enlyMjeWN66ZNte3bUllg6e9Pzarl7W92zj4PDuJWtrqXSVtrKpp5/hkUG6O3vnystl04Zetnb1\nA3CU/WzaMP9b+M0dvVwW11uoLeu30N95/zq97Vvo79wGwBAH6W3fEsuHzmj7Z6K3o4f+0oV3e8Lv\nIEmSJCUMSJIkSQkDkiRJUsKAJEmSlDAgSZIkJQxIkiRJCQOSJElSwoAkSZKUMCBJkiQlDEiSJEkJ\nA5IkSVLCgCRJkpQ4L36s9oFUZmYYGtw193jLlq3s3r3rfuVKpQI00dx80Unl09U5Vf3lqrMUbVYq\nM/OPW6XCzp13n1E/a7c1MzPLYBz3wZrxv/+252+/tv5C+5m2cyqnqrPQNmvXW+rxPFX92bMcj9mZ\nmVPWr62z0Pl5qvqnG49THffavi20/umO9an3Z/4xW2j9xczjhVrMthZa51T7c6pjHfazbZF7srg2\n031YyJwM/exZsn4udE4v5Pq3qDZnKmd+XZmpsLfmOruJtQvqQ239oeS60DRPP9Pz4FQ/t7yQ43Ym\n41ncpanMzMzt5/2WLeL61dn5qDMZpnk9KALS0MQEs7d8GTo2sGt8jMHHX8PsLV8EYPDx/4PZWz5H\nb0cHX79nJz3r2+jtaOfr9wywcV0oD45PMPiEZ1D5yr+FOk94FpWbPxbKT/wFKjd/mC0d6/j6zt10\nr13Flo5L2T1+L4PX/Br3ffldYb1rns99X3p7KD/pJUx/8c1s6WjlGzvHKa1fxZaOVnaPTzH4pFdw\n75ffFOu8ggM3/2Vs55WMf+Uv2dTRyrfvGaezfVW1vL6Fno5W9o5PMfj4V7HvljfGfXsVQ1+N5ce9\nit1ffSMbN7Rw290TdLSvYuOGFvaNTbP1sX80/7gN7eGer/0fuje0csfdE7SvX0XXhlaGx6a44jF/\nzPf+688BeOiPvZrtsfywH3v1XP3RiWlGbv1zRna2cNfdk2SXrz9p29+5NdT5watfzXzGx48yOv56\nvj/QwujoND969f8+ZT//8xtvYENnC3fnk2zLHvgX6oeG9vCFb7+B9s4WBvJJLot1Dowf5d/H30D7\n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\nrkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f3f8dbd6e10>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('location',data=train2,kind='count',size=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# check type of features from log files"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f3f8cc86f98>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nx1j3uiTXDcu2J7l0eaYDAPBJ2gAAf4dAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQA\ngIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFA\nAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAY\nCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQA\ngIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFA\nAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgIFAAgAYCCQAgMGiAqmq\nrn6MZX+w5NMAABwE1uzpzqq6MMkbk3xnVf3hLnc9IclJyzkYAMBU9hhI3f3hqvpMkg8neccud21P\n8l+XcS4AgMnsMZCSpLvvSXJWVT05yfFJVs3vOjbJ5mWcDQBgEo8bSElSVe9LcmmSTfnbQNqR5JnL\nNBcAwGQWFUhJzkmyrrsfXM5hAAAOBosNpC8vZRzNT/7+10keSfL2JLcnuTazd9Xdl+Si7n5kvt5l\nSbYlubK7r6qqNUmuTnJqkkeTXNLdG5ZqNgCAxQbSf5+/i+2WzKIkSdLdb9/bDVbV8ZlF0f+c5Ogk\nlye5IMn7u/u6qvrJJJdW1bVJ3pbkufNt/klVXZfkvCQL3f26qnpxkvckec3ezgEAsDuLDaT7k/yn\nJdrmdyX5VHc/kOSBJG+oqjuTvGF+//VJfjTJnye5rbu3JklV3ZLkhUnOTXLNfN0bk1y1RHMBACRZ\nfCD9+BJu87QkR1XV72T2Trh3JTmyux+Z378xySmZfc7Spl2+b9O4vLt3VNX2qlrT3Y8GAGAJLDaQ\nHs3sXWs77UjytSQn7MM2V2X2cQH/e2axdFP+9p1xGW6P3/dYXC4FAFhSiwqk7v6bCKmqJ2R2mOt/\n2sdtfiXJrd29PcmdVbUlySNVdUR3P5RkfZJ7ktyb2R6jndYn+dx8+clJbp+fsJ3H23t03HFHZs2a\n1fs47sqwsLA2C1MPsQjHH78269YdPfUYwAqxsLA2ycp4A7Xnt0PLYvcg/Y3ufjjJ71fVj2Z2gvTe\n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\nGxbAoWrFBlJVXZHk+Um2J/mX3f2nE4/EEjrxKU/KU0/2uUEATGNFBlJV/aMk/0N3n1FVfz/JVUnO\n2JufsVIO4ziEA8DBaqW8liZ7/3q6IgMpyblJfjtJuvtLVXVsVa3t7q2L/QEbNtyZuz78Gzn1hHXL\nNuT+uuv+TcmFr3EIB4CD0oYNd2bD1bfm6cc/depR9ujuzfcmF+/dSecrNZBOTrLrIbWvzpf9xd78\nkFNPWJfTTzplKeeCQ96h/BsjsPeefvxTc/qJp049xpJbqYE0WrUv33TX/ZuWeo4lddf9m7I3f+Xu\nuf+BZZtlKdxz/wM5bpHrbvzqN5Z1lqWwNzMubDz4H89iZ9yw4c58/9XvyhNPOGaZJ9o/D97/1/ng\nxe9Y1G+M39h4cD8XJHs349c33reMkyyN2Yx/f1Hr3v+Vu5d3mCUwm/HvLWrde7568D+ee756d07O\n4va23L353mWeZv/dvfnenJbT9up7Vu3YsWN5pllGVfWOJPd295Xzr+9I8uzu/vq0kwEAh4KVeqmR\nG5K8Kkmq6n9Jco84AgCWyorcg5QkVfXuJGcm2ZbkTd19+8QjAQCHiBUbSAAAy2WlHmIDAFg2AgkA\nYCCQAAAGh8rnIE2uqp6d5LokV3T3L049z/6qqvcmeWGS1Une090fm3ikfVZVT0pydZKTkhyR5Ce6\n+xOTDrUEquqJSb6Y5PLu/tWp59lXVXVmko9m9lhWJflCd1827VT7p6ouTPKvkzyS5O3d/fsTj7TP\nqurSJBcl2ZHZ/5/ndPfB/SFYe1BVRyX51STHJXlCZv9+bph2qn1XVauS/FKS70zyUJI3dvefTzvV\n3htfQ6vq25Jcm9mOnPuSXNTdjxzImexBWgJVdWSSn87s4wdWvKo6K8l3dPcZSf5xkp+ddqL99k+S\n/El3n5Xk1UmumHacJfO2JPdPPcQS+Ux3n9PdZx8CcXR8krdndn3Ilyd5xbQT7Z/uvmr+/+WcJO9I\ncs3UM+2ni5N8af54LkjyvmnH2W+vSHJMd78gyQ9kBT6/7eY19PIk7+/uM5PckeTSAz2XQFoaDyZ5\nWZKvTD3IEvnDzJ44kuSvkhw5/y1lReru3+zun5p/+fQkfznlPEuhqipJJVnxe8LmVuzfr8fwXUk+\n1d0PdPdXuvuNUw+0hN6e5MenHmI/bUxywvz28UkO/o9R37NnJbktSbr7jiTPXIHP14/1GnpWkuvn\nt6/P7N/VAeUQ2xLo7u1JHp69Zq1888ez87ol35/k97p7xX8eRFX9X0nWZ/Zb/Ur3U0nelOSSqQdZ\nIt9eVb+d2QvW5d1949QD7YfTkhxVVb+T5Ngk7+ruT0870v6rqucmubu7N049y/7o7o9W1cVV9eUk\nT07yPVPPtJ++mOSyqnpfZrH0tCRPyQoKv928hh61yyG1jUkO+IVT7UFit6rqFZm9AL956lmWwnwX\n9CuSfHjqWfZHVV2U5LPdvfOCTivtt8XRl5O8s7v/aWaHP36lqlbyL2+rMgu9f5rZv58PTTvOkvn+\nzM7lW9Hm54f9ZXc/K7O9Er8w8Uj7ZX5+2+cz2/P/fZmdr7PSnxNGkzwegcRjqqqXJvmxJN/d3Vum\nnmd/VNVzquppSdLd/yXJmqp6ysRj7Y+XJbmgqj6X2YvW/1FV50w80z7r7nu7+6Pz23cm+f8y29O3\nUn0lya3dvWP+eLas8L9vO52V5Naph1gCL0jyB0nS3V9I8m0r8JDUN+nuf9fdL0zy75I8eaXv5Zvb\nUlVHzG+vT3LAr4grkJbeiv6HliRVdUyS9yZ5eXd/bep5lsCLkvxIklTVSZntuv3qtCPtu+5+TXc/\nr7v/tyQfTPLjK/kQTlW9dn4B6lTViUnWJbln2qn2yw1JzqmqVVV1Qlb437ckqapTkmzp7kennmUJ\n/EWS5ydJVZ2aZOtKPoWgqp5dVVfOv7wgyWcmHGcp3Zjk/Pnt85N88kAPsJJ3Yx80qup5mb1QrUvy\naFW9IcmZ3b0w7WT77NWZncT4m/PfrHYk+efd/d+nHWuf/VJmh23+MMkTk/zwxPPwzT6e5Neq6pbM\nfmn7oZX8Qtzd91bVbyX5o8z+7RwKh6hPyew8kEPBLye5qqo+k9nHmPzgtOPst9uTrK6qP0rycJLv\nnXievfYYr6FvTPLSJNfMX0/vygTvnnQtNgCAgUNsAAADgQQAMBBIAAADgQQAMBBIAAADgQQAMBBI\nAItUVSvuM2aAfSOQABbvXVXleRMOAz4oEjggqurMJG9L8o0k1yd5bpLTkxyd5Ne7+2eq6juSfCDJ\ng0mOTHJ5d//+/JN2fyqzTwrekeTN3f2lqrop80utzC8bcUt3P62qPpTkoSSV5MLMrnD+M/NlmzP7\nZPivV9VPJjkjyZMyuwDwv93D/O9M8vYkn03yZ0k2dvc75/f928wuUPtAkmdmdjX1k5Pc1N0/Ol9n\n0dsCpuc3IeBAek6Si5Ick+Se7j43s+tifW9V/YMkP5Dkt+fL/0lmlx5IZpcZuGy+/GeS/OJufv6u\nv/Ed2d2GdSNbAAACc0lEQVRnd/e9Sa5N8n3dfXZmgfOyqnpVkvXzdZ6f5FlV9bLdDT6PoR1Jzkny\nf84fx06vyexSCUnyHUlePn9cr6iq79zbbQHTcy024EDq7v6rqjo7yfqqOmu+/IjM9ib9VmbXXzo1\nySe6+1er6slJTuzuz8/X/UySX1/Etm5NkvkFY5/c3X82H+Dn5st/Icnzq+rTmV1k+pgkz1jEz13V\n3XdV1Zeq6tzMLqz7te7+clUlyafnFz99pKr+JMm3JzlzH7cFTEQgAQfSw/P/PpTZ4bPrxhXmh9nO\nTfL6qnpdkh/KLCp22nkB5eSb9xg9YTfb2pHH3lv+UJIPdPcVezH/rnN8IMklSe5I8iu7LP+W4faO\nfdwWMCGH2IAp3JLk1UlSVd9SVT9dVcdW1ZuTPK27P5Hk+5P8r93910nurap/OP/eFyf5o/ntv87s\n/KJkFlV/R3dvTvLVqnrOfHv/an618FuSnF9Vq+fL31ZVpz/O3NuTfOv89u9mdh7VeUk+uss6/6iq\nVlXVEUn+YZIvzLf1yr3cFjAhe5CAKfxCkm+vqlsz+0Xtd+eH3r6U5Ner6mtJVifZeSLz65NcUVWP\nJtmW2V6lJPn5JL9UVa9N8gd72N5FSX6uqh5O8ldJLururfOTv2+d/9zPJ7nzceb+ZJI/rarzuvv/\nrarfSXJ0dz+4yzp3ZBZMz0jya93dSXoftgVMyLvYAPZBVT0hsz1D/7y7vzRf9o4kq7v77ZMOB+w3\ne5AA5qrqtCQfyjef27TznKd/2d1fmK/33Zm9k+2XdsYRcGixBwkAYOAkbQCAgUACABgIJACAgUAC\nABgIJACAgUACABj8/yuAB6jAi/S0AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f3f8d9cbc88>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('resource_type',data=resource,kind='count',size=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f3f8c1b0ac8>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f3f8c1b0ba8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('severity_type',data=severity,kind='count',size=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# select only part of the date to plot (xlim)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f3f8c210b00>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nsV/xPkgAAB2BBADQEUgAAB2BBADQ2W9P0m6tvSnJDybZmeQXq+qvBx4JADhA7Jd7kFprP5zk31TV\nxiQvTvLWgUcCAA4g+2UgJTkjyUeSpKr+LsmRrbXVw44EABwo9tdAOibJ3KLrXx0vAwD4tu235yB1\nVjyQlW++be7+V9rHbr5tLnt7Q/wv3/a1qc2yeJt7ew/aW756x9RmWbzNvb2V2Za56c+0lG1u2zLd\nuZayvdu3bJ/CJA9se3fcunUKkzywbd6xZfrPB/e3zdu3/POUJln6Nr++5StTmqTf5ro93j53683T\nG+Y+29zzu0Nvnvvy9IZZtM3j9jJTMn5n6yn68tZ/zGPyiL2vc9stU5rm3u2dkO9e8vorFhYWJjjO\nZLTWXpNkc1VdNL5+U5LHVtXXh50MADgQ7K+H2K5I8uwkaa39hyRfEUcAwL6yX+5BSpLW2uuTnJJk\nPsnLquqGgUcCAA4Q+20gAQBMyv56iA0AYGIEEgBARyABAHQOlPdBmorW2mOTfCjJm6rq7UPPkySt\ntQuTPD7JyiRvqKoPDzzPIUnendGblzw8yW9U1ceGnGmX1tp3JPliktdV1R8MPMspSS4dz7MiyReq\n6twhZ0qS1trZSf5LkruSvLqqPj7wPOckeUGShYz+Pz2uqo4YeKbDkvxBkqOSPCyjn6crhpwpSVpr\nK5L8bpLvS/KNJC+tqv830Cz3ea5srX1nkvdm9Ev5LUleUFV3DTnTeNm5Sd6Y5Miqun2a8+xuptba\no5O8K8lDk3wzyfOrassymOs/Jrkwo+eFOzP6/t025EyLlp+Z5ONVtc93+NiDtESttUOT/FZGbzGw\nLLTWTk3yvePPpHtqkjcPO1GS5EeT/FVVnZrkOUneNOw49/GqJFP9S30/PlVVp1fVacskjtYkeXWS\njUl+JMkzhp0oqap3jf//nJ7kNUneM/RMSV6Y5O/GM52V5C3DjnOPZyQ5oqp+KMlLMtDfvT08V74u\nyduq6pQkNyU5Z+iZWms/mVHkTv8dMPcwU5JfT/J74+fPjyT55WUy1y9mFGunJ/lsRj9fQ8+U1trD\nk5yfZPMktiuQlu7OJE9LcuvQgyxyTUZP0EnyL0kOHf8WOZiq+pOq+s3x1eOSTPftW/egtdaStCTL\nYm/W2KDfq914YpIrq+r2qrq1ql469ECdV2f0D8jQtiQ5enx5Te77sUdDOinJ9UlSVTclOXGg54Pd\nPVeemuSy8eXLMvpZG3qmD1TVBVOeY7HdzfTzST44vjyX0c/XtH3LXFX1nKq6efzzdGySfxp6prFf\nTfK2jPa27XMCaYmqamdVTeSb8GCNZ9q1W/jFSf5XVS2L921orf1lkj/M6DeP5eA3k/xSlleUfE9r\n7SOttWtaa9P+B2N3TkhyWGvtT1trf9FaO33ogXZprX1/ki8PcbihV1WXJnl0a+3GJFdn9HO1HHwx\nyZmttYeMfyF4dJJHTnuIPTxXHrbokNqWJOuHnmmIQ2rd9nc7U1XtbK09JMnLkvzRcpgruedQ1t8l\nWVtVfzj0TK2178roCMqHMqHndYF0AGitPSPJi5K8fOhZdhnv5n9GkvcNPUtr7QVJ/qKqdn1I0nKI\npBuTXFBVP5bRIZvfb60NfU7giox+Y/2xjH6eLhl2nPt4cUbntg1ufJ7WP1bVSRntCfmfA4+UJBmf\nL/b5jPYs/3RG5/osh5/13nKcadkYx9F7k3yiqq4eep5dqurPq6olqdbaK4eeJ/f+0jsxAmk/N676\nVyZ5SlVN99NJdz/P48YnGqaq/jbJqtba1H+L7TwtyVmttc9k9A/trw29d6SqNo/3RKSqNiX554x2\nXQ/p1iSfrqqF8Uzbl8H3bpdTk3x66CHGfijJnydJVX0hyXcOfWh7l6r61ap6fEaHHh6xHPa4jW0f\nny+SjH7OJ3LOyIO0LPa6L3JJkqqq5XA4OUnSWnvmoqsfzOjvwGBaa49K8m+T/PH4eX19a22fx+TQ\nv7Hur5bFk2Fr7YiMXllwRlX969DzjD0hyfFJzmutrcto1/pXhxyoqp676/L4g47/oao+OeBIaa09\nL8lJVfXa1traJDMZ6GTRRa5Icsn4lZFrsgy+d0nSWlufZHtV3T30LGN/n+QHk3y4tXZ8kh3L4dD2\n+FU+r6iql2R0buKnhp3oPq5K8qyMDhk9K8nlA87SP3+v2M2yaVuR3LN38htV9bqB5+m9urX29+Nf\nCE5OUgPOsqKqNif5rl0LWmv/UFWn7esNCaQlaq2dnOTijP4hu7u19rNJTqmqbQOO9ZyMThb9k/Fv\nsAtJfrKqpn0C3WK/m9HhomuSfEdGJx3yrT6a5I9aa9dltCf354YOgKra3Fr7QEavUlnI8jlkuz6j\n81aWi3cmeVdr7VMZvb3Gzww7zj1uSLKytfbZjE5a/YkhhtjNc+VLk5yZ5D3j582bM+VXI+7h+fva\nJD+c0c/X51pr11TV1J6v9jDTyiR3jPeGLCT5v1U11b+He/j+/XSSd7TW7kpyR0ZvvTHkTP2/vxP5\nBcVnsQEbket2AAACBUlEQVQAdJyDBADQEUgAAB2BBADQEUgAAB2BBADQEUgAAB2BBBxwWmuXtNam\n+onxwIFFIAEAdLyTNrBfaK1dn+QXquqz4+tXJrksow9FfkhG70J8flV9etHXHJ/kuqp69Pj6a5Ks\nrKpXt9a2J/n1JE9P8rAk/yOjz+r7roze2fyq8ecKvj3JIUlWJ/lvVfWJqTxgYFD2IAH7iz/M6DPG\n0lqbSfLdGX0Q8dvHn8P08xl9CnpvTx8XcFiSvxp/uOvXkzytqp6W5Ddy70fkvCPJb1bVEzMKsYvH\nn7YOHOD8RQf2F+/PaG9Pkjw7yaVJfiDJlUlSVV9Mcnhrbc0DuM+/HP/3n5J8etHlR4wvn5bktePP\nxvrjJN9IsvbBPgBg/+EQG7BfqKpbW2ubWms/kNEHNZ+X5Ce71XZ9aPMuC7nvJ7U/LMn8out37+Hy\nrq/5RpIfH/hDqYEB2IME7E/el9Enix9VVX+T5LNJnpIkrbV/n+S2Lma+luSo1tp3tNZWZvTp7Q/E\ntUmeO77/R7bWfvvbfQDA/kEgAfuTDyf5iSR/NL7+C0le0lr7ZJK3JHn+ePlCklTVvyR5d5K/TvLB\nJJ9fdF/9nqbdOTfJj7fWrknyZ0mcoA0HiRULC3t6XgAAODjZgwQA0BFIAAAdgQQA0BFIAAAdgQQA\n0BFIAAAdgQQA0BFIAACd/w8jlhcOAXW4CgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f3f8c210470>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('volume',data=log[log['volume'] < 15],kind='count',size=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# extract from log num of features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(386,)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log['log_feature'].unique().shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f3f8c214898>"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjgAAAI5CAYAAACsKUBTAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAIABJREFUeJzs3Xd8HGeBN/DfFvXeLFtyLxmX2LGd5nSnkAIhCSR0wkHu\njuOF9+B9rwHX4OAFcke54zjgDkJCiBNIQnp3EjuJ7Ti2417HTW6yJKu37bvz/rGSrN2d3Z3ZnZnd\nZ/b3/Xz4EK1mpcejnWd+81SHoiggIiIishNnrgtAREREZDQGHCIiIrIdBhwiIiKyHQYcIiIish0G\nHCIiIrIdBhwiIiKyHbfZv0CSpH8DcDUAF4D7AWwD8Aii4aoDwL2yLAclSfoMgK8BCAP4tSzLD5pd\nNiIiIrInh5nr4EiStBrA38qy/CFJkuoB7ATwJoCXZFl+SpKk7wE4hWjg2QHgEgAhREPQNbIsD5hW\nOCIiIrIts7uo3gHwsbH/HgBQAeA6AM+PvfYCgA8AuBzAVlmWR2RZ9gHYCOAqk8tGRERENmVqF5Us\nyxEAnrEv/xTASwBukWU5OPbaOQDTADQD6J701u6x14mIiIh0M30MDgBIknQngPsA3Azg6KRvOZK8\nJdnrRERERGlZMcj4FgDfRLTlZliSpGFJkkpkWfYDaAXQDuAsYltsWgFsTvVzQ6Gw4na7zCo2ERER\n5V7GDR6mBhxJkqoB/BuAG2VZHhx7+Q0AdwN4bOz/XwWwFcADY8dHAFyJ6IyqpPr7Pam+TURERIJr\naqrK+L1mt+B8AkADgCckSXIAUAD8CYDfSJL0FwBOAnhYluWwJEnfALAW0YDzbVmWh00uGxEREdmU\nqdPEzdTdPSxmwYmIiEiTpqaqjLuouJIxERER2Q4DDhEREdkOAw4RERHZDgMOERER2Q4DDhEREdkO\nAw4RERHZDgMOERER2Q4DDhEREdkOAw4RERHZDgMOERER2Q4DDhEREdkOAw4RERHZDgMOERER2Q4D\nDhEREdkOAw4RERHZDgMOERER2Q4DDglvzVoZ992/DmvWyrkuChER5QkGHBKaLxDC+h3tAID1O9vh\nC4RyXCIiIsoHDDgktFBYgTL234oS/ZqIiIgBh4iIiGyHAYeIiIhshwGHiIiIbIcBh4iIiGyHAYeI\niIhshwGHiIiIbIcBh4iIiGyHAYeIiITAVctJDwYcIiLKe1y1nPRiwCEiorzHVctJLwYcIiIish0G\nHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHQYc\nIiIish0GHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwi\nIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHQYcIiIish0GHCIi\nIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHQYcIiIi\nsh0GHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKy\nHQYcIiIish0GHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrId\nBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHQYcIiIish0G\nHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHQYcIiIish0GHCIiIrIdBhwiIiKyHbfZ\nv0CSpGUAngbwE1mWfyFJ0kMALgbQM3bID2VZfkWSpM8A+BqAMIBfy7L8oNllIyIiInsyNeBIklQO\n4McA1sZ96xuyLL8cd9w/AbgEQAjANkmSnpZlecDM8hEREZE9md1F5QPwIQBdaY67HMBWWZZHZFn2\nAdgI4CqTy0ZEREQ2ZWoLjizLEQABSZLiv/W/JUn6a0SDz18CmAqge9L3uwFMM7NsREREZF+mj8FR\n8TsAvbIs75Ek6esAvg3g3bhjHOl+SF1dOdxulwnFI5GUjAZivm5oqER1RXGOSkNEZuG1TnpZHnBk\nWV4/6cvnAfwCwJMAPjzp9VYAm1P9nP5+j/GFI+GMeIMxX/f2jsDvKcpRaYjILLzWC1NTU1XG77V8\nmrgkSX+UJGnp2JfXAdgHYCuASyRJqpYkqRLAlQA2WF02IiIisgezZ1FdDuABAE0AQpIkfQnAtwA8\nJEnSMIARAF+QZdknSdI3EJ1tFQHwbVmWh80sGxEREdmX2YOMtwBYqvKtZ1SOfRrR9XKIiIiIssKV\njImIiMh2GHCIiIhIszVrZdx3/zqsWSvnuigpMeAQERGRJr5ACOt3tAMA1u9shy8QynGJkmPAISIi\n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\n/wfXUSxJczx1dQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f3f8c1f1a58>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('log_feature','volume',data=log,size=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
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
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>14121</td>\n",
       "      <td>118</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9320</td>\n",
       "      <td>91</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>14394</td>\n",
       "      <td>152</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8218</td>\n",
       "      <td>931</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>14804</td>\n",
       "      <td>120</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      id  location  fault_severity\n",
       "0  14121       118               1\n",
       "1   9320        91               0\n",
       "2  14394       152               1\n",
       "3   8218       931               1\n",
       "4  14804       120               0"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
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
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>11066</td>\n",
       "      <td>481</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      id  location\n",
       "0  11066       481"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.loc[test['id'] == 11066]"
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
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
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
       "      <th>predict_0</th>\n",
       "      <th>predict_1</th>\n",
       "      <th>predict_2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>11066</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      id  predict_0  predict_1  predict_2\n",
       "0  11066          0          1          0"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample.loc[sample['id'] == 11066]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
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
       "      <th>predict_0</th>\n",
       "      <th>predict_1</th>\n",
       "      <th>predict_2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>11066</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>18000</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>16964</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4795</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3392</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      id  predict_0  predict_1  predict_2\n",
       "0  11066          0          1          0\n",
       "1  18000          0          1          0\n",
       "2  16964          0          1          0\n",
       "3   4795          0          1          0\n",
       "4   3392          0          1          0"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
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
       "      <th>severity_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>8815</th>\n",
       "      <td>11066</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         id  severity_type\n",
       "8815  11066              2"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "severity.loc[severity['id'] == 11066]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
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
       "      <th>severity_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6597</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8011</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2597</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5022</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6852</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     id  severity_type\n",
       "0  6597              2\n",
       "1  8011              2\n",
       "2  2597              2\n",
       "3  5022              1\n",
       "4  6852              1"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "severity.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
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
       "      <th>log_feature</th>\n",
       "      <th>volume</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>29815</th>\n",
       "      <td>11066</td>\n",
       "      <td>230</td>\n",
       "      <td>24</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29816</th>\n",
       "      <td>11066</td>\n",
       "      <td>310</td>\n",
       "      <td>28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29817</th>\n",
       "      <td>11066</td>\n",
       "      <td>228</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29818</th>\n",
       "      <td>11066</td>\n",
       "      <td>308</td>\n",
       "      <td>26</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          id  log_feature  volume\n",
       "29815  11066          230      24\n",
       "29816  11066          310      28\n",
       "29817  11066          228      20\n",
       "29818  11066          308      26"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log.loc[log['id'] == 11066]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
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
       "      <th>log_feature</th>\n",
       "      <th>volume</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6597</td>\n",
       "      <td>68</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8011</td>\n",
       "      <td>68</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2597</td>\n",
       "      <td>68</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5022</td>\n",
       "      <td>172</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5022</td>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     id  log_feature  volume\n",
       "0  6597           68       6\n",
       "1  8011           68       7\n",
       "2  2597           68       1\n",
       "3  5022          172       2\n",
       "4  5022           56       1"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
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
       "      <th>resource_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>9541</th>\n",
       "      <td>11066</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         id  resource_type\n",
       "9541  11066              2"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "resource.loc[(resource['id'] == 11066) & (resource['resource_type'] == 2)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
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
       "      <th>resource_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6597</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8011</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2597</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5022</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6852</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     id  resource_type\n",
       "0  6597              8\n",
       "1  8011              8\n",
       "2  2597              8\n",
       "3  5022              8\n",
       "4  6852              8"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "resource.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
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
       "      <th>event_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>15453</th>\n",
       "      <td>11066</td>\n",
       "      <td>35</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          id  event_type\n",
       "15453  11066          35"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "event.loc[(event['id'] == 11066) & (event['event_type'] == 35)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
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
       "      <th>event_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6597</td>\n",
       "      <td>11</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8011</td>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2597</td>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5022</td>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5022</td>\n",
       "      <td>11</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     id  event_type\n",
       "0  6597          11\n",
       "1  8011          15\n",
       "2  2597          15\n",
       "3  5022          15\n",
       "4  5022          11"
      ]
     },
     "execution_count": 50,
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
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
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
   "version": "3.4.3+"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}