{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "##############\n",
    "############## EXPLORE DATA\n",
    "##############\n",
    "\n",
    "\n",
    "import warnings; warnings.simplefilter('ignore')\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from pandas import Series, DataFrame\n",
    "import seaborn as sns\n",
    "from sklearn import preprocessing\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.preprocessing import Imputer\n",
    "from scipy import stats"
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
    "train = pd.read_csv(\"../../../github_data/bnp_paribas_cardif_data/train.csv\")\n",
    "test = pd.read_csv(\"../../../github_data/bnp_paribas_cardif_data/test.csv\")\n",
    "sample = pd.read_csv(\"../../../github_data/bnp_paribas_cardif_data/sample_submission.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div style=\"max-height:1000px;max-width:1500px;overflow:auto;\">\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>PredictedProb</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>  0</td>\n",
       "      <td> 0.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>  1</td>\n",
       "      <td> 0.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>  2</td>\n",
       "      <td> 0.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>  7</td>\n",
       "      <td> 0.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td> 10</td>\n",
       "      <td> 0.5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ID  PredictedProb\n",
       "0   0            0.5\n",
       "1   1            0.5\n",
       "2   2            0.5\n",
       "3   7            0.5\n",
       "4  10            0.5"
      ]
     },
     "execution_count": 4,
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
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div style=\"max-height:1000px;max-width:1500px;overflow:auto;\">\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>target</th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "      <th>v3</th>\n",
       "      <th>v4</th>\n",
       "      <th>v5</th>\n",
       "      <th>v6</th>\n",
       "      <th>v7</th>\n",
       "      <th>v8</th>\n",
       "      <th>...</th>\n",
       "      <th>v122</th>\n",
       "      <th>v123</th>\n",
       "      <th>v124</th>\n",
       "      <th>v125</th>\n",
       "      <th>v126</th>\n",
       "      <th>v127</th>\n",
       "      <th>v128</th>\n",
       "      <th>v129</th>\n",
       "      <th>v130</th>\n",
       "      <th>v131</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td> 3</td>\n",
       "      <td> 1</td>\n",
       "      <td> 1.335739</td>\n",
       "      <td> 8.727474</td>\n",
       "      <td> C</td>\n",
       "      <td> 3.921026</td>\n",
       "      <td>  7.915266</td>\n",
       "      <td> 2.599278</td>\n",
       "      <td> 3.176895</td>\n",
       "      <td> 0.012941</td>\n",
       "      <td>...</td>\n",
       "      <td> 8.000000</td>\n",
       "      <td> 1.989780</td>\n",
       "      <td> 0.035754</td>\n",
       "      <td> AU</td>\n",
       "      <td> 1.804126</td>\n",
       "      <td> 3.113719</td>\n",
       "      <td> 2.024285</td>\n",
       "      <td> 0</td>\n",
       "      <td> 0.636365</td>\n",
       "      <td> 2.857144</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td> 4</td>\n",
       "      <td> 1</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td> C</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>  9.191265</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td> 2.301630</td>\n",
       "      <td>...</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td> 0.598896</td>\n",
       "      <td> AF</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td> 1.957825</td>\n",
       "      <td> 0</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td> 5</td>\n",
       "      <td> 1</td>\n",
       "      <td> 0.943877</td>\n",
       "      <td> 5.310079</td>\n",
       "      <td> C</td>\n",
       "      <td> 4.410969</td>\n",
       "      <td>  5.326159</td>\n",
       "      <td> 3.979592</td>\n",
       "      <td> 3.928571</td>\n",
       "      <td> 0.019645</td>\n",
       "      <td>...</td>\n",
       "      <td> 9.333333</td>\n",
       "      <td> 2.477596</td>\n",
       "      <td> 0.013452</td>\n",
       "      <td> AE</td>\n",
       "      <td> 1.773709</td>\n",
       "      <td> 3.922193</td>\n",
       "      <td> 1.120468</td>\n",
       "      <td> 2</td>\n",
       "      <td> 0.883118</td>\n",
       "      <td> 1.176472</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td> 6</td>\n",
       "      <td> 1</td>\n",
       "      <td> 0.797415</td>\n",
       "      <td> 8.304757</td>\n",
       "      <td> C</td>\n",
       "      <td> 4.225930</td>\n",
       "      <td> 11.627438</td>\n",
       "      <td> 2.097700</td>\n",
       "      <td> 1.987549</td>\n",
       "      <td> 0.171947</td>\n",
       "      <td>...</td>\n",
       "      <td> 7.018256</td>\n",
       "      <td> 1.812795</td>\n",
       "      <td> 0.002267</td>\n",
       "      <td> CJ</td>\n",
       "      <td> 1.415230</td>\n",
       "      <td> 2.954381</td>\n",
       "      <td> 1.990847</td>\n",
       "      <td> 1</td>\n",
       "      <td> 1.677108</td>\n",
       "      <td> 1.034483</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td> 8</td>\n",
       "      <td> 1</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td> C</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>       NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>  Z</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td> 0</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 133 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   ID  target        v1        v2 v3        v4         v5        v6        v7  \\\n",
       "0   3       1  1.335739  8.727474  C  3.921026   7.915266  2.599278  3.176895   \n",
       "1   4       1       NaN       NaN  C       NaN   9.191265       NaN       NaN   \n",
       "2   5       1  0.943877  5.310079  C  4.410969   5.326159  3.979592  3.928571   \n",
       "3   6       1  0.797415  8.304757  C  4.225930  11.627438  2.097700  1.987549   \n",
       "4   8       1       NaN       NaN  C       NaN        NaN       NaN       NaN   \n",
       "\n",
       "         v8    ...         v122      v123      v124  v125      v126      v127  \\\n",
       "0  0.012941    ...     8.000000  1.989780  0.035754    AU  1.804126  3.113719   \n",
       "1  2.301630    ...          NaN       NaN  0.598896    AF       NaN       NaN   \n",
       "2  0.019645    ...     9.333333  2.477596  0.013452    AE  1.773709  3.922193   \n",
       "3  0.171947    ...     7.018256  1.812795  0.002267    CJ  1.415230  2.954381   \n",
       "4       NaN    ...          NaN       NaN       NaN     Z       NaN       NaN   \n",
       "\n",
       "       v128  v129      v130      v131  \n",
       "0  2.024285     0  0.636365  2.857144  \n",
       "1  1.957825     0       NaN       NaN  \n",
       "2  1.120468     2  0.883118  1.176472  \n",
       "3  1.990847     1  1.677108  1.034483  \n",
       "4       NaN     0       NaN       NaN  \n",
       "\n",
       "[5 rows x 133 columns]"
      ]
     },
     "execution_count": 5,
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
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div style=\"max-height:1000px;max-width:1500px;overflow:auto;\">\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "      <th>v3</th>\n",
       "      <th>v4</th>\n",
       "      <th>v5</th>\n",
       "      <th>v6</th>\n",
       "      <th>v7</th>\n",
       "      <th>v8</th>\n",
       "      <th>v9</th>\n",
       "      <th>...</th>\n",
       "      <th>v122</th>\n",
       "      <th>v123</th>\n",
       "      <th>v124</th>\n",
       "      <th>v125</th>\n",
       "      <th>v126</th>\n",
       "      <th>v127</th>\n",
       "      <th>v128</th>\n",
       "      <th>v129</th>\n",
       "      <th>v130</th>\n",
       "      <th>v131</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>  0</td>\n",
       "      <td> 1.375465</td>\n",
       "      <td> 11.361141</td>\n",
       "      <td> C</td>\n",
       "      <td> 4.200778</td>\n",
       "      <td> 6.57700</td>\n",
       "      <td> 2.081784</td>\n",
       "      <td> 1.784386</td>\n",
       "      <td> 0.011094</td>\n",
       "      <td> 9.523810</td>\n",
       "      <td>...</td>\n",
       "      <td> 7.619048</td>\n",
       "      <td> 1.815241</td>\n",
       "      <td> 0.000000</td>\n",
       "      <td> AF</td>\n",
       "      <td> 1.292368</td>\n",
       "      <td> 3.903345</td>\n",
       "      <td> 1.485925</td>\n",
       "      <td> 0</td>\n",
       "      <td> 2.333334</td>\n",
       "      <td> 1.428572</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>  1</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>       NaN</td>\n",
       "      <td> C</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>     NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>  I</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "      <td> 0</td>\n",
       "      <td>      NaN</td>\n",
       "      <td>      NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>  2</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>  8.201529</td>\n",
       "      <td> C</td>\n",
       "      <td> 4.544371</td>\n",
       "      <td> 6.55010</td>\n",
       "      <td> 1.558442</td>\n",
       "      <td> 2.467532</td>\n",
       "      <td> 0.007164</td>\n",
       "      <td> 7.142858</td>\n",
       "      <td>...</td>\n",
       "      <td> 5.714286</td>\n",
       "      <td> 1.970928</td>\n",
       "      <td> 0.014123</td>\n",
       "      <td> AV</td>\n",
       "      <td> 1.128724</td>\n",
       "      <td> 5.844156</td>\n",
       "      <td> 1.475892</td>\n",
       "      <td> 0</td>\n",
       "      <td> 1.263157</td>\n",
       "      <td>-0.000001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>  7</td>\n",
       "      <td> 2.661870</td>\n",
       "      <td>  3.041241</td>\n",
       "      <td> C</td>\n",
       "      <td> 1.657216</td>\n",
       "      <td> 9.77308</td>\n",
       "      <td> 2.078337</td>\n",
       "      <td> 1.430855</td>\n",
       "      <td> 1.252157</td>\n",
       "      <td> 7.959596</td>\n",
       "      <td>...</td>\n",
       "      <td> 4.404040</td>\n",
       "      <td> 8.163614</td>\n",
       "      <td> 1.100329</td>\n",
       "      <td>  B</td>\n",
       "      <td> 1.988688</td>\n",
       "      <td> 1.558753</td>\n",
       "      <td> 2.448814</td>\n",
       "      <td> 0</td>\n",
       "      <td> 5.385474</td>\n",
       "      <td> 1.493777</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td> 10</td>\n",
       "      <td> 1.252822</td>\n",
       "      <td> 11.283352</td>\n",
       "      <td> C</td>\n",
       "      <td> 4.638388</td>\n",
       "      <td> 8.52051</td>\n",
       "      <td> 2.302484</td>\n",
       "      <td> 3.510159</td>\n",
       "      <td> 0.074263</td>\n",
       "      <td> 7.612904</td>\n",
       "      <td>...</td>\n",
       "      <td> 6.580644</td>\n",
       "      <td> 1.325654</td>\n",
       "      <td> 0.258459</td>\n",
       "      <td>  A</td>\n",
       "      <td> 1.863796</td>\n",
       "      <td> 2.666478</td>\n",
       "      <td> 2.374275</td>\n",
       "      <td> 0</td>\n",
       "      <td> 0.681672</td>\n",
       "      <td> 2.264151</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 132 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   ID        v1         v2 v3        v4       v5        v6        v7  \\\n",
       "0   0  1.375465  11.361141  C  4.200778  6.57700  2.081784  1.784386   \n",
       "1   1       NaN        NaN  C       NaN      NaN       NaN       NaN   \n",
       "2   2 -0.000000   8.201529  C  4.544371  6.55010  1.558442  2.467532   \n",
       "3   7  2.661870   3.041241  C  1.657216  9.77308  2.078337  1.430855   \n",
       "4  10  1.252822  11.283352  C  4.638388  8.52051  2.302484  3.510159   \n",
       "\n",
       "         v8        v9    ...         v122      v123      v124  v125      v126  \\\n",
       "0  0.011094  9.523810    ...     7.619048  1.815241  0.000000    AF  1.292368   \n",
       "1       NaN       NaN    ...          NaN       NaN       NaN     I       NaN   \n",
       "2  0.007164  7.142858    ...     5.714286  1.970928  0.014123    AV  1.128724   \n",
       "3  1.252157  7.959596    ...     4.404040  8.163614  1.100329     B  1.988688   \n",
       "4  0.074263  7.612904    ...     6.580644  1.325654  0.258459     A  1.863796   \n",
       "\n",
       "       v127      v128  v129      v130      v131  \n",
       "0  3.903345  1.485925     0  2.333334  1.428572  \n",
       "1       NaN       NaN     0       NaN       NaN  \n",
       "2  5.844156  1.475892     0  1.263157 -0.000001  \n",
       "3  1.558753  2.448814     0  5.385474  1.493777  \n",
       "4  2.666478  2.374275     0  0.681672  2.264151  \n",
       "\n",
       "[5 rows x 132 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
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
       "      <th>ID</th>\n",
       "      <th>PredictedProb</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>0.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7</td>\n",
       "      <td>0.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>10</td>\n",
       "      <td>0.5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ID  PredictedProb\n",
       "0   0            0.5\n",
       "1   1            0.5\n",
       "2   2            0.5\n",
       "3   7            0.5\n",
       "4  10            0.5"
      ]
     },
     "execution_count": 8,
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
       "      <th>ID</th>\n",
       "      <th>target</th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "      <th>v3</th>\n",
       "      <th>v4</th>\n",
       "      <th>v5</th>\n",
       "      <th>v6</th>\n",
       "      <th>v7</th>\n",
       "      <th>v8</th>\n",
       "      <th>...</th>\n",
       "      <th>v122</th>\n",
       "      <th>v123</th>\n",
       "      <th>v124</th>\n",
       "      <th>v125</th>\n",
       "      <th>v126</th>\n",
       "      <th>v127</th>\n",
       "      <th>v128</th>\n",
       "      <th>v129</th>\n",
       "      <th>v130</th>\n",
       "      <th>v131</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1.335739</td>\n",
       "      <td>8.727474</td>\n",
       "      <td>C</td>\n",
       "      <td>3.921026</td>\n",
       "      <td>7.915266</td>\n",
       "      <td>2.599278</td>\n",
       "      <td>3.176895</td>\n",
       "      <td>0.012941</td>\n",
       "      <td>...</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>1.989780</td>\n",
       "      <td>0.035754</td>\n",
       "      <td>AU</td>\n",
       "      <td>1.804126</td>\n",
       "      <td>3.113719</td>\n",
       "      <td>2.024285</td>\n",
       "      <td>0</td>\n",
       "      <td>0.636365</td>\n",
       "      <td>2.857144</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>C</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>9.191265</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.301630</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.598896</td>\n",
       "      <td>AF</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.957825</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>0.943877</td>\n",
       "      <td>5.310079</td>\n",
       "      <td>C</td>\n",
       "      <td>4.410969</td>\n",
       "      <td>5.326159</td>\n",
       "      <td>3.979592</td>\n",
       "      <td>3.928571</td>\n",
       "      <td>0.019645</td>\n",
       "      <td>...</td>\n",
       "      <td>9.333333</td>\n",
       "      <td>2.477596</td>\n",
       "      <td>0.013452</td>\n",
       "      <td>AE</td>\n",
       "      <td>1.773709</td>\n",
       "      <td>3.922193</td>\n",
       "      <td>1.120468</td>\n",
       "      <td>2</td>\n",
       "      <td>0.883118</td>\n",
       "      <td>1.176472</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6</td>\n",
       "      <td>1</td>\n",
       "      <td>0.797415</td>\n",
       "      <td>8.304757</td>\n",
       "      <td>C</td>\n",
       "      <td>4.225930</td>\n",
       "      <td>11.627438</td>\n",
       "      <td>2.097700</td>\n",
       "      <td>1.987549</td>\n",
       "      <td>0.171947</td>\n",
       "      <td>...</td>\n",
       "      <td>7.018256</td>\n",
       "      <td>1.812795</td>\n",
       "      <td>0.002267</td>\n",
       "      <td>CJ</td>\n",
       "      <td>1.415230</td>\n",
       "      <td>2.954381</td>\n",
       "      <td>1.990847</td>\n",
       "      <td>1</td>\n",
       "      <td>1.677108</td>\n",
       "      <td>1.034483</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>C</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>Z</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 133 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   ID  target        v1        v2 v3        v4         v5        v6        v7  \\\n",
       "0   3       1  1.335739  8.727474  C  3.921026   7.915266  2.599278  3.176895   \n",
       "1   4       1  0.000000  0.000000  C  0.000000   9.191265  0.000000  0.000000   \n",
       "2   5       1  0.943877  5.310079  C  4.410969   5.326159  3.979592  3.928571   \n",
       "3   6       1  0.797415  8.304757  C  4.225930  11.627438  2.097700  1.987549   \n",
       "4   8       1  0.000000  0.000000  C  0.000000   0.000000  0.000000  0.000000   \n",
       "\n",
       "         v8    ...         v122      v123      v124  v125      v126      v127  \\\n",
       "0  0.012941    ...     8.000000  1.989780  0.035754    AU  1.804126  3.113719   \n",
       "1  2.301630    ...     0.000000  0.000000  0.598896    AF  0.000000  0.000000   \n",
       "2  0.019645    ...     9.333333  2.477596  0.013452    AE  1.773709  3.922193   \n",
       "3  0.171947    ...     7.018256  1.812795  0.002267    CJ  1.415230  2.954381   \n",
       "4  0.000000    ...     0.000000  0.000000  0.000000     Z  0.000000  0.000000   \n",
       "\n",
       "       v128  v129      v130      v131  \n",
       "0  2.024285     0  0.636365  2.857144  \n",
       "1  1.957825     0  0.000000  0.000000  \n",
       "2  1.120468     2  0.883118  1.176472  \n",
       "3  1.990847     1  1.677108  1.034483  \n",
       "4  0.000000     0  0.000000  0.000000  \n",
       "\n",
       "[5 rows x 133 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainclean = trains.fillna(0).head()\n",
    "trainclean"
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
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f76197daa58>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f7627d7db38>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f76197fa518>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nMd4CzuCNAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f76197bac88>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f76196d3160>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('target',data=trainclean,kind='count',size=3)\n",
    "sns.factorplot('v3',data=trainclean,kind='count',size=3)\n",
    "sns.factorplot('target',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v3',data=trains,kind='count',size=3)"
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
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>target</th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "      <th>v3</th>\n",
       "      <th>v4</th>\n",
       "      <th>v5</th>\n",
       "      <th>v6</th>\n",
       "      <th>v7</th>\n",
       "      <th>v8</th>\n",
       "      <th>...</th>\n",
       "      <th>v122</th>\n",
       "      <th>v123</th>\n",
       "      <th>v124</th>\n",
       "      <th>v125</th>\n",
       "      <th>v126</th>\n",
       "      <th>v127</th>\n",
       "      <th>v128</th>\n",
       "      <th>v129</th>\n",
       "      <th>v130</th>\n",
       "      <th>v131</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1031</th>\n",
       "      <td>2032</td>\n",
       "      <td>1</td>\n",
       "      <td>2.138728</td>\n",
       "      <td>11.262827</td>\n",
       "      <td>A</td>\n",
       "      <td>5.418991</td>\n",
       "      <td>9.845327</td>\n",
       "      <td>1.926782</td>\n",
       "      <td>2.581889</td>\n",
       "      <td>0.043371</td>\n",
       "      <td>...</td>\n",
       "      <td>6.666666</td>\n",
       "      <td>0.935229</td>\n",
       "      <td>0.278880</td>\n",
       "      <td>V</td>\n",
       "      <td>1.444342</td>\n",
       "      <td>2.492774</td>\n",
       "      <td>4.463485</td>\n",
       "      <td>2</td>\n",
       "      <td>1.223880</td>\n",
       "      <td>2.926830</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1360</th>\n",
       "      <td>2694</td>\n",
       "      <td>1</td>\n",
       "      <td>1.150855</td>\n",
       "      <td>12.118168</td>\n",
       "      <td>A</td>\n",
       "      <td>4.551629</td>\n",
       "      <td>9.191264</td>\n",
       "      <td>2.721618</td>\n",
       "      <td>2.931570</td>\n",
       "      <td>2.301630</td>\n",
       "      <td>...</td>\n",
       "      <td>6.412213</td>\n",
       "      <td>1.420637</td>\n",
       "      <td>0.598896</td>\n",
       "      <td>AF</td>\n",
       "      <td>1.444587</td>\n",
       "      <td>2.012053</td>\n",
       "      <td>1.957825</td>\n",
       "      <td>1</td>\n",
       "      <td>1.092838</td>\n",
       "      <td>1.553397</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1464</th>\n",
       "      <td>2902</td>\n",
       "      <td>1</td>\n",
       "      <td>1.428227</td>\n",
       "      <td>6.031662</td>\n",
       "      <td>A</td>\n",
       "      <td>5.546056</td>\n",
       "      <td>10.741116</td>\n",
       "      <td>2.036188</td>\n",
       "      <td>2.330520</td>\n",
       "      <td>8.022925</td>\n",
       "      <td>...</td>\n",
       "      <td>6.402640</td>\n",
       "      <td>3.043932</td>\n",
       "      <td>11.549883</td>\n",
       "      <td>AZ</td>\n",
       "      <td>1.452025</td>\n",
       "      <td>2.623644</td>\n",
       "      <td>5.433482</td>\n",
       "      <td>1</td>\n",
       "      <td>1.962733</td>\n",
       "      <td>1.350212</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1933</th>\n",
       "      <td>3896</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>A</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>AR</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2013</th>\n",
       "      <td>4056</td>\n",
       "      <td>1</td>\n",
       "      <td>1.735731</td>\n",
       "      <td>11.334333</td>\n",
       "      <td>A</td>\n",
       "      <td>3.825931</td>\n",
       "      <td>8.643161</td>\n",
       "      <td>2.048475</td>\n",
       "      <td>2.197029</td>\n",
       "      <td>11.245300</td>\n",
       "      <td>...</td>\n",
       "      <td>5.398772</td>\n",
       "      <td>3.486510</td>\n",
       "      <td>1.472451</td>\n",
       "      <td>BH</td>\n",
       "      <td>2.055575</td>\n",
       "      <td>6.215794</td>\n",
       "      <td>1.875762</td>\n",
       "      <td>2</td>\n",
       "      <td>1.651245</td>\n",
       "      <td>2.068966</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2025</th>\n",
       "      <td>4074</td>\n",
       "      <td>1</td>\n",
       "      <td>1.767868</td>\n",
       "      <td>4.329228</td>\n",
       "      <td>A</td>\n",
       "      <td>4.421207</td>\n",
       "      <td>10.827169</td>\n",
       "      <td>2.858849</td>\n",
       "      <td>2.516424</td>\n",
       "      <td>0.220787</td>\n",
       "      <td>...</td>\n",
       "      <td>7.142856</td>\n",
       "      <td>3.361024</td>\n",
       "      <td>0.028551</td>\n",
       "      <td>BL</td>\n",
       "      <td>1.436129</td>\n",
       "      <td>4.643639</td>\n",
       "      <td>2.784175</td>\n",
       "      <td>1</td>\n",
       "      <td>1.183544</td>\n",
       "      <td>2.566844</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3371</th>\n",
       "      <td>6700</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>A</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>BH</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3399</th>\n",
       "      <td>6760</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>A</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>AC</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4790</th>\n",
       "      <td>9572</td>\n",
       "      <td>0</td>\n",
       "      <td>2.847306</td>\n",
       "      <td>8.996227</td>\n",
       "      <td>A</td>\n",
       "      <td>5.348026</td>\n",
       "      <td>10.875775</td>\n",
       "      <td>2.219523</td>\n",
       "      <td>2.486836</td>\n",
       "      <td>5.234731</td>\n",
       "      <td>...</td>\n",
       "      <td>6.301370</td>\n",
       "      <td>2.134766</td>\n",
       "      <td>7.748497</td>\n",
       "      <td>AR</td>\n",
       "      <td>1.056036</td>\n",
       "      <td>5.315917</td>\n",
       "      <td>5.164720</td>\n",
       "      <td>1</td>\n",
       "      <td>1.237786</td>\n",
       "      <td>3.999999</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>9 rows × 133 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        ID  target        v1         v2 v3        v4         v5        v6  \\\n",
       "1031  2032       1  2.138728  11.262827  A  5.418991   9.845327  1.926782   \n",
       "1360  2694       1  1.150855  12.118168  A  4.551629   9.191264  2.721618   \n",
       "1464  2902       1  1.428227   6.031662  A  5.546056  10.741116  2.036188   \n",
       "1933  3896       1       NaN        NaN  A       NaN        NaN       NaN   \n",
       "2013  4056       1  1.735731  11.334333  A  3.825931   8.643161  2.048475   \n",
       "2025  4074       1  1.767868   4.329228  A  4.421207  10.827169  2.858849   \n",
       "3371  6700       1       NaN        NaN  A       NaN        NaN       NaN   \n",
       "3399  6760       1       NaN        NaN  A       NaN        NaN       NaN   \n",
       "4790  9572       0  2.847306   8.996227  A  5.348026  10.875775  2.219523   \n",
       "\n",
       "            v7         v8    ...         v122      v123       v124  v125  \\\n",
       "1031  2.581889   0.043371    ...     6.666666  0.935229   0.278880     V   \n",
       "1360  2.931570   2.301630    ...     6.412213  1.420637   0.598896    AF   \n",
       "1464  2.330520   8.022925    ...     6.402640  3.043932  11.549883    AZ   \n",
       "1933       NaN        NaN    ...          NaN       NaN        NaN    AR   \n",
       "2013  2.197029  11.245300    ...     5.398772  3.486510   1.472451    BH   \n",
       "2025  2.516424   0.220787    ...     7.142856  3.361024   0.028551    BL   \n",
       "3371       NaN        NaN    ...          NaN       NaN        NaN    BH   \n",
       "3399       NaN        NaN    ...          NaN       NaN        NaN    AC   \n",
       "4790  2.486836   5.234731    ...     6.301370  2.134766   7.748497    AR   \n",
       "\n",
       "          v126      v127      v128  v129      v130      v131  \n",
       "1031  1.444342  2.492774  4.463485     2  1.223880  2.926830  \n",
       "1360  1.444587  2.012053  1.957825     1  1.092838  1.553397  \n",
       "1464  1.452025  2.623644  5.433482     1  1.962733  1.350212  \n",
       "1933       NaN       NaN       NaN     1       NaN       NaN  \n",
       "2013  2.055575  6.215794  1.875762     2  1.651245  2.068966  \n",
       "2025  1.436129  4.643639  2.784175     1  1.183544  2.566844  \n",
       "3371       NaN       NaN       NaN     2       NaN       NaN  \n",
       "3399       NaN       NaN       NaN     1       NaN       NaN  \n",
       "4790  1.056036  5.315917  5.164720     1  1.237786  3.999999  \n",
       "\n",
       "[9 rows x 133 columns]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trains.loc[trains['v3'] == \"A\"]"
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
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>target</th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "      <th>v3</th>\n",
       "      <th>v4</th>\n",
       "      <th>v5</th>\n",
       "      <th>v6</th>\n",
       "      <th>v7</th>\n",
       "      <th>v8</th>\n",
       "      <th>...</th>\n",
       "      <th>v122</th>\n",
       "      <th>v123</th>\n",
       "      <th>v124</th>\n",
       "      <th>v125</th>\n",
       "      <th>v126</th>\n",
       "      <th>v127</th>\n",
       "      <th>v128</th>\n",
       "      <th>v129</th>\n",
       "      <th>v130</th>\n",
       "      <th>v131</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>0 rows × 133 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [ID, target, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92, v93, v94, v95, v96, v97, v98, ...]\n",
       "Index: []\n",
       "\n",
       "[0 rows x 133 columns]"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainclean.loc[trainclean['v3'] == 'A']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "################\n",
    "################ CLEAN DATA\n",
    "################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "### Float Data"
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
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "      <th>v4</th>\n",
       "      <th>v5</th>\n",
       "      <th>v6</th>\n",
       "      <th>v7</th>\n",
       "      <th>v8</th>\n",
       "      <th>v9</th>\n",
       "      <th>v10</th>\n",
       "      <th>v11</th>\n",
       "      <th>...</th>\n",
       "      <th>v120</th>\n",
       "      <th>v121</th>\n",
       "      <th>v122</th>\n",
       "      <th>v123</th>\n",
       "      <th>v124</th>\n",
       "      <th>v126</th>\n",
       "      <th>v127</th>\n",
       "      <th>v128</th>\n",
       "      <th>v130</th>\n",
       "      <th>v131</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.335739</td>\n",
       "      <td>8.727474</td>\n",
       "      <td>3.921026</td>\n",
       "      <td>7.915266</td>\n",
       "      <td>2.599278</td>\n",
       "      <td>3.176895</td>\n",
       "      <td>0.012941</td>\n",
       "      <td>9.999999</td>\n",
       "      <td>0.503281</td>\n",
       "      <td>16.434108</td>\n",
       "      <td>...</td>\n",
       "      <td>1.059603</td>\n",
       "      <td>0.803572</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>1.989780</td>\n",
       "      <td>0.035754</td>\n",
       "      <td>1.804126</td>\n",
       "      <td>3.113719</td>\n",
       "      <td>2.024285</td>\n",
       "      <td>0.636365</td>\n",
       "      <td>2.857144</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>9.191265</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2.301630</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.312910</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.598896</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.957825</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.943877</td>\n",
       "      <td>5.310079</td>\n",
       "      <td>4.410969</td>\n",
       "      <td>5.326159</td>\n",
       "      <td>3.979592</td>\n",
       "      <td>3.928571</td>\n",
       "      <td>0.019645</td>\n",
       "      <td>12.666667</td>\n",
       "      <td>0.765864</td>\n",
       "      <td>14.756098</td>\n",
       "      <td>...</td>\n",
       "      <td>2.138728</td>\n",
       "      <td>2.238806</td>\n",
       "      <td>9.333333</td>\n",
       "      <td>2.477596</td>\n",
       "      <td>0.013452</td>\n",
       "      <td>1.773709</td>\n",
       "      <td>3.922193</td>\n",
       "      <td>1.120468</td>\n",
       "      <td>0.883118</td>\n",
       "      <td>1.176472</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.797415</td>\n",
       "      <td>8.304757</td>\n",
       "      <td>4.225930</td>\n",
       "      <td>11.627438</td>\n",
       "      <td>2.097700</td>\n",
       "      <td>1.987549</td>\n",
       "      <td>0.171947</td>\n",
       "      <td>8.965516</td>\n",
       "      <td>6.542669</td>\n",
       "      <td>16.347483</td>\n",
       "      <td>...</td>\n",
       "      <td>1.166281</td>\n",
       "      <td>1.956521</td>\n",
       "      <td>7.018256</td>\n",
       "      <td>1.812795</td>\n",
       "      <td>0.002267</td>\n",
       "      <td>1.415230</td>\n",
       "      <td>2.954381</td>\n",
       "      <td>1.990847</td>\n",
       "      <td>1.677108</td>\n",
       "      <td>1.034483</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.050328</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 108 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         v1        v2        v4         v5        v6        v7        v8  \\\n",
       "0  1.335739  8.727474  3.921026   7.915266  2.599278  3.176895  0.012941   \n",
       "1       NaN       NaN       NaN   9.191265       NaN       NaN  2.301630   \n",
       "2  0.943877  5.310079  4.410969   5.326159  3.979592  3.928571  0.019645   \n",
       "3  0.797415  8.304757  4.225930  11.627438  2.097700  1.987549  0.171947   \n",
       "4       NaN       NaN       NaN        NaN       NaN       NaN       NaN   \n",
       "\n",
       "          v9       v10        v11    ...         v120      v121      v122  \\\n",
       "0   9.999999  0.503281  16.434108    ...     1.059603  0.803572  8.000000   \n",
       "1        NaN  1.312910        NaN    ...          NaN       NaN       NaN   \n",
       "2  12.666667  0.765864  14.756098    ...     2.138728  2.238806  9.333333   \n",
       "3   8.965516  6.542669  16.347483    ...     1.166281  1.956521  7.018256   \n",
       "4        NaN  1.050328        NaN    ...          NaN       NaN       NaN   \n",
       "\n",
       "       v123      v124      v126      v127      v128      v130      v131  \n",
       "0  1.989780  0.035754  1.804126  3.113719  2.024285  0.636365  2.857144  \n",
       "1       NaN  0.598896       NaN       NaN  1.957825       NaN       NaN  \n",
       "2  2.477596  0.013452  1.773709  3.922193  1.120468  0.883118  1.176472  \n",
       "3  1.812795  0.002267  1.415230  2.954381  1.990847  1.677108  1.034483  \n",
       "4       NaN       NaN       NaN       NaN       NaN       NaN       NaN  \n",
       "\n",
       "[5 rows x 108 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trains.loc[:, trains.dtypes == np.float64].head()"
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
     "data": {
      "text/plain": [
       "219324"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Do I have NaN values?\n",
    "trains.loc[:, trains.dtypes == np.float64].isnull().sum().sum()"
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
      "{0: 'normal', 1: 'normal', 2: 'normal', 3: 'normal', 4: 'normal', 5: 'normal', 6: 'normal', 7: 'normal', 8: 'normal', 9: 'normal', 10: 'normal', 11: 'normal', 12: 'nonnormal', 13: 'normal', 14: 'normal', 15: 'normal', 16: 'normal', 17: 'normal', 18: 'normal', 19: 'normal', 20: 'normal', 21: 'normal', 22: 'normal', 23: 'normal', 24: 'normal', 25: 'normal', 26: 'normal', 27: 'normal', 28: 'normal', 29: 'normal', 30: 'normal', 31: 'normal', 32: 'normal', 33: 'normal', 34: 'normal', 35: 'normal', 36: 'normal', 37: 'normal', 38: 'normal', 39: 'normal', 40: 'normal', 41: 'normal', 42: 'normal', 43: 'normal', 44: 'normal', 45: 'normal', 46: 'normal', 47: 'normal', 48: 'normal', 49: 'normal', 50: 'normal', 51: 'normal', 52: 'normal', 53: 'normal', 54: 'normal', 55: 'normal', 56: 'normal', 57: 'normal', 58: 'normal', 59: 'normal', 60: 'normal', 61: 'normal', 62: 'normal', 63: 'normal', 64: 'normal', 65: 'normal', 66: 'normal', 67: 'normal', 68: 'normal', 69: 'normal', 70: 'normal', 71: 'normal', 72: 'normal', 73: 'normal', 74: 'normal', 75: 'normal', 76: 'normal', 77: 'normal', 78: 'normal', 79: 'normal', 80: 'normal', 81: 'normal', 82: 'normal', 83: 'normal', 84: 'normal', 85: 'normal', 86: 'normal', 87: 'normal', 88: 'normal', 89: 'normal', 90: 'normal', 91: 'normal', 92: 'nonnormal', 93: 'normal', 94: 'normal', 95: 'normal', 96: 'normal', 97: 'normal', 98: 'normal', 99: 'normal', 100: 'normal', 101: 'normal', 102: 'normal', 103: 'normal', 104: 'normal', 105: 'normal', 106: 'normal', 107: 'normal'}\n"
     ]
    }
   ],
   "source": [
    "### Type of distribution of my data\n",
    "\n",
    "dist = {}\n",
    "# Normalize data for each column\n",
    "columns = trains.columns[(trains.dtypes == np.float64)]\n",
    "for i in range(len(columns)):\n",
    "    trains.update(trains[columns[i]].notnull().apply(lambda x: (x - x.mean()) / (x.max() - x.min())).dropna())\n",
    "    # check if normal distribution\n",
    "    hyp = stats.shapiro(trains[columns[i]])\n",
    "    # if p-value < 0.05 is not normally distributed\n",
    "    if hyp[1] < 0.05:\n",
    "        dist[i] = \"nonnormal\"\n",
    "    else:\n",
    "        dist[i] = \"normal\"\n",
    "print(dist)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# plot a couple of features to see the distribution they follow\n",
    "#f, ax = plt.subplots(2)\n",
    "#sns.distplot(trains['v1'], bins=25, kde=False, rug=True, ax=ax[0])\n",
    "#sns.distplot(trains['v15'], bins=25, kde=False, rug=True, ax=ax[1])\n",
    "#sns.distplot(trains['v34'], bins=25, kde=False, rug=True, ax=ax[2])\n",
    "#sns.distplot(trains['v45'], bins=25, kde=False, rug=True, ax=ax[3])\n",
    "#sns.distplot(trains['v60'], bins=25, kde=False, rug=True, ax=ax[4])\n",
    "#sns.distplot(trains['v78'], bins=25, kde=False, rug=True, ax=ax[5])\n",
    "#sns.distplot(trains['v90'], bins=25, kde=False, rug=True, ax=ax[6])\n",
    "#sns.distplot(trains['v108'], bins=25, kde=False, rug=True, ax=ax[7])\n",
    "#sns.distplot(trains['v130'], bins=25, kde=False, rug=True, ax=ax[8])"
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
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>...</th>\n",
       "      <th>98</th>\n",
       "      <th>99</th>\n",
       "      <th>100</th>\n",
       "      <th>101</th>\n",
       "      <th>102</th>\n",
       "      <th>103</th>\n",
       "      <th>104</th>\n",
       "      <th>105</th>\n",
       "      <th>106</th>\n",
       "      <th>107</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.335739</td>\n",
       "      <td>8.727474</td>\n",
       "      <td>3.921026</td>\n",
       "      <td>7.915266</td>\n",
       "      <td>2.599278</td>\n",
       "      <td>3.176895</td>\n",
       "      <td>0.012941</td>\n",
       "      <td>9.999999</td>\n",
       "      <td>0.503281</td>\n",
       "      <td>16.434108</td>\n",
       "      <td>...</td>\n",
       "      <td>1.059603</td>\n",
       "      <td>0.803572</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>1.989780</td>\n",
       "      <td>0.035754</td>\n",
       "      <td>1.804126</td>\n",
       "      <td>3.113719</td>\n",
       "      <td>2.024285</td>\n",
       "      <td>0.636365</td>\n",
       "      <td>2.857144</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.478275</td>\n",
       "      <td>6.817772</td>\n",
       "      <td>4.182575</td>\n",
       "      <td>9.191265</td>\n",
       "      <td>2.400521</td>\n",
       "      <td>2.447117</td>\n",
       "      <td>2.301630</td>\n",
       "      <td>9.108172</td>\n",
       "      <td>1.312910</td>\n",
       "      <td>15.511311</td>\n",
       "      <td>...</td>\n",
       "      <td>1.164874</td>\n",
       "      <td>2.457737</td>\n",
       "      <td>6.779235</td>\n",
       "      <td>2.766150</td>\n",
       "      <td>0.598896</td>\n",
       "      <td>1.611466</td>\n",
       "      <td>2.965299</td>\n",
       "      <td>1.957825</td>\n",
       "      <td>1.593623</td>\n",
       "      <td>1.592920</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.943877</td>\n",
       "      <td>5.310079</td>\n",
       "      <td>4.410969</td>\n",
       "      <td>5.326159</td>\n",
       "      <td>3.979592</td>\n",
       "      <td>3.928571</td>\n",
       "      <td>0.019645</td>\n",
       "      <td>12.666667</td>\n",
       "      <td>0.765864</td>\n",
       "      <td>14.756098</td>\n",
       "      <td>...</td>\n",
       "      <td>2.138728</td>\n",
       "      <td>2.238806</td>\n",
       "      <td>9.333333</td>\n",
       "      <td>2.477596</td>\n",
       "      <td>0.013452</td>\n",
       "      <td>1.773709</td>\n",
       "      <td>3.922193</td>\n",
       "      <td>1.120468</td>\n",
       "      <td>0.883118</td>\n",
       "      <td>1.176472</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.797415</td>\n",
       "      <td>8.304757</td>\n",
       "      <td>4.225930</td>\n",
       "      <td>11.627438</td>\n",
       "      <td>2.097700</td>\n",
       "      <td>1.987549</td>\n",
       "      <td>0.171947</td>\n",
       "      <td>8.965516</td>\n",
       "      <td>6.542669</td>\n",
       "      <td>16.347483</td>\n",
       "      <td>...</td>\n",
       "      <td>1.166281</td>\n",
       "      <td>1.956521</td>\n",
       "      <td>7.018256</td>\n",
       "      <td>1.812795</td>\n",
       "      <td>0.002267</td>\n",
       "      <td>1.415230</td>\n",
       "      <td>2.954381</td>\n",
       "      <td>1.990847</td>\n",
       "      <td>1.677108</td>\n",
       "      <td>1.034483</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.478275</td>\n",
       "      <td>6.817772</td>\n",
       "      <td>4.182575</td>\n",
       "      <td>8.641835</td>\n",
       "      <td>2.400521</td>\n",
       "      <td>2.447117</td>\n",
       "      <td>0.391733</td>\n",
       "      <td>9.108172</td>\n",
       "      <td>1.050328</td>\n",
       "      <td>15.511311</td>\n",
       "      <td>...</td>\n",
       "      <td>1.164874</td>\n",
       "      <td>2.457737</td>\n",
       "      <td>6.779235</td>\n",
       "      <td>2.766150</td>\n",
       "      <td>0.137220</td>\n",
       "      <td>1.611466</td>\n",
       "      <td>2.965299</td>\n",
       "      <td>1.789658</td>\n",
       "      <td>1.593623</td>\n",
       "      <td>1.592920</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 108 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        0         1         2          3         4         5         6    \\\n",
       "0  1.335739  8.727474  3.921026   7.915266  2.599278  3.176895  0.012941   \n",
       "1  1.478275  6.817772  4.182575   9.191265  2.400521  2.447117  2.301630   \n",
       "2  0.943877  5.310079  4.410969   5.326159  3.979592  3.928571  0.019645   \n",
       "3  0.797415  8.304757  4.225930  11.627438  2.097700  1.987549  0.171947   \n",
       "4  1.478275  6.817772  4.182575   8.641835  2.400521  2.447117  0.391733   \n",
       "\n",
       "         7         8          9      ...          98        99        100  \\\n",
       "0   9.999999  0.503281  16.434108    ...     1.059603  0.803572  8.000000   \n",
       "1   9.108172  1.312910  15.511311    ...     1.164874  2.457737  6.779235   \n",
       "2  12.666667  0.765864  14.756098    ...     2.138728  2.238806  9.333333   \n",
       "3   8.965516  6.542669  16.347483    ...     1.166281  1.956521  7.018256   \n",
       "4   9.108172  1.050328  15.511311    ...     1.164874  2.457737  6.779235   \n",
       "\n",
       "        101       102       103       104       105       106       107  \n",
       "0  1.989780  0.035754  1.804126  3.113719  2.024285  0.636365  2.857144  \n",
       "1  2.766150  0.598896  1.611466  2.965299  1.957825  1.593623  1.592920  \n",
       "2  2.477596  0.013452  1.773709  3.922193  1.120468  0.883118  1.176472  \n",
       "3  1.812795  0.002267  1.415230  2.954381  1.990847  1.677108  1.034483  \n",
       "4  2.766150  0.137220  1.611466  2.965299  1.789658  1.593623  1.592920  \n",
       "\n",
       "[5 rows x 108 columns]"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# substitute in float data NaN by median value \n",
    "# save values into a numpy array(n_samples,n_features)\n",
    "imp = Imputer(missing_values='NaN', strategy='median', axis=0)\n",
    "imp.fit(trains.loc[:, trains.dtypes == np.float64])\n",
    "X = trains.loc[:, trains.dtypes == np.float64]\n",
    "floatFeatures = pd.DataFrame(imp.transform(X))\n",
    "floatFeatures.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7f761976fb00>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f76701bc208>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot a couple of features to see the distribution they follow once the median is applied to NaN\n",
    "contdata2 = floatFeatures.apply(lambda x: (x - x.mean()) / (x.max() - x.min())).dropna()\n",
    "f, ax = plt.subplots(2, sharex=True, figsize=(10,8))\n",
    "sns.distplot(contdata2[0], bins=25, kde=False, rug=True, ax=ax[0])\n",
    "sns.distplot(contdata2[15], bins=25, kde=False, rug=True, ax=ax[1])\n",
    "#sns.distplot(contdata2[29], bins=25, kde=False, rug=True, ax=ax[2])\n",
    "#sns.distplot(contdata2[39], bins=25, kde=False, rug=True, ax=ax[3])\n",
    "#sns.distplot(contdata2[51], bins=25, kde=False, rug=True, ax=ax[4])\n",
    "#sns.distplot(contdata2[63], bins=25, kde=False, rug=True, ax=ax[5])\n",
    "#sns.distplot(contdata2[74], bins=25, kde=False, rug=True, ax=ax[6])\n",
    "#sns.distplot(contdata2[89], bins=25, kde=False, rug=True, ax=ax[7])\n",
    "#sns.distplot(contdata2[107], bins=25, kde=False, rug=True, ax=ax[8])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "### Integers Categorical Data"
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
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>target</th>\n",
       "      <th>v38</th>\n",
       "      <th>v62</th>\n",
       "      <th>v72</th>\n",
       "      <th>v129</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ID  target  v38  v62  v72  v129\n",
       "0   3       1    0    1    1     0\n",
       "1   4       1    0    2    2     0\n",
       "2   5       1    0    1    3     2\n",
       "3   6       1    0    1    2     1\n",
       "4   8       1    0    1    1     0"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trains.loc[:, trains.dtypes == np.int].head()"
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
       "0"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Do I have NaN values?\n",
    "trains.loc[:, trains.dtypes == np.int].isnull().sum().sum()"
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
    "# No NaN values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f7610b89f28>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAOAAAADRCAYAAADVC77pAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAD4NJREFUeJzt3X2Q1dV9x/E3QiUV5EkXtEkTo3U+M200HaTVYgwgAzRx\nNDPFpyqMkSZVE1MlzTSaNmpMYjMx0mnNMKaaINI4Q0jUgfiM9QlNfJhJjHQm3ybaTVMkAkIR7CzC\n7vaP89vsdWcfLsv+7tl7f5/XjOPl3nO5Zxk+/H73d8/53DHd3d2YWR6H5Z6AWZU5gGYZOYBmGTmA\nZhk5gGYZOYBmGY0r+wUkvQvYDNwI/DuwhhT8rcDSiNgv6WLgKqATuD0iviNpHHAn8D7gAHBpRLSX\nPV+zRmrEEfCLwBvF7RuBWyNiDvAKsEzSEcWYM4F5wHJJU4CLgF0RcQZwE/C1BszVrKFKDaAkAQLu\nB8YAc4ANxcMbgAXAqcDzEbE3IjqATcCHgPnAvcXYjcDpZc7VLIeyj4DfAD5LCh/AhIjYX9zeBhwL\nzAC21zxne9/7I6Ib6CpOS81aRmkBlLQUeDIi/nuAIWMO8n5fMLKWU+YR5Szg/ZIWA+8G3gb2Shof\nEfuK+7YAr5GOeD3eDfyouP8Y4OWeI19EHBjqRQ8c6OweN27siP4gZodooINKeQGMiAt7bku6DmgH\nZgPnAt8FFgMPAc8Dd0iaBHQVY64CJgPnAY8C5wCP1/O6u3b934j9DGYjoa3tyAEfa9RpXc+/ANcD\nl0h6EpgKrC4uvFwDPFL8d0NE7AHWAuMkPQ1cAVzboLmaNcyYVtuOtH37ntb6gazptbUdOeApqC9s\nmGXkAJpl5ACaZeQAmmXklSU2LJ2dnbS3v5p7GqPCcccdz9ixw/vs2QG0YWlvf5UvrruRiUdPyj2V\nrPbueJMvn3cdJ5xw4rCe7wDasE08ehKTj5maexpNze8BzTJyAM0ycgDNMnIAzTIq9SKMpN8l9brM\nAMYDXyHthjgF2FEMuzkiHnQvjFVR2VdBzwZeiIhvSHovaWvRM8A1EfFAz6CaXphZpKC9IOke0jak\nXRGxRNICUi/MhX1fxKxZlRrAiPhezS/fC/y6uN13dfhve2EAJNX2wqwuxmwEvlPebM0aryHvASU9\nA/wbcDUpfJ+W9JikuyUdRdr57l4Yq5yG/GWOiNMlnUzaCX818EZE/EzS54EbgGf7PGXYvTBTpx6B\nKynKt2vXxNxTGDWmTZs46K73wZR9EeYUYFtE/LoI3Djg5YjouQCzHlgJrCO9X+wx7F4YV1I0xs6d\ne3NPYdTYuXMv27fvGfDxnJUUZ5BqCZE0A5gIfEvSScXjc0it2c8DsyRNkjSR1AvzNOmizXnF2Lp7\nYcyaRdmnoLcB35b0FPAu4FPAXmCVpD3F7UsjokNSTy9MF0UvjKS1wIKiF6YD+HjJ8zVrqLKvgnYA\nF/fz0Kx+xt4D3NPnvi5gWTmzM8vPK2HMMnIAzTJyAM0ycgDNMnIAzTJyAM0ycgDNMnIAzTJyAM0y\ncgDNMnIAzTLK0QnzErCGFP6twNKI2O9OGKuiso+APZ0wc4ELgBXAjcA3I2IO8AqwrKYT5kxgHrBc\n0hTgIlInzBnATaROGLOWkaMTZg5wWXHfBuBzwH/iThiroEZ3wiwHJkTE/uKhbfTpfim4E8YqIUcn\nTG3fy0DdL+6EGeXcCdOrmTphxgJ7JI2PiH2k7pctpO6XY2ue6k6YUc6dML2arRNmI6kdG2Ax8BDu\nhLGKKjuAtwHTi06YDcAVwPXAJZKeBKYCq4vqip5OmEcoOmGAtcC4ohPmCuDakudr1lC5OmEW9jPW\nnTBWOV4JY5aRA2iWkQNolpEDaJaRA2iWkQNolpEDaJaRA2iWkQNolpEDaJZR6duRJH2dtLl2LGlH\n+znAKUDPt+TeHBEPupLCqqjs7UhzgT+KiNmSpgE/AR4DromIB2rG9VRSzCIF7QVJ95DCuisilkha\nQArwhWXO2ayRyj4FfYre7UT/C0wgHQn7brg9laKSoljAXVtJcW8xZiNwesnzNWuosndDdAE9O2Q/\nAdxPOsW8UtJngdeBz5A23Q5ZSSGpS9K4oTblmjWLRnXCfAy4FLiSVEn4+YiYT6oovKGfpwy7ksKs\nmTTiIswi0kbaRcUm29pd7euBlcA6UoVhj2FXUrgTpjHcCdNrNHfCTAK+DsyPiN3Ffd8HvhQRL5Mq\nCjeTKinuKMZ3kSoprgImk95DPkqdlRTuhGkMd8L0OpROmLKPgBcARwHfkzQG6AZWAask7QH2kj5a\n6JDUU0nRRVFJIWktsKCopOgAPl7yfM0aquyLMLcDt/fz0Jp+xrqSwirHFzXMMnIAzTJyAM0ycgDN\nMnIAzTKqK4CS7uznvodHfDZmFTPoxxDFFqHLgQ8U9fI9Diet0zSzQzBoACPiu5KeIH2t2PU1D3UB\n/1HivMwqYcgP4iNiCzBX0mRgGr0LpacAO0ucm1nLq2sljKR/Jq1I2U5vALuB40ual1kl1LsU7Uyg\nrdgsa2YjpN4A/mK44eunE+YF0lrQw4CtwNKI2O9OGKuiegP4P8VV0E2kIAAQEdcN9qRBOmG+GRE/\nkPRVYJmkNbgTxiqo3g/i3yAFZx/pCNXz31D664SZQ9qIC+lbcxfgThirqHqPgF8ezm/epxPmr0id\nMIsiYn9x3zb6dL8U3AljlVBvAA+Qrnr26AZ2kzbbDqnohFlG+mrqX9Y8NFD3izthrBLqCmBE/PYv\nvqTDSaeGH6znuX07YSTtkTQ+IvaRul+2kLpfjq15mjthRjl3wvRqaCdMRLwNPCjpc6SLIgPqrxOG\n9F5uMXB38f+HcCdM03EnTK/SO2Ek9a2F+H3SUWoo/XXCXAJ8W9JlwK+A1RHR6U4Yq6J6j4Bn1Nzu\nBt4Ezh/qSYN0wizsZ6w7Yaxy6n0PeClA8Vled0TsKnVWZhVR7ynobNLqlSOBMZLeAJZExItlTs6s\n1dV7Wf9rwMciYnpEtAF/Cawob1pm1VBvADsjYnPPLyLiJ9QsSTOz4an3IkyXpMWkjwMA/pz6lqKZ\n2SDqDeDlwK3AHaSPCX4KfLKsSZlVRb2noAuBfRExNSKOKp730fKmZVYN9QZwCfAXNb9eCFw88tMx\nq5Z6Azg2Imrf83WVMRmzqqn3PeB6Sc8CT5NCOx/4QWmzMquIuo6AEfEV4O9I+/e2Ap+KiK+WOTGz\nKqh7N0REbCLtVD8okk4mrfFcERErJa0CTgF2FENujogH3QljVVT2V1QfAdxC2uVQ65qIeKDPOHfC\nWOWUvcO8AzgLeH2Ice6EsUoqNYAR0VVs4O3rSkmPSbpb0lGkXe9DdsKQVuSU/b32Zg2To2PlLtIp\n6HzgJeCGfsa4E8YqoeFHk4iorZVYD6wE1gFn19zvTphRzp0wvRraCXOoJH0f+FJEvEzqCN2MO2Ga\njjthepXeCTNckk4lLeBuAw5Iupz0NWerJO0B9pI+WuhwJ4xVUakBjIjngJP6eejefsa6E8Yqxxc1\nzDJyAM0ycgDNMnIAzTJyAM0yqtSyrs7OTtrbX809jVHhuOOOZ+xYL1jIrVIBbG9/lWtvWcuEyW25\np5LVW7u3849/ewEnnHBi7qlUXqUCCDBhchuTph079ECzBvB7QLOMHECzjBxAs4xKfw/YTyfMe0jf\ntHQYqeBpaUTsdyeMVVGpR8ABOmFuBG6NiDnAK8Cymk6YM4F5wHJJU4CLSJ0wZwA3McRXYps1mxyd\nMHOBDcXtDcAC3AljFZWjE2ZCROwvbm+jT/dLwZ0wVgm5/zIP1P0y7E6YwSopXKPQ61BqFMB/lrWa\nqpIC2CNpfETsI3W/bCF1v9R+Oj7sTpjBKilco9BrqBqFep5vyaFUUuT4GGIjsLi4vRh4iNQJM0vS\nJEkTSZ0wT5O6YM4rxtbVCWPWTHJ0wiwCVku6DPgVsDoiOt0JY1WUqxNmYT9j3QljleOVMGYZOYBm\nGTmAZhk5gGYZOYBmGTmAZhk5gGYZOYBmGTmAZhk5gGYZOYBmGeX4htw5pK+k3kza9/cz4Gbq7Ilp\n9HzNypTrCPhERJwZEfMi4ioOrifGrGXkCmDfHe9zqa8nxp0w1lJyVVL8oaT7gGmko98RB9ETY9Yy\ncgTwF6QNt+skHU/a5V47j4PtiXkHd8LUx50wI6epOmEi4jXSRRgi4lVJvyHVUdTbEzMod8LUx50w\nI6epOmEkXSTp+uL2dGA6sAo4txgyVE+MWcvIcQq6Hrhb0ibSPwCXAy8Bd0n6a4boickwX7PS5DgF\n3UtqOOurrp4Ys1bilTBmGTmAZhk5gGYZOYBmGTmAZhk5gGYZOYBmGTmAZhk5gGYZOYBmGeX+iuoh\nSVoBnEZaD3p1RLyYeUpmI2ZUHwElfRj4g4iYDXwC+JfMUzIbUaM6gMB84D6AiPg5MKXYmmTWEkZ7\nAI/hnbUUO4r7zFrCqH8P2EddtRSDeWv39qEHtbiR+jPYu+PNEfl9mtmh/hmM9gC+xjuPeL9H6g0d\nUFvbkQOGtK1tJo+vmzlCU6u2traZPHrafbmn0fRG+ynoIxRVFZJmAlsi4q28UzIbOWO6u7tzz2FQ\nkm4C5pDasT8dES9nnpLZiBn1ATRrZaP9FNSspTmAZhk5gGYZjfaPIVqO17aOHEknk2orV0TEytzz\nGQ4fARvIa1tHTvH1dbeQPqpqWg5gY3lt68jpAM4CXs89kUPhADaW17aOkIjoioi3c8/jUDmAeR3y\n2lZrbg5gYx302lZrbQ5gY3ltazma9kzCS9EazGtbR4akU4E7gDbgALATmBMRu7JO7CA5gGYZ+RTU\nLCMH0CwjB9AsIwfQLCMH0CwjB9AsIwewAiRdXPLv/xFJU8p8jVblALY4SWOB60p+meXAUSW/Rkvy\nB/EtTtKdwAXAk8BzwALSypEtwJKI6JS0m7Sq5HDgb4B/Bf4Y+CVpxc4jEXGXpPOBK4vfejvwSeB8\n4J+AnwKXFtusrE4+Ara+60lh+SjwFvChiPgwMBVYVIyZCNwfEZ8hBfQDEfEnwFXARwAkvQf4AjC/\neP6TwLURcRvwG+Aih+/gOYAVERFdpBqMpyQ9AXwQOLp4eAzwbHH7JOCZ4jnbau7/M+BY4GFJj5OO\nqrU7O5p2QXRO7oSpCEmzgWXAzIjokLSuz5Ceza2HkYLa1z7guYg4p8RpVo6PgK2vi/TebgbwX0X4\n3kc6oo3vZ/zPgVkAkqYX4wBeAP5U0ozisXMlnV3zGr9T3o/QuhzA1vcaadPvF4BjJG0C/oF0ZfTv\nJZ0I1F6JexDYIenHwArS6eiBiNhKek/4w+IUdhnw4+I5DwMbJJ3WgJ+npfgqqL2DpMnAORGxRtIY\n4CVgmesTy+EjoPW1B5gn6UXgR8APHb7y+AholpGPgGYZOYBmGTmAZhk5gGYZOYBmGTmAZhn9P1/C\nMd4CzuCNAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f76196d3eb8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f7618a5b588>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f7610cb9c18>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f7610aecdd8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nTWlEfj+bqV8AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f7610b91d68>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('target',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v38',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v62',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v72',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v129',data=trains,kind='count',size=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Transform each category into one boolean feature"
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
       "array([ 0, 13, 20, 33, 41])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "enc = preprocessing.OneHotEncoder()\n",
    "columns = trains.columns[(trains.dtypes == np.int)]\n",
    "# exclude first column of IDs\n",
    "enc = enc.fit(trains.loc[:, columns[2:]])\n",
    "enc.feature_indices_"
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
       "array([13,  7, 13,  8])"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "enc.n_values_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['ID', 'target', 'v38', 'v62', 'v72', 'v129'], dtype='object')"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "columns"
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
       "array([   3,    4,    5, ..., 9970, 9972, 9973])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trains.loc[:, columns[0]].unique()"
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
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>v38</th>\n",
       "      <th>v62</th>\n",
       "      <th>v72</th>\n",
       "      <th>v129</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   v38  v62  v72  v129\n",
       "0    0    1    1     0\n",
       "1    0    2    2     0\n",
       "2    0    1    3     2\n",
       "3    0    1    2     1\n",
       "4    0    1    1     0"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trains.loc[:, columns[2:]].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
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
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>...</th>\n",
       "      <th>28</th>\n",
       "      <th>29</th>\n",
       "      <th>30</th>\n",
       "      <th>31</th>\n",
       "      <th>32</th>\n",
       "      <th>33</th>\n",
       "      <th>34</th>\n",
       "      <th>35</th>\n",
       "      <th>36</th>\n",
       "      <th>37</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 38 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   0   1   2   3   4   5   6   7   8   9  ...  28  29  30  31  32  33  34  35  \\\n",
       "0   1   0   0   0   0   0   0   0   0   0 ...   0   0   1   0   0   0   0   0   \n",
       "1   1   0   0   0   0   0   0   0   0   0 ...   0   0   1   0   0   0   0   0   \n",
       "2   1   0   0   0   0   0   0   0   0   0 ...   0   0   0   0   1   0   0   0   \n",
       "3   1   0   0   0   0   0   0   0   0   0 ...   0   0   0   1   0   0   0   0   \n",
       "4   1   0   0   0   0   0   0   0   0   0 ...   0   0   1   0   0   0   0   0   \n",
       "\n",
       "   36  37  \n",
       "0   0   0  \n",
       "1   0   0  \n",
       "2   0   0  \n",
       "3   0   0  \n",
       "4   0   0  \n",
       "\n",
       "[5 rows x 38 columns]"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = enc.transform(trains.loc[:, columns[2:]]).toarray()\n",
    "intFeatures = pd.DataFrame(x)\n",
    "intFeatures.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "### Objects Categorical Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
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
       "      <th>v3</th>\n",
       "      <th>v22</th>\n",
       "      <th>v24</th>\n",
       "      <th>v30</th>\n",
       "      <th>v31</th>\n",
       "      <th>v47</th>\n",
       "      <th>v52</th>\n",
       "      <th>v56</th>\n",
       "      <th>v66</th>\n",
       "      <th>v71</th>\n",
       "      <th>v74</th>\n",
       "      <th>v75</th>\n",
       "      <th>v79</th>\n",
       "      <th>v91</th>\n",
       "      <th>v107</th>\n",
       "      <th>v110</th>\n",
       "      <th>v112</th>\n",
       "      <th>v113</th>\n",
       "      <th>v125</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>C</td>\n",
       "      <td>XDX</td>\n",
       "      <td>C</td>\n",
       "      <td>C</td>\n",
       "      <td>A</td>\n",
       "      <td>C</td>\n",
       "      <td>G</td>\n",
       "      <td>DI</td>\n",
       "      <td>C</td>\n",
       "      <td>F</td>\n",
       "      <td>B</td>\n",
       "      <td>D</td>\n",
       "      <td>E</td>\n",
       "      <td>A</td>\n",
       "      <td>E</td>\n",
       "      <td>B</td>\n",
       "      <td>O</td>\n",
       "      <td>NaN</td>\n",
       "      <td>AU</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>C</td>\n",
       "      <td>GUV</td>\n",
       "      <td>C</td>\n",
       "      <td>C</td>\n",
       "      <td>A</td>\n",
       "      <td>E</td>\n",
       "      <td>G</td>\n",
       "      <td>DY</td>\n",
       "      <td>A</td>\n",
       "      <td>F</td>\n",
       "      <td>B</td>\n",
       "      <td>D</td>\n",
       "      <td>D</td>\n",
       "      <td>B</td>\n",
       "      <td>B</td>\n",
       "      <td>A</td>\n",
       "      <td>U</td>\n",
       "      <td>G</td>\n",
       "      <td>AF</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>C</td>\n",
       "      <td>FQ</td>\n",
       "      <td>E</td>\n",
       "      <td>NaN</td>\n",
       "      <td>A</td>\n",
       "      <td>C</td>\n",
       "      <td>F</td>\n",
       "      <td>AS</td>\n",
       "      <td>A</td>\n",
       "      <td>B</td>\n",
       "      <td>B</td>\n",
       "      <td>B</td>\n",
       "      <td>E</td>\n",
       "      <td>G</td>\n",
       "      <td>C</td>\n",
       "      <td>B</td>\n",
       "      <td>S</td>\n",
       "      <td>NaN</td>\n",
       "      <td>AE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>C</td>\n",
       "      <td>ACUE</td>\n",
       "      <td>D</td>\n",
       "      <td>C</td>\n",
       "      <td>B</td>\n",
       "      <td>C</td>\n",
       "      <td>H</td>\n",
       "      <td>BW</td>\n",
       "      <td>A</td>\n",
       "      <td>F</td>\n",
       "      <td>B</td>\n",
       "      <td>D</td>\n",
       "      <td>B</td>\n",
       "      <td>B</td>\n",
       "      <td>B</td>\n",
       "      <td>B</td>\n",
       "      <td>J</td>\n",
       "      <td>NaN</td>\n",
       "      <td>CJ</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>C</td>\n",
       "      <td>HIT</td>\n",
       "      <td>E</td>\n",
       "      <td>NaN</td>\n",
       "      <td>A</td>\n",
       "      <td>I</td>\n",
       "      <td>H</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "      <td>F</td>\n",
       "      <td>B</td>\n",
       "      <td>D</td>\n",
       "      <td>C</td>\n",
       "      <td>G</td>\n",
       "      <td>C</td>\n",
       "      <td>A</td>\n",
       "      <td>T</td>\n",
       "      <td>G</td>\n",
       "      <td>Z</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  v3   v22 v24  v30 v31 v47 v52  v56 v66 v71 v74 v75 v79 v91 v107 v110 v112  \\\n",
       "0  C   XDX   C    C   A   C   G   DI   C   F   B   D   E   A    E    B    O   \n",
       "1  C   GUV   C    C   A   E   G   DY   A   F   B   D   D   B    B    A    U   \n",
       "2  C    FQ   E  NaN   A   C   F   AS   A   B   B   B   E   G    C    B    S   \n",
       "3  C  ACUE   D    C   B   C   H   BW   A   F   B   D   B   B    B    B    J   \n",
       "4  C   HIT   E  NaN   A   I   H  NaN   C   F   B   D   C   G    C    A    T   \n",
       "\n",
       "  v113 v125  \n",
       "0  NaN   AU  \n",
       "1    G   AF  \n",
       "2  NaN   AE  \n",
       "3  NaN   CJ  \n",
       "4    G    Z  "
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trains.loc[:, trains.dtypes == np.object].head()"
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
       "<seaborn.axisgrid.FacetGrid at 0x7f760d83e438>"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f7610c1add8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\niEf/D4bTpoc9vQKaAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f7610c36ba8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f7610b9b5f8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\niej/AXCRKyF5OW+8AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760cd69cc0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nRK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f7610b9b2e8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nrkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760cec32b0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760cdfca90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nEBOgYUTI/wNNLwE9JxF8vwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760d006ba8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760d0ff048>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nmGB46J3TXwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760d3e8e80>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760d46ac18>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\ncMwAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760d43a2e8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAOAAAADRCAYAAADVC77pAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAFGVJREFUeJzt3XuYHFWZx/HvZBDMBUNmMjOQGBIYyZv1grrLCouy3IQg\nsLoKEkEUxA2gsiJ5kEVdQVEUEVEu6iNhQUSyIgRdwkUCEkE2LheFXUB5kYwdY6KZIYQkkHDJzOwf\np5p02u6emkl3n56u3+d55kl31UnX6Z55u6qrz/lVy+DgICISx5jYHRDJMhWgSEQqQJGIVIAiEakA\nRSJSAYpEtF2tN2BmewI3ARe7+3fMbDvgGuB1wHrgaHdfZ2YfBE4H+oH57n5V0vb7wHRgM/ARd8/V\nus8i9VLTPaCZjQO+ASwuWDwX6HX3vYHrgf2Sdp8HDgIOBM4ws52A44C17r4f8BXgglr2V6Tean0I\n+gJwBLC6YNk/AdcBuPuV7n4LsDfwgLs/5+4vAPcB7wAOBn6S/L+7gLfXuL8idVXTAnT3AXd/qWjx\nDOBwM1tiZgvMbBKwM9BX0KYP2AXoyi9390FgIDksFWkKMU7CtAC/c/cDgceBz5RpU4pOGklTifEH\n/Rfg3uT2HcDrgZWEPV7e1GTZKsLekfyez903V3rwzZv7BwH96KeRfsqKcTh3O/AuwtnNvwMceAC4\n0sxeAwwA+xLOiE4E3g/cCbwbWDLUg69du7EmnRYZqY6OHcuua6nlbAgz2xu4EuggfI3wDDAbuISw\nx9sAnODufWb2PuAsQgFe6u4/MrMxyf/fg3BC50R3X1lpm319G2r3hERGoKNjx3IfqWpbgDGoAKXR\nVCpAndQQiUgFKBKRClAkIn2pXUX9/f3kcj0V28yYsTutra116pE0OhVgFeVyPfzsxyfTOXlcyfW9\nT2/ksGOuoLt7jzr3TBqVCrDKOiePY+rO42N3Q0YJfQYUiUgFKBKRClAkIhWgSEQqQJGI6p4JU7B8\nNnC7u49J7isTRjInRiYMZrYDcDZhvh/KhJGsipEJA/BZ4DIgH1ehTBjJpLpnwpjZTOAN7n5TwWJl\nwkgmxfhjvgg4Lbldbp7UiDNhJk0ax3bbxRlruXbthCHbtLVNqDhDWrKlrgVoZlOAWcCPzKwF2MXM\nlgDnEuIK86YCv2JLJsyjaTNhYkZSPPPMc6na9PVtqENvpFFUesOtZwG2uPsqYGZ+gZn9wd0PNLNX\nU6VMGJHRpKYFWJwJY2anAPu7+9qkySCAu79gZmcTzpYOAF9w9w1mdj1wiJn9kiQTppb9Fam3mhag\nu98PvKnC+t0Lbt9E+L6wcP0AcFLNOigSmUbCiESkAhSJSAUoEpEKUCQiFaBIRCpAkYhUgCIRqQBF\nIlIBikSkAhSJSAUoElHdM2HMbBpwFfAqwoz44929V5kwkkUxMmG+BFzh7gcAPwXmKRNGsipGJszH\ngYXJ7T6gHWXCSEbVPRPG3Te6+0By/fdPAAtQJoxkVJQ/5qT4rgXucvclZnZsURNlwkgmxNqbXA24\nu385ub+KsMfLa5hMmDQX3YRw4U1lwkgpDZMJA68kYL/o7ucVrLsfmN+ImTC5XA8PLPgYU8pcdBNg\n1dMb4bjv1rFX0ixiZMK0ApuSNLRB4LfuflojZ8JMmTyO6V1DH16KDFfUTJiitsqEkczRSBiRiFSA\nIhGpAEUiUgGKRKQCFIlIBSgSkQpQJKKmG9icZujYjBm709oaZ7yoSKGmK8Bcrofl1/2I6e0dJdcv\nX9MHH/wA3d171LlnIn+t6QoQYHp7B91duwzdUCSyGJEUryVMRRoD/Bn4kLu/rEgKyaIYkRTnAZe5\n+/7AMuAkRVJIVsWIpDgAWJTcXgQcgiIpJKPqHkkBjHf3l5PbvRRFTyQUSSGZEPt7wHLREyOOpBAZ\nTWLsTTaY2Q7u/iIhemIlVYykeM1rxrJ+iA4MJ5dl7doJ/ClFu7a2dBN2lQkjhWIU4F3AUYQ0tKOA\nnwEPAFdWI5Ji/fpNQ3ZgOLksaXJehttOmTDZEi0TpkQkxanAbOCaJJ5iOXCNu/c3ciSFSK3EiqQ4\ntERbRVJI5uikhkhEKkCRiFSAIhGpAEUiUgGKRJSqAM3s+yWW3VH13ohkTMWvIZIpQqcCbzSzewtW\nbU8Ypyki26BiAbr7dWb2C+A64NyCVQPA4zXsl0gmDPlFvLuvBA4ws4lAG1sGSu8EPFPDvok0vVQj\nYczsEsKIlD62FOAgsHuN+iWSCWmHoh0EdCSTZUWkStIW4O+rVXxmNh74ATCJcDLnPOC3pMyJqUYf\nRBpF2gL8U3IW9D5COBIA7n7OCLZ5IvCEu3/OzHYB7ibM/bvc3Rea2fmEnJhrCTkxeyXbfNDMbnL3\nZ0ewTZGGlPaL+DXAz4EXCXuj/M9I9ALtye02wufK/YGbk2WVcmKUCSNNJe0e8EvV2qC732BmJ5rZ\n7wkTbo8AFg0jJ0akaaTdA24GXi74eYmtiyO15HPdCnffg5B69u2iJsPNiREZtVLtAd39lUI1s+0J\nhfPmEW7z7cAdyeM+amZTgeeHkRNTkTJhZDQZ9oz4JGbwdjM7k5EF5T4F7AP8xMymA88Rsl6OJoy4\nGSonpiJlwkij2eZMGDMrjoWYRtgjjcT3gKuSIW6twMmAAz8ws5MZIidmhNsUaUhp94D7FdweBNYD\nx4xkg+7+PDCnxKpUOTEizSTtZ8CPAJhZGzDo7mtr2iuRjEh7CLovYaTKjkCLma0Bjnf3h2rZOZFm\nl/ZriAuA97h7p7t3AMcCF9euWyLZkLYA+939sfwdd3+YgiFpIjIyaU/CDJjZUYSIeIDDGPlQNBFJ\npC3AU4HLCDHzA8AjwNxadUokK9Iegh4KvOjuk9y9Pfl/h9euWyLZkLYAjwfeV3D/UOCD1e+OSLak\nLcBWdy/8zDdQi86IZE3az4A3m9lS4JeEoj0YWFizXolkRKo9oLt/GTiLMFfvz8DH3f38WnZMJAtS\nz4Zw9/sIs9K3WTIn8NOEuYXnAI+iTBjJoLpfGyIZT3oOYXrRkcA/E4KZLnP3/YFlhEyYcYRMmIOA\nA4EzzGynevdXpJZiXCP+ncCd7r4R2AicYmY9wCnJ+kXAmcCTJJkwAGaWz4S5tf5dFqmNGAU4Axhv\nZv9FSNf+IjBOmTCSRTEKsIWQhvZeQjEuYeu8F2XCSGbEKMDVwFJ3HwB6zGwD8LIyYSSLYhTgYuBq\nM7uQsCecQMiAUSaMNKVKb7h1Pwvq7quAG4H/IZxQ+QTh0mcnmNk9hMj6a5Iw3nwmzGKUCSNNKMYe\nEHefD8wvWqxMGMkcXSNeJCIVoEhEKkCRiFSAIhGpAEUiUgGKRKQCFIlIBSgSkQpQJCIVoEhEKkCR\niKKMBQUws1cDjxHiKO5GmTCSQTH3gJ8H1iS3lQkjmRSlAM3MACNMR2oB9idkwZD8ewiwN0kmTDI1\nKZ8JI9I0Yu0BLwLmsSVmYrwyYSSL6v4Z0Mw+BNzj7n8MO8K/sk2ZMIqkkNEkxkmYI4DdkusNTgVe\nAp6rViaMIimk0VR6w617Abr7B/K3zewcIEfIe6lKJozIaBL7e8D8YaUyYSSTon0PCODuXyy4q0wY\nyZzYe0CRTFMBikSkAhSJSAUoEpEKUCQiFaBIRCpAkYhUgCIRqQBFIlIBikSkAhSJKMpY0OTquO8A\nWoELgAepcyZMf38/uVxPxTYzZuxerc2JlBRjQu4BwBvcfV8zawMeBn4OXO7uC83sfEImzLWETJi9\ngM3Ag2Z2k7s/W41+5HI9/OGHFzCtfWLJ9SvWrIPjz67GpkTKirEHvJcw1w/gWWA8IRPmlGTZIuBM\n4EmSTBgAM8tnwtxarY5Ma59Id1dbtR5OZNhiTMgdADYmdz9KKKjZWcqESXv429raWqceSSwxc0Hf\nA5xEmAf4VMGqumTCQKj0NO2qnQmzfn0vCxbOZXLH2JJtnu7bxGlz/5OZM2emeszY+vv7WbZsWcU2\n3d3dekMpIdZJmNnAZwh7vg1mtqHemTBppG03ksec3DGWrl3GV2wXMzsmzV4awp46l+vhjFtuY1xn\nV8k2G3tX880jD6e7e49qd3NUaKhMmCTj5ULgYHdflyy+i5AFswBlwjSEXK6H02+5nrGdHWXbbOrt\n45Ij5wAwrrOLCVNeW6/uNY0Ye8A5QDvwYzNrAQaBE4D/MLNTgOWETJh+M8tnwgygTJi6G9vZwYQp\nTfGxu2HFOAkzH5hfYpUyYSRzooYySf1pAEJjUQFmTC7XwydvvZKxXe0l129avYZLj/iXOvcqu1SA\nGTS2q50JUzpjd0PQYGyRqFSAIhHpELSBDefLcI0yGZ1UgA0sl+vh4ltOZmJn6SFrAOt6NzHvyCua\napRJlt54VIANbmLnWNqmlB+y1oxyuR7m3+ZM6ppWts3a1SuYezij/o1HBSgNaVLXNDqmdMfuRs2p\nACUTGnUKWMMXoJldDOxDGA/6KXd/KHKXZBTK5Xr4vwVPMrV915LrV675IxxX/0Pahi5AM/tH4HVJ\nfMUs4CrCrAgpMhqGmMXu49T2XZnR1ViHtQ1dgMDBwE8B3P0JM9vJzCbkYypki1yuh3+9/XzGdZbO\nuNnYu47L3vW5Ovdqa7lcD2feupTxnVNLrn++dyUXHTG8x4xd1Nuq0QtwZ6DwkPPpZNlTpZtn27jO\niYyfOil2Nyoa3zmVHadMr9rj5XI93HXzU3R1lH7M1X3Leee7q7a5qmv0AiyWKpZi+Zq+iuvyv6oV\na9aVbbdizTp2S26venpj2Xb59fmpqL0V2haue7qv/Mz9wnXreivP8C9cv7G3/PMpXLdp9Zqy7QrX\nbeot/zoWr9/Yu7rCtrese753Zdl2YV347axdvaLitsN6q9im2Mo1f6y4rp0QAbJs2e+HfKz8Z8Wh\n2g71mbJlcHBwyI3FYmbnAquSOYSY2TJgT3d/Pm7PRKqj0ceCLgaOBjCzvwVWqvikmTT0HhDAzL5C\nyA3tBz7h7o9G7pJI1TR8AYo0s0Y/BBVpaipAkYhUgCIRjbbvAVMzs+nAo2z5Ij+fQfq+wissFbUb\nA7wMfNXd707xmDsAn3b3pSXavg74FjCZcBm2pUnblyr090Z3//sUz2cMsD1wobv/tEz7Y4HvA7u4\n+zNl2uye9LEr6eN/A//m7i9U6ltyWYEzgEMKrumRb/cH4G2FY3bN7AHgMXc/qUQfbgPeCnzU3W9L\n87okX0/1uft3KrxG+d/3I+4+r6jdDODS5HmPIVww6LNJMnulx9s+uf8xdx8sarcMeIu7P5YsOwEY\ndPcfFD+nQk1bgIkn3P2g4bRL/igXmdmc/ItZoe1+wDnAYYUNzGwMsJBw1va+ZNklhMutfb5CP4Y6\nI1a47UnAw2Z2e/EfTuLYpA9HA1cUr0xCkRcCZ7j7L5Jl85K2Hy7XNzN7E/AF4KDC4iuwDDiG5E3K\nzHYFyg7PcffDzWyo6z4O50xhxd958rxvAuYVPe/vAScO9XhmdjVwHHBdUbvfEq51eeQw+qpD0GLu\n3gN8GTitTJPC0Tg7U/raLYcAv8sXX+Is4LyqdBJw97WEi5nuXLwuKc6ZwFcJfyylHBoeJvwRJo95\nMfA2M5tc6j+YWTtwDTAn2X4p9xPG8OYdDdxR8ckMPcIp1QiolA4BnizxvPdJnt9Q7gdKDW/5NfCc\nmR04nM40ewGO9Bf3a+BvyqwzM7vbzH4FXJT8FJsFPFK4wN1fLLPHGI5Xnk9yGNUGlBqz9X7g1uQ7\n0ylmVipffhbh4qjFHqf0H9j2hD3m9e7+ZIU+vgw8YmZvS+4fCfzVoeUw5V/zu81sCeFSBuUM9Tuv\n9LxLXY6q8DV/FfAe4Dcl2g0CnwPOH2L7W2n2Q1Azs7vZ8iI+4e4fS/H/diR88V9K4WGgATeY2VuS\n6x7mDRI+U1Vb/vmMATYBHy7abt5xbDnUvZlwPY5vFbUp18cWSj93A+YBnzKza919VYV+3gDMMbNV\nwDPAto5eKj4MPLdC28Lf+SBwp7t/tWD9GMo/71LFW/h4ewIXuPvNpTbs7svM7NdmNqfy09mi2Qsw\n7WfAYntR+l1yK+7uZrYJmEa4qMwr26XoENbMtgf2cPfHR9CfVx53qOdjZlOBvYFLw/sDYwlXIi4u\nwCeAUm9GbyBcnbjYo+7+XTPrBRaY2YGFJyIKDBIuOX4B4TUpeW0PM5sIPO/umwlFsbnS8xqGoV6j\nJ9hyNeZCrwe80uOZ2Y8p/doU+hLhkPtywtFARToELWpnZt2EM3zfTNG2jfAZrHiI/53ArmZ2RNJu\nDPA1wsmJbelvmudzLHC5u781+ZkFtJnZbkXt7gRmmNkrJ5DM7Azg3sKzxMXbdveFhOlgZfdCyaH2\nw4QLsC4q0+zbwHuTkyKzKP3Hv9W2Uxqq7WJgVonnvdTdS00TKXy8s4Cvmdmry7Vz917CHNZT03S2\n2QtwZuFnh+TfvSq0W0o4u/Vxdy93Ydx82yXALYQznVu9eyd7htnAKckp+HuBZ9290qETDH22L83Z\nwA8AVxctuyZZXraPZvYQ4TDzkym2fTpwTJJYUM4NwIoKl5T7AuGN7pfALe6+vEy74m0PpWLb5JD9\nMOCzZvaImf0v4TNvuYIZLPi/OeBG4N+H2O5FQKqLJWosqGSWmf0D8A13jxZz0ux7QJGy3P1XwAPJ\niZOjYvRBe0CRiLQHFIlIBSgSkQpQJCIVoEhEzT4SRoYhGemRH4jdQrgkwG6EaVdXJP8CnFZmpogM\nk86CSklmdhAw192PNbMbgcXufoWZvRG4uty8RRkeHYJmVDL6ZZ+C+3ea2ezkdgthNMeZyepZhMm6\nJHu+iWY2pc5dbkoqwOz6IWHaEmbWQSiyxcm6Y4AH3T0/xvU3wHuTtnsCU5If2UYqwOy6HshfNeFo\n4IaC2Q2nA5cVtD0TeLOZ3UOY6tRDmA4l20ifATPMzO4gDCz+OiGa4uFk8u497l5qcipm1gr0AdMr\nDLSWlLQHzLbrgI8Ck9w9P/9xX7a+IhVmdraZzU3uHg88pOKrDn0NkW0/IRxqfqVg2TTgL0Xtfghc\nb2YnEiaZlgptkhHQIahIRDoEFYlIBSgSkQpQJCIVoEhEKkCRiFSAIhGpAEUiUgGKRPT/gZQpCVCO\nCPEAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760d46a860>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760d5ff208>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nzgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760d695710>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nBVAkkAIoEui/ynivai6xg84AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760d7bfe10>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n1u4O5Y0AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760d8c5c18>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nT21RudPqLwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760d7bfac8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f760d955f60>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.factorplot('v3',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v22',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v24',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v30',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v31',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v47',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v52',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v56',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v66',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v71',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v74',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v75',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v79',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v91',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v107',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v110',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v112',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v113',data=trains,kind='count',size=3)\n",
    "sns.factorplot('v125',data=trains,kind='count',size=3)"
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
       "5663"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Do I have NaN values?\n",
    "trains.loc[:, trains.dtypes == np.object].isnull().sum().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# First transform object data type into categorical\n",
    "# Second change categorical string data into integers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# change np.object into np.categorical\n",
    "columns = trains.columns[(trains.dtypes == np.object)]  \n",
    "for i in range(len(columns)):\n",
    "    # convert to categorical\n",
    "    trains[columns[i]] = trains[columns[i]].astype('category')\n",
    "    # convert to integers (NaN == -1)\n",
    "    trains[columns[i]] = trains[columns[i]].cat.codes"
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
       "      <th>v3</th>\n",
       "      <th>v22</th>\n",
       "      <th>v24</th>\n",
       "      <th>v30</th>\n",
       "      <th>v31</th>\n",
       "      <th>v47</th>\n",
       "      <th>v52</th>\n",
       "      <th>v56</th>\n",
       "      <th>v66</th>\n",
       "      <th>v71</th>\n",
       "      <th>v74</th>\n",
       "      <th>v75</th>\n",
       "      <th>v79</th>\n",
       "      <th>v91</th>\n",
       "      <th>v107</th>\n",
       "      <th>v110</th>\n",
       "      <th>v112</th>\n",
       "      <th>v113</th>\n",
       "      <th>v125</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>2727</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>6</td>\n",
       "      <td>54</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>14</td>\n",
       "      <td>-1</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1256</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>6</td>\n",
       "      <td>66</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>20</td>\n",
       "      <td>15</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>1138</td>\n",
       "      <td>4</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>18</td>\n",
       "      <td>-1</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>250</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>7</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>-1</td>\n",
       "      <td>63</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "      <td>1305</td>\n",
       "      <td>4</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>7</td>\n",
       "      <td>-1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>19</td>\n",
       "      <td>15</td>\n",
       "      <td>88</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   v3   v22  v24  v30  v31  v47  v52  v56  v66  v71  v74  v75  v79  v91  v107  \\\n",
       "0   2  2727    2    2    0    2    6   54    2    2    1    3    4    0     4   \n",
       "1   2  1256    2    2    0    4    6   66    0    2    1    3    3    1     1   \n",
       "2   2  1138    4   -1    0    2    5   12    0    0    1    1    4    6     2   \n",
       "3   2   250    3    2    1    2    7   31    0    2    1    3    1    1     1   \n",
       "4   2  1305    4   -1    0    7    7   -1    2    2    1    3    2    6     2   \n",
       "\n",
       "   v110  v112  v113  v125  \n",
       "0     1    14    -1    21  \n",
       "1     0    20    15     6  \n",
       "2     1    18    -1     5  \n",
       "3     1     9    -1    63  \n",
       "4     0    19    15    88  "
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Third substitute in float data NaN by mean value \n",
    "imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)\n",
    "imp.fit(trains[columns[:]])\n",
    "X = trains[columns[:]]\n",
    "X.head()"
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
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>10</th>\n",
       "      <th>11</th>\n",
       "      <th>12</th>\n",
       "      <th>13</th>\n",
       "      <th>14</th>\n",
       "      <th>15</th>\n",
       "      <th>16</th>\n",
       "      <th>17</th>\n",
       "      <th>18</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>2727</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>6</td>\n",
       "      <td>54</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>14</td>\n",
       "      <td>15</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1256</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>6</td>\n",
       "      <td>66</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>20</td>\n",
       "      <td>15</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>1138</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>18</td>\n",
       "      <td>15</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>250</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>7</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>15</td>\n",
       "      <td>63</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "      <td>1305</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>7</td>\n",
       "      <td>31</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>19</td>\n",
       "      <td>15</td>\n",
       "      <td>88</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   0     1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  \\\n",
       "0   2  2727   2   2   0   2   6  54   2   2   1   3   4   0   4   1  14  15   \n",
       "1   2  1256   2   2   0   4   6  66   0   2   1   3   3   1   1   0  20  15   \n",
       "2   2  1138   4   2   0   2   5  12   0   0   1   1   4   6   2   1  18  15   \n",
       "3   2   250   3   2   1   2   7  31   0   2   1   3   1   1   1   1   9  15   \n",
       "4   2  1305   4   2   0   7   7  31   2   2   1   3   2   6   2   0  19  15   \n",
       "\n",
       "   18  \n",
       "0  21  \n",
       "1   6  \n",
       "2   5  \n",
       "3  63  \n",
       "4  88  "
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "objpreFeatures = pd.DataFrame(imp.transform(X))\n",
    "objpreFeatures.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Forth Transform each category into one boolean feature"
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
      "text/plain": [
       "array([[ 0.,  0.,  1., ...,  0.,  0.,  0.],\n",
       "       [ 0.,  0.,  1., ...,  0.,  0.,  0.],\n",
       "       [ 0.,  0.,  1., ...,  0.,  0.,  0.],\n",
       "       ..., \n",
       "       [ 0.,  0.,  1., ...,  0.,  0.,  0.],\n",
       "       [ 0.,  0.,  1., ...,  0.,  0.,  0.],\n",
       "       [ 0.,  0.,  1., ...,  0.,  0.,  1.]])"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "enc = preprocessing.OneHotEncoder()\n",
    "enc = enc.fit(objpreFeatures)\n",
    "x = enc.transform(objpreFeatures).toarray()\n",
    "x"
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
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>...</th>\n",
       "      <th>3279</th>\n",
       "      <th>3280</th>\n",
       "      <th>3281</th>\n",
       "      <th>3282</th>\n",
       "      <th>3283</th>\n",
       "      <th>3284</th>\n",
       "      <th>3285</th>\n",
       "      <th>3286</th>\n",
       "      <th>3287</th>\n",
       "      <th>3288</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 3289 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   0     1     2     3     4     5     6     7     8     9     ...   3279  \\\n",
       "0     0     0     1     0     0     0     0     0     0     0  ...      0   \n",
       "1     0     0     1     0     0     0     0     0     0     0  ...      0   \n",
       "2     0     0     1     0     0     0     0     0     0     0  ...      0   \n",
       "3     0     0     1     0     0     0     0     0     0     0  ...      0   \n",
       "4     0     0     1     0     0     0     0     0     0     0  ...      0   \n",
       "\n",
       "   3280  3281  3282  3283  3284  3285  3286  3287  3288  \n",
       "0     0     0     0     0     0     0     0     0     0  \n",
       "1     0     0     0     0     0     0     0     0     0  \n",
       "2     0     0     0     0     0     0     0     0     0  \n",
       "3     0     0     0     0     0     0     0     0     0  \n",
       "4     0     0     0     0     0     0     0     0     1  \n",
       "\n",
       "[5 rows x 3289 columns]"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "intFeatures = pd.DataFrame(x)\n",
    "intFeatures.head()"
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
   "version": "3.4.3+"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}