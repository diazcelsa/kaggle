class NullToNaNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return X.fillna(np.nan)

myshit = NaNToValueTransformer()
myothershit = IntCategorizer()

myshit = myshit.fit(x_train)
x_train_aftershit = myshit.tranform(x_train)
myothershit = myothershit.fit(x_train_aftershit)
x_train_afterothershit = myothershit.tranform(x_train)

x_test_aftershit = myshit.tranform(x_test)
x_test_afterothershit = myothershit.tranform(x_test_aftershit)

mypipeline = make_pipeline(NaNToValueTransformer(),IntCategorizer())

x_train_afterothershit = mypipeline.fit(x_test)

x_test_afterothershit = mypipeline.transform(x_train)


    
class NaNToValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,columns=None,value=0):
        self.value = value
        self.columns = columns

    def get_feature_names(self):
        return self.columns

    def fit(self, X, y=None, **fit_params):
        if self.columns is None:
            self.columns = X.columns.tolist()
            self.values = [self.value]*len(self.columns)
        else:
            self.values = [self.value]
        return self

    def transform(self, X, **transform_params):
        X_ = X.copy()
        for col,val in zip(self.columns,self.values):
            X_.loc[All,[col]] = X_.loc[All,[col]].applymap(lambda x: val if pd.isnull(x) else x)
        return X_


class IntCategorizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, boolean=False):
        self.columns = columns
        self.boolean = boolean

    def fit(self, X, y=None, **fit_params):
        if self.columns is None:
            self.columns = X.columns
        self.d = {}
        self.fnames = []
        for col in self.columns:
            w = X[col].notnull()
            uniq = X.ix[w, col].unique()
            names = list(map(lambda f: col+'_IS_'+f.replace(' ','_'),uniq))
            if not self.boolean:
                self.fnames.extend(names)
            else:
                self.fnames.extend([names[-1]])
            vals = enumerate(uniq, 1)
            vals = [(x, y) for (y, x) in vals]
            self.d[col] = defaultdict(lambda: np.nan, vals)
        return self

    def get_feature_names(self):
        return self.fnames

    def transform(self, X, **transform_params):
        X_ = X.copy()
        for col in self.columns:
            w = X_[col].notnull()
            X_.ix[w, col] = X_.ix[w, col].map(lambda x: self.d[col][x])
        return X_
    
    
class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,columns=None,add_feature_names=False, to_numeric=False):
        self.columns = columns
        self.add_feature_names = add_feature_names
        self.to_numeric = to_numeric

    def fit(self, X, y=None, **fit_params):
        if self.columns is None:
            self.columns = X.columns
        return self

    def get_feature_names(self):
        return self.columns

    def transform(self, X, **transform_params):
        if self.to_numeric:
            X = X.loc[All,self.columns].apply(pd.to_numeric,errors='coerce')
#            X = X.loc[All,self.columns].convert_objects(convert_numeric=True)
        else:
            X = X.loc[All,self.columns]
        return X


    
    
def simple_classifier(Classifier):
    '''
    Returns an estimator that estimates the return probability of order
    positions which uses only the information available at the time shipping.
    '''
    pipeline = make_pipeline(
        NullToNaNTransformer(),
        make_union(
            make_pipeline(
                ColumnExtractor(columns=['a','b']),
                Imputer(strategy='median')
            ),
            make_pipeline(
                ColumnExtractor(columns=['c']),
                NaNToValueTransformer(value = 'unkown'),
                IntCategorizer(),
                Imputer(strategy='most_frequent'),
                OneHotEncoder()
            )
        ),
        Classifier()
        )
    return pipeline

pipe = simple_classifier(RandomForestClassifier)
pipe.fit(x_train,y_train)

y_predicted = pip.predict__proba(x_test)
