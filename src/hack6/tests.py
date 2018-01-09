import os

import category_encoders
import pandas as pd
import numpy as np
from sklearn import pipeline, model_selection
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.externals import joblib

from hack6.modelling import DATA_DIR, MODEL_DIR, hyper_fit

dataset = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

# TODO test with and without
# dataset = dataset.drop(['earned dividends', 'gender'], axis=1)

features = dataset.drop(['target'], axis=1)
target = dataset['target']
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    features,
    target,
    test_size=0.4,
    random_state=0
)

# pipeline_ = pipeline.make_pipeline(
#     category_encoders.OneHotEncoder(handle_unknown='ignore'),
#     preprocessing.Imputer(strategy='mean'),
#     ensemble.GradientBoostingClassifier(),
# )
# pipeline.fit(X_train, y_train)
pipe = pipeline.Pipeline([
        ('onehot', category_encoders.OneHotEncoder(handle_unknown='ignore')),
        ('inputer', preprocessing.Imputer(strategy='mean')),
        ('gb', ensemble.GradientBoostingClassifier()),
    ])
params = {
        'gb__loss': ['deviance', 'exponential'],
        'gb__learning_rate': np.arange(0.05, 0.2, 0.05),
        # 'gb__n_estimators': range(1, 400),
        # 'gb__subsample': 1.0,
        # 'gb__criterion': ['friedman_mse', 'mse', 'mae'],
        # 'gb__min_samples_split': range(1, 400),
        # 'gb__min_samples_leaf': range(1, 400),
        # 'gb__min_weight_fraction_leaf': 0.0,
        # 'gb__max_depth': range(1, 20),
        # 'gb__min_impurity_decrease': 0.0,
        # 'gb__min_impurity_split': None,
        # 'gb__init': None,
        # 'gb__random_state': None,
        # 'gb__max_features': [None, 'sqrt', 'log2'],
        # 'gb__verbose': 0,
        # 'gb__max_leaf_nodes': None,
        # 'gb__warm_start': [False, True],
        # 'gb__presort': 'auto'
    },
cv = model_selection.ShuffleSplit(n_splits=5,
                                  test_size=0.3,
                                  random_state=1)
search_cv = hyper_fit(pipe, params, cv, X_train, y_train)
pipeline_ = search_cv.best_estimator_

joblib.dump(X_train.columns.tolist(),
            '%s.pkl' % os.path.join(MODEL_DIR, 'columns'))
joblib.dump(pipe, '%s.pkl' % os.path.join(MODEL_DIR, 'pipe'))
joblib.dump(X_train.dtypes, '%s.pkl' % os.path.join(MODEL_DIR, 'dtypes'))
