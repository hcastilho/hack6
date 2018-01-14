import os

import category_encoders
import pandas as pd
import numpy as np
from sklearn import pipeline, model_selection
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import neural_network
from sklearn.externals import joblib

from hack6.modelling import (DATA_DIR,
                             MODEL_DIR,
                             hyper_fit,
                             replace_if_better,
                             ForceDType,
                             )


##########################
# Setup
##########################
dataset = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

dataset = dataset.set_index('id')

# dataset = dataset.drop(['earned dividends', 'gender'], axis=1)

try:
    best = joblib.load('%s.pkl' % os.path.join(MODEL_DIR, 'pipe'))
except FileNotFoundError:
    best = None

features = dataset.drop(['target'], axis=1)
target = dataset['target']
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    features,
    target,
    test_size=0.4,
    random_state=0
)

cv = model_selection.ShuffleSplit(n_splits=5,
                                  test_size=0.3,
                                  random_state=1)

# joblib.dump(best, '%s.pkl' % os.path.join(MODEL_DIR, 'pipe'))
# joblib.dump(X_train.columns.tolist(),
#             '%s.pkl' % os.path.join(MODEL_DIR, 'columns'))
# joblib.dump(X_train.dtypes, '%s.pkl' % os.path.join(MODEL_DIR, 'dtypes'))


##########################
# GradientBoostClassifier
##########################
pipe = pipeline.Pipeline([
        ('dtype', ForceDType()),
        ('onehot', category_encoders.OneHotEncoder(handle_unknown='ignore')),
        ('inputer', preprocessing.Imputer(strategy='mean')),
        ('gb', ensemble.GradientBoostingClassifier()),
    ])
params = {
    # 'gb__loss': ['deviance', 'exponential'],
    # 'gb__learning_rate': np.arange(0.05, 0.2, 0.05),
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
}
search_cv = hyper_fit(pipe, params, cv, X_train, y_train, n_iter=1, n_jobs=-1)
pipe = search_cv.best_estimator_
best = replace_if_better(best, pipe, X_test, y_test)


##########################
# MPLClassifier
##########################
pipe = pipeline.Pipeline([
    ('onehot', category_encoders.OneHotEncoder(handle_unknown='ignore')),
    ('inputer', preprocessing.Imputer(strategy='mean')),
    ('model', neural_network.MLPClassifier()),
])
params = {
    'model__hidden_layer_sizes': [
        (100,),
        (100, 200, 100),
        (50, 100, 50, 20),
        (200, 100, 50, 10),
    ],
    # 'model__activation': ['logistic', 'tanh', 'relu'],
    'model__activation': ['logistic', 'relu'],
    'model__solver': ['lbfgs', 'sgd', 'adam'],
    # 'model__alpha': 0.0001,
    # 'model__batch_size': 'auto',
    # 'model__learning_rate': 'constant',
    # 'model__learning_rate_init': 0.001,
    # 'model__power_t': 0.5,
    # 'model__max_iter': 200,
    # 'model__shuffle': True,
    # 'model__random_state': None,
    # 'model__tol': 0.0001,
    # 'model__verbose': False,
    # 'model__warm_start': False,
    # 'model__momentum': 0.9,
    # 'model__nesterovs_momentum': True,
    # 'model__early_stopping': False,
    # 'model__validation_fraction': 0.1,
    # 'model__beta_1': 0.9,
    # 'model__beta_2': 0.999,
    # 'model__epsilon': 1e-08,
}
search_cv = hyper_fit(pipe, params, cv, X_train, y_train, n_iter=40, n_jobs=-1)
pipe = search_cv.best_estimator_
best = replace_if_better(best, pipe, X_test, y_test)
