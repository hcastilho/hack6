import os

import category_encoders
import numpy as np
import pandas as pd
from sklearn import model_selection, pipeline, preprocessing, ensemble
from sklearn.externals import joblib

from verification.data import datasets
from hack6.modelling import ForceDType, hyper_fit, MODEL_DIR

newdata = pd.concat((
    datasets['original'].copy(),
    datasets['targets'].drop('proba', axis=1).copy(),
)).drop('id', axis=1)

features = newdata.drop('target', axis=1)
target = newdata['target']

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    features,
    target,
    test_size=0.4,
    random_state=0
)

cv = model_selection.ShuffleSplit(n_splits=5,
                                  test_size=0.3,
                                  random_state=1)

pipe = pipeline.Pipeline([
    ('dtype', ForceDType()),
    ('onehot', category_encoders.OneHotEncoder(handle_unknown='ignore')),
    ('inputer', preprocessing.Imputer(strategy='mean')),
    ('gb', ensemble.GradientBoostingClassifier()),
])
params = {
    # 'gb__loss': ['deviance', 'exponential'],
    # 'gb__learning_rate': np.arange(0.05, 0.2, 0.05),
    'gb__n_estimators': range(1, 400),
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
search_cv = hyper_fit(pipe, params, cv, X_train, y_train, n_iter=20, n_jobs=-1)
pipe = search_cv.best_estimator_
joblib.dump(pipe, '%s.pkl' % os.path.join(MODEL_DIR, 'new'))
print("Trained with new data:", pipe.score(X_test, y_test))

best = joblib.load('%s.pkl' % os.path.join(MODEL_DIR, 'best'))
print("Old model:", best.score(X_test, y_test))
