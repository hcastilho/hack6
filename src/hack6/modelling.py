import os

import multiprocessing
from sklearn import model_selection

try:
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except:
    BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'data/models')


def hyper_fit(pipeline, params, cv, xtrain, ytrain,
        n_iter=20, scoring='roc_auc', n_jobs=None, verbose=1):

    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count() - 1

    param_grid = model_selection.ParameterGrid(params)
    grid_size = len(param_grid)
    if grid_size < n_iter:
        # If we can exaust search space use GridSearchCV
        rs = model_selection.GridSearchCV(
            pipeline,
            params,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    else:
        rs = model_selection.RandomizedSearchCV(
            pipeline,
            params,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    rs.fit(xtrain, ytrain)
    return rs
