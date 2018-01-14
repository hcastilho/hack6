import os

import multiprocessing
from sklearn import model_selection
from sklearn.externals import joblib

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


def replace_if_better(best, new, x_test, y_test):
    if best is None:
        new_score = new.score(x_test, y_test)
        joblib.dump(new, '%s.pkl' % os.path.join(MODEL_DIR, 'best'))
        print("Score:", new_score)
        return new

    best_score = best.score(x_test, y_test)

    new_score = new.score(x_test, y_test)

    print("Score:", new_score)
    print("Best:", best_score)

    if new_score > best_score:
        print("!!!!!!!!!!!!")
        print("!! BETTER !!")
        print("!!!!!!!!!!!!")
        joblib.dump(new, '%s.pkl' % os.path.join(MODEL_DIR, 'best'))
        return new

    return best
