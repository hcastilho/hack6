import os

import pandas as pd
import numpy as np
import multiprocessing
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin

try:
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except:
    BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'data/models')


class ForceDType(BaseEstimator, TransformerMixin):

    def transform(self, X):
        X.loc[:, 'birth date'] = pd.to_datetime(
            X['birth date']).map(lambda x: x.timestamp())

        X.loc[:, 'job type'] = X['job type'].astype(object)
        X.loc[:, 'school level'] = X['school level'].astype(object)
        X.loc[:, 'domestic status'] = X['domestic status'].astype(object)
        X.loc[:, 'profession'] = X['profession'].astype(object)
        X.loc[:, 'domestic relationship type'] = X[
            'domestic relationship type'].astype(object)
        X.loc[:, 'ethnicity'] = X['ethnicity'].astype(object)
        X.loc[:, 'gender'] = X['gender'].astype(object)

        try:
            X.loc[:, 'earned dividends'] = X[
                'earned dividends'].astype(np.float64)
        except (ValueError, TypeError):
            X.loc[:, 'earned dividends'] = np.nan

        try:
            X.loc[:, 'interest earned'] = X[
                'interest earned'].astype(np.float64)
        except (ValueError, TypeError):
            X.loc[:, 'interest earned'] = np.nan

        try:
            X.loc[:, 'monthly work'] = X['monthly work'].astype(np.float64)
        except (ValueError, TypeError):
            X.loc[:, 'monthly work'] = np.nan

        X.loc[:, 'country of origin'] = X['country of origin'].astype(object)

        return X

    def fit(self, *args, **kwargs):
        return self


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
        print("Storing first")
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
