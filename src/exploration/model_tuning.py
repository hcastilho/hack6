import os

import numpy as np
from sklearn import discriminant_analysis
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import tree
from sklearn.externals import joblib

from exploration import modelling
from exploration.data_exploration import X_train, y_train, X_test, y_test
from exploration.modelling import cv

try:
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except:
    BASE_DIR = os.getcwd()

MODEL_DIR = os.path.join(BASE_DIR, 'data/models')


def format_result(result, xtest, ytest):
    res = ('\n{result[name]}'
           '\nbest_params_: {result[rs].best_params_}'
           '\nscore: {score}').format(
        result=result,
        params=(
            result['rs'].best_params_
                if hasattr(result['rs'], 'best_params_')
                else result['estimator'].get_params()
        ),
        score=result['estimator'].score(xtest, ytest)
    )
    return res


def load_models():
    results = {}
    for fname in os.listdir(MODEL_DIR):
        model_name = fname.split('.')[0]
        fpath = os.path.join(MODEL_DIR, fname)
        model = joblib.load(fpath)
        results[model_name] = {'estimator': model}
    return results


def save_models(results):
    for name, res in results.items():
        print('%s.pkl' % os.path.join(MODEL_DIR, name))
        if 'estimator' in res:
            joblib.dump(res['estimator'],
                        '%s.pkl' % os.path.join(MODEL_DIR, name))
        else:
            joblib.dump(res['rs'].best_estimator_,
                        '%s.pkl' % os.path.join(MODEL_DIR, name))


def get_best(results, xtest, ytest):
    best = None
    for r in results.values():
        model = r['estimator']
        if best is None:
            best = model
        elif (model.score(xtest, ytest)
              > best.score(xtest, ytest)):
            best = model
    return best


def replace_if_better(res, results, xtest, ytest):
    if res['name'] not in results:
        results[res['name']] = res

    elif (res['estimator'].score(xtest, ytest)
          > results[res['name']]['estimator'].score(xtest, ytest)):
        results[res['name']] = res


# results = {}
results = load_models()

#############################
# GradientBoostingClassifier
##############################
model = {
    'name': 'gradient_boosting',
    'model': ensemble.GradientBoostingClassifier,
    'params': {
        'loss': ['deviance', 'exponential'],
        # 'learning_rate': np.arange(0.05, 0.2, 0.05),
        'n_estimators': range(1, 400),
        # 'subsample': 1.0,
        # 'criterion': 'friedman_mse',
        'min_samples_split': range(1, 400),
        # 'min_samples_leaf': 1,
        # 'min_weight_fraction_leaf': 0.0,
        # 'max_depth': range(1, 20),
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
        # 'init': None,
        # 'random_state': None,
        # 'max_features': None,
        # 'verbose': 0,
        # 'max_leaf_nodes': None,
        # 'warm_start': False,
        # 'presort': 'auto'
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
replace_if_better(result, results, X_test, y_test)
print(format_result(result, X_test, y_test))

##############################
# AdaBoostClassifier
##############################
model = {
    'name': 'adaboost',
    'model': ensemble.AdaBoostClassifier,
    'params': {
        'base_estimator': [
            None,
            ensemble.RandomForestClassifier(),
            ensemble.RandomForestClassifier(max_depth=2),
            tree.DecisionTreeClassifier(),
            tree.DecisionTreeClassifier(max_depth=2),
            tree.DecisionTreeClassifier(max_depth=4),
            tree.DecisionTreeClassifier(criterion='entropy', max_depth=2),
            tree.DecisionTreeClassifier(criterion='entropy', max_depth=4),
        ],
        'n_estimators': range(1, 500),
        # 'learning_rate': np.arange(.5, 2.0, .2),
        # 'algorithm': 'SAMME.R',
        # 'random_state': None,
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
replace_if_better(result, results, X_test, y_test)
print(format_result(result, X_test, y_test))

##############################
# BaggingClassifier
##############################
model = {
    'name': 'bagging',
    'model': ensemble.BaggingClassifier,
    'params': {
        'base_estimator': [
            None,
            tree.DecisionTreeClassifier(),
            tree.DecisionTreeClassifier(max_depth=2),
            tree.DecisionTreeClassifier(max_depth=4),
            tree.DecisionTreeClassifier(criterion='entropy', max_depth=2),
            tree.DecisionTreeClassifier(criterion='entropy', max_depth=4),
            ensemble.RandomForestClassifier(),
            ensemble.RandomForestClassifier(max_depth=2),
        ],
        'n_estimators': range(1, 500),
        # 'max_samples': [1.0, .5],
        # 'max_features': 1.0,
        # 'bootstrap': True,
        # 'bootstrap_features': False,
        # 'oob_score': False,
        # 'warm_start': False,
        # 'n_jobs': 1,
        # 'random_state': None,
        # 'verbose': 0
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
replace_if_better(result, results, X_test, y_test)
print(format_result(result, X_test, y_test))

##############################
# VotingClassifier
##############################
model = {
    'name': 'voting',
    'model': ensemble.VotingClassifier,
    'model_init_params': {
        'estimators': [
            # ('clf1', linear_model.LogisticRegression()),
            # ('clf2', discriminant_analysis.QuadraticDiscriminantAnalysis()),
            # ('clf3', ensemble.AdaBoostClassifier()),
            # ('clf4', ensemble.RandomForestClassifier(max_depth=2)),
            ('clf7', tree.DecisionTreeClassifier(max_depth=2)),
            ('clf5', ensemble.GradientBoostingClassifier()),
            # ('clf6', naive_bayes.GaussianNB()),
        ],
        'voting': 'soft',  # otherwise no probas
    },
    'params': {
        # 'voting': 'hard',
        # 'weights': None,
        # 'flatten_transform': None
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
replace_if_better(result, results, X_test, y_test)
print(format_result(result, X_test, y_test))

##############################
# RandomForestClassifier
##############################
model = {
    'name': 'random_forest',
    'model': ensemble.RandomForestClassifier,
    'params': {
        'n_estimators': range(1, 400),
        # 'criterion': ['gini', 'entropy'],
        # 'max_depth': range(1, 400),
        'min_samples_split': range(1, 400),
        # 'min_samples_leaf': range(1, 400),
        # 'min_weight_fraction_leaf': 0.0,
        # 'max_features': 'auto',
        # 'max_leaf_nodes': None,
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
        # 'bootstrap': True,
        # 'oob_score': False,
        # 'random_state': None,
        # 'verbose': 0,
        # 'warm_start': False,
        # 'class_weight': None

    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
replace_if_better(result, results, X_test, y_test)
print(format_result(result, X_test, y_test))

save_models(results)
best = get_best(results, X_test, y_test)

