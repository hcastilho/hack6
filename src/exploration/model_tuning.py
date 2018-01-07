import numpy as np
from sklearn import ensemble
from sklearn import tree

from exploration import modelling
from exploration.data_exploration import X_train, y_train, X_test, y_test
from exploration.modelling import cv
from exploration.utils import (load_models,
                               save_models,
                               get_best,
                               replace_if_better,
                               sort_results,
                               )


def format_result(result):
    res = ('\n{result[name]}'
           '\nbest_params_: {result[rs].best_params_}'
           '\nscore: {result[score]}').format(
        result=result,
    )
    return res


def format_table(results):
    results = sort_results(results)
    txt = ''
    for res in results:
        txt += '{} & {:.4f} \\\\\n'.format(
            res['model'].__name__,
            res['score'])
    return txt

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
        'learning_rate': np.arange(0.05, 0.2, 0.05),
        'n_estimators': range(1, 400),
        # 'subsample': 1.0,
        'criterion': ['friedman_mse', 'mse', 'mae'],
        'min_samples_split': range(1, 400),
        'min_samples_leaf': range(1, 400),
        # 'min_weight_fraction_leaf': 0.0,
        'max_depth': range(1, 20),
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
        # 'init': None,
        # 'random_state': None,
        'max_features': [None, 'sqrt', 'log2'],
        # 'verbose': 0,
        # 'max_leaf_nodes': None,
        'warm_start': [False, True],
        # 'presort': 'auto'
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
replace_if_better(result, results)
print(format_result(result))

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
        'learning_rate': np.arange(.05, 2.0, .2),
        'algorithm': ['SAMME.R', 'SAMME'],
        # 'random_state': None,
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
replace_if_better(result, results)
print(format_result(result))

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
        'max_samples': np.arange(.5, 1.0, .1),
        'max_features': np.arange(.5, 1.0, .1),
        # 'bootstrap': True,
        # 'bootstrap_features': False,
        # 'oob_score': False,
        # 'warm_start': False,
        # 'random_state': None,
        # 'verbose': 0
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
replace_if_better(result, results)
print(format_result(result))

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
replace_if_better(result, results)
print(format_result(result))

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
replace_if_better(result, results)
print(format_result(result))

save_models(results)

best = get_best(results)
print(format_result(best))

print('\n\n')
print(format_table(results))

