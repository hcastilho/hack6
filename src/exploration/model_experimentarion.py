import pprint

import pandas as pd
import numpy as np
import scipy as sp
from sklearn import discriminant_analysis
from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import linear_model
from sklearn import model_selection
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn import tree

from exploration.data_exploration import X_train, y_train, X_test, y_test
from exploration import modelling
from exploration.modelling import cv


#####################
# Run once
#####################
# model = ensemble.RandomForestClassifier()
# proba, y_pred, score = modelling.run_once(
#     model, X_train, y_train, X_test, y_test)
# print("ROC AUC: " + str(score))


############################
# Cross Validation
############################
# model = ensemble.RandomForestClassifier()
# modelling.run_cross(model, X_train, y_train, X_test, y_test, cv)


############################
# Hyper Tuning
############################

results = {}


def format_result(result):
    res = ('\n{result[name]}'
           '\nbest_params_: {result[rs].best_params_}'
           '\nscore: {result[score]}').format(result=result)
    return res


def full_table(results):
    txt = ''
    for res in results.values():
        txt += ('\subsubsection*{{{}}}\n'
                '\\begin{{lstlisting}}\n{}\n\\end{{lstlisting}} \n\n'
                .format(res['model'].__name__,
                        pprint.pformat(res['rs'].get_params())
                        ))
    return txt


def small_table(results):
    txt = ''
    for res in results.values():
        txt += '{} & {:.4f} \\\\\n'.format(res['model'].__name__, res['score'])
    return txt


######################################
# 1. Supervised learning
# http://scikit-learn.org/stable/supervised_learning.html
######################################

######################################
# 1.1. Generalized Linear Models
######################################
# For regression

######################################
# 1.2. Linear and Quadratic Discriminant Analysis
# http://scikit-learn.org/stable/modules/lda_qda.html
# These classifiers are attractive because they have closed-form solutions
#  that can be easily computed, are inherently multiclass, have proven to
#  work well in practice and have no hyperparameters to tune.
######################################
model = {
    'name': 'quadratic_discriminant_analysis',
    'model': discriminant_analysis.QuadraticDiscriminantAnalysis,
    'params': {
        # 'priors': None,
        # 'reg_param': 0.0,
        # 'store_covariance': False,
        #  'tol': 0.0001,
        # 'store_covariances': None
    },
}
# Collinear variables (the categories) cause problems since they
# are not completely independent
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
results[result['name']] = result
print(format_result(result))

######################################
# 1.3. Kernel ridge regression
######################################
# regression

######################################
# 1.4. Support Vector Machines
# http://scikit-learn.org/stable/modules/svm.html
######################################
model = {
    'name': 'svc',
    'model': svm.SVC,
    'params': {
        #  'C': 1.0,
        # 'kernel': 'rbf',
        # 'degree': 3,
        # 'gamma': 'auto',
        # 'coef0': 0.0,
        # 'shrinking': True,
        # 'probability': False,
        # make predict_proba available
        'probability': [True],
        # 'tol': 0.001,
        # 'cache_size': 200,
        # 'class_weight': None,
        # 'verbose': False,
        # 'max_iter': -1,
        # 'decision_function_shape': 'ovr',
        # 'random_state': None
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
results.append(result)
print(format_result(result))

######################################
# 1.5. Stochastic Gradient Descent
# http://scikit-learn.org/stable/modules/sgd.html
######################################
model = {
    'name': 'sdg',
    'model': linear_model.SGDClassifier,
    'params': {
        # hinge no predict_proba available
        # 'loss': 'hinge',
        'loss': ['log'],
        # 'penalty': 'l2',
        # 'alpha': sp.stats.uniform(0.00001, 0.001),  # 0.0001
        # 'l1_ratio': 0.15,
        # 'fit_intercept': True,
        # 'max_iter': None,
        # 'tol': None,
        # 'shuffle': True,
        # 'verbose': 0,
        # 'epsilon': 0.1,
        # 'n_jobs': 1,
        # 'random_state': None,
        # 'learning_rate': 'optimal',
        # 'eta0': 0.0,
        # 'power_t': 0.5,
        # 'class_weight': None,
        # 'warm_start': False,
        # 'average': False,
        # 'n_iter': None
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
results.append(result)
# Rouding error on the output of predict_proba
print(format_result(result))

######################################
# 1.6. Nearest Neighbors
# http://scikit-learn.org/stable/modules/neighbors.html
######################################
model = {
    'name': 'kneighbors',
    'model': neighbors.KNeighborsClassifier,
    'params': {
        # 'n_neighbors': 5,
        # 'weights': 'uniform',
        # 'algorithm': 'auto',
        #  'leaf_size': 30,
        # 'p': 2,
        # 'metric': 'minkowski',
        # 'metric_params': None,
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
results.append(result)
print(format_result(result))

######################################
# 1.7. Gaussian Processes
# http://scikit-learn.org/stable/modules/gaussian_process.html
######################################
model = {
    'name': 'gpc',
    'model': gaussian_process.GaussianProcessClassifier,
    'params': {
        # 'kernel': None,
        # 'optimizer': 'fmin_l_bfgs_b',
        # 'n_restarts_optimizer': 0,
        # 'max_iter_predict': 100,
        # 'warm_start': False,
        # 'copy_X_train': True,
        # 'random_state': None,
        # 'multi_class': 'one_vs_rest',
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
results.append(result)
print(format_result(result))

######################################
# 1.8. Cross decomposition
######################################
# linear relations between datasets, regression

######################################
# 1.9. Naive Bayes
# http://scikit-learn.org/stable/modules/naive_bayes.html
######################################
model = {
    'name': 'gnb',
    'model': naive_bayes.GaussianNB,
    'params': {
        # 'priors': None,
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
results.append(result)
print(format_result(result))

######################################
# 1.10. Decision Trees
# http://scikit-learn.org/stable/modules/tree.html
######################################
model = {
    'name': 'tree',
    'model': tree.DecisionTreeClassifier,
    'params': {
        # 'criterion': 'gini',
        # 'splitter': 'best',
        # 'max_depth': None,
        # 'min_samples_split': 2,
        # 'min_samples_leaf': 1,
        # 'min_weight_fraction_leaf': 0.0,
        # 'max_features': None,
        # 'random_state': None,
        # 'max_leaf_nodes': None,
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
        # 'class_weight': None,
        # 'presort': False,
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
results.append(result)
print(format_result(result))

######################################
# 1.11. Ensemble methods
######################################

######################################
# 1.11.1. Bagging meta-estimator
######################################
model = {
    'name': 'bagging',
    'model': ensemble.BaggingClassifier,
    'params': {
        # 'base_estimator': None,  # Decision tree
        # 'n_estimators': 10,
        # 'max_samples': 1.0,
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
results.append(result)
print(format_result(result))

######################################
# 1.11.2. Forests of randomized trees
######################################

# 1.11.2.1. Random Forests
model = {
    'name': 'random_forest',
    'model': ensemble.RandomForestClassifier,
    'params': {
        # 'n_estimators': 10,
        # 'criterion': 'gini',
        # 'max_depth': None,
        # 'min_samples_split': 2,
        # 'min_samples_leaf': 1,
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
results.append(result)
print(format_result(result))

# 1.11.2.2. Extremely Randomized Trees
model = {
    'name': 'extra_trees',
    'model': ensemble.ExtraTreesClassifier,
    'params': {
        # 'n_estimators': 10,
        # 'criterion': 'gini',
        # 'max_depth': None,
        # 'min_samples_split': 2,
        # 'min_samples_leaf': 1,
        # 'min_weight_fraction_leaf': 0.0,
        # 'max_features': 'auto',
        # 'max_leaf_nodes': None,
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
        # 'bootstrap': False,
        # 'oob_score': False,
        # 'n_jobs': 1,
        # 'random_state': None,
        # 'verbose': 0,
        # 'warm_start': False,
        # 'class_weight': None
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
results.append(result)
print(format_result(result))

######################################
# 1.11.3. AdaBoost
######################################
model = {
    'name': 'adaboost',
    'model': ensemble.AdaBoostClassifier,
    'params': {
        # 'base_estimator': None,  # DecisionTreeClassifier
        # 'n_estimators': 50,
        # 'learning_rate': 1.0,
        # 'algorithm': 'SAMME.R',
        # 'random_state': None,
    },
}
result = modelling.run_hyper(model, X_train, y_train, X_test, y_test, cv)
results.append(result)
print(format_result(result))

######################################
# 1.11.4. Gradient Tree Boosting
######################################
model = {
    'name': 'gradient_boosting',
    'model': ensemble.GradientBoostingClassifier,
    'params': {
        # 'loss': 'deviance',
        # 'learning_rate': 0.1,
        # 'n_estimators': 100,
        # 'subsample': 1.0,
        # 'criterion': 'friedman_mse',
        # 'min_samples_split': 2,
        # 'min_samples_leaf': 1,
        # 'min_weight_fraction_leaf': 0.0,
        # 'max_depth': 3,
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
results.append(result)
print(format_result(result))

######################################
# 1.11.5. Voting Classifier
######################################
model = {
    'name': 'voting',
    'model': ensemble.VotingClassifier,
    'model_init_params': {
        'estimators': [
            ('clf1', linear_model.LogisticRegression()),
            ('clf2', ensemble.RandomForestClassifier()),
            ('clf3', naive_bayes.GaussianNB()),
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
results.append(result)
print(format_result(result))

print(full_table(results))
print(small_table(results))
