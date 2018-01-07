import multiprocessing
from copy import deepcopy

from sklearn import metrics
from sklearn import model_selection

# Use all but one cpus
n_jobs = multiprocessing.cpu_count() - 1


############################
# Cross Validation Split
############################
cv = model_selection.ShuffleSplit(n_splits=5,
                                  test_size=0.3,
                                  random_state=1)


def run_once(model, xtrain, ytrain, xtest, ytest):
    """Fit a model return results"""
    model.fit(xtrain, ytrain)
    proba = model.predict_proba(xtest)
    y_pred = model.predict(xtest)
    score = metrics.roc_auc_score(ytest, y_pred[:, 1])

    return proba, y_pred, score


def run_cross(predictor, xtrain, ytrain, xtest, ytest, cv):
    """Fit model with cross-validation return stuff"""
    best = None
    for train, test in cv.split(xtrain, ytrain):
        proba, y_pred, score = run_once(
            predictor,
            xtrain.iloc[train],
            ytrain.iloc[train],
            xtrain.iloc[test],
            ytrain.iloc[test],
        )
        # model.fit(X_train.iloc[train], y_train.iloc[train])
        # y_pred = model.predict_proba(X_train.iloc[test])
        # score = metrics.roc_auc_score(y_train.iloc[test], y_pred[:, 1])
        if best is None:
            best = {
                'model': deepcopy(predictor),
                'y_pred': y_pred,
                'score': score
            }
        elif score > best['score']:
            best = {
                'model': deepcopy(predictor),
                'y_pred': y_pred,
                'score': score
            }

        print("ROC AUC: " + str(score))

    # TODO run best on validation data
    return best


def run_hyper(hyper, xtrain, ytrain, xtest, ytest, cv):
    n_iter = 20

    if 'model_init_params' not in hyper:
        hyper['model_init_params'] = {}

    param_grid = model_selection.ParameterGrid(hyper['params'])
    grid_size = len(param_grid)
    if grid_size < n_iter:
        # If we can exaust search space use GridSearchCV
        rs = model_selection.GridSearchCV(
            hyper['model'](**hyper['model_init_params']),
            hyper['params'],
            scoring='roc_auc',
            cv=cv,
            n_jobs=n_jobs,
            # verbose=1,
        )
    else:
        rs = model_selection.RandomizedSearchCV(
            hyper['model'](**hyper['model_init_params']),
            hyper['params'],
            n_iter=n_iter,
            scoring='roc_auc',
            cv=cv,
            n_jobs=n_jobs,
            # verbose=1,
        )

    rs.fit(xtrain, ytrain)
    probas = rs.predict_proba(xtest)
    pred = rs.predict(xtest)

    # new_model = hyper['model'](
    #     **rs.best_params_, n_jobs=n_jobs).fit(xtrain, ytrain)
    # new_model.fit(xtrain, ytrain)
    # y = new_model.predict_proba(xtest)[:, -1]

    res = {
        **hyper,
        'rs': rs,
        'estimator': rs.best_estimator_,
        'probas': probas,
        'y_pred': pred,
        'roc_auc': metrics.roc_auc_score(ytest, probas[:, -1]),
        'score': rs.score(xtest, ytest),
    }
    return res
