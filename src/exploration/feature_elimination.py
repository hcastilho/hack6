import os
from operator import itemgetter

import matplotlib
import matplotlib.pyplot as plt
from sklearn import feature_selection

from exploration.utils import load_models, get_best, BASE_DIR, latex_table
from exploration.modelling import cv, n_jobs
from exploration.data_exploration import X_test, y_test, dataset, X_train, \
    y_train

# Output pgf
# matplotlib.use('pgf')
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pgf', FigureCanvasPgf)

results = load_models()
best = get_best(results)

# model = best['model'](**best['estimator'].get_params())

# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = feature_selection.RFECV(estimator=best['estimator'],
                                # step=1,
                                cv=cv,
                                # cv=StratifiedKFold(2),
                                scoring='roc_auc',
                                n_jobs=n_jobs,
                                )
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

print("Optimal features :")
for f in X_test.columns[rfecv.get_support(indices=True)]:
    print("\item %s" % f)

# print("Feature importances: %s" % rfecv.estimator.feature_importances_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("ROC AUC score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

plt.savefig(os.path.join(BASE_DIR,
                         'doc/report/img/score-features.pgf'))

# Compare with cross correlation
print(latex_table(
    dataset.corr()['target'][
        X_test.columns[rfecv.get_support(indices=True)]]
        .map(lambda x: '{:4f}'.format(x))
        .items()
))
