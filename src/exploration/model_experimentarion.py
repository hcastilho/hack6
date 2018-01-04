# Import pandas and numpy
import pandas as pd
import numpy as np

# Import the classifiers we will be using
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Import train/test split function
from sklearn.model_selection import train_test_split

# Import cross validation scorer
from sklearn.model_selection import cross_val_score

# Import ROC AUC scoring function
from sklearn.metrics import roc_auc_score


#####################
# Train/Test Split
#####################
features = dataset.drop(['target'], axis=1)
target = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.4,
                                                    random_state=0)
# Choose the model
model = RandomForestClassifier()

# Fit the model
model.fit(X_train, y_train)

# Make the predictions
y_pred = model.predict_proba(X_test)

# Score the predictions
score = roc_auc_score(y_test, y_pred[:,1])

print("ROC AUC: " + str(score))


############################
# Cross Validation
############################
cv = model_selection.ShuffleSplit(n_splits=5, test_size=0.3, random_state=1)

model = RandomForestClassifier()
for train, test in cv.split(X_train, y_train):
    model.fit(X_train.iloc[train], y_train.iloc[train])
    y_pred = model.predict_proba(X_train.iloc[test])
    score = roc_auc_score(y_train.iloc[test], y_pred[:, 1])
    print("ROC AUC: " + str(score))


############################
# Hyper-parameter Tuning
############################
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.4,
                                                    random_state=0)
tests = [
    {
        'name': 'random_forest',
        'model': RandomForestClassifier(),
        'params': {
            'min_samples_split': range(400),
            'n_estimators': range(400),
            # ...
        },
    },
]
results = {}
for test in tests:
    cv = model_selection.ShuffleSplit(n_splits=3,
                                      test_size=0.3,
                                      random_state=1)
    rs = model_selection.RandomizedSearchCV(test['model'],
                                            test['params'],
                                            n_iter=20,
                                            scoring='roc_auc',
                                            cv=cv,
                                            n_jobs=-1,
                                            verbose=1,
                                            )
    rs.fit(X_train, y_train)
    probas = rs.predict_proba(X_test)
    results[test['name']] = {
        'rs': rs,
        'y_pred': probas[:, 1],
        'roc_auc': roc_auc_score(y_test, probas[:, 1]),
    }

