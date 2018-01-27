import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from exploration.utils import latex_table, BASE_DIR
from verification.data import datasets

OUTPUT = True
STORE = False

if OUTPUT:
    # print(datasets['new'].dtypes)
    # print(datasets['new'].shape)
    # print(datasets['new'].describe())
    print(datasets['new'].shape)
    print("AUC ROC",
          metrics.roc_auc_score(datasets['targets']['target'],
                                datasets['targets']['proba']))

    print("AUC ROC ones",
          metrics.roc_auc_score(
              datasets['targets']['target'],
              np.ones(datasets['targets'].shape[0])))

    print("AUC ROC zeros",
          metrics.roc_auc_score(
              datasets['targets']['target'],
              np.ones(datasets['targets'].shape[0])))

if OUTPUT:

    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(
        datasets['targets']['target'], datasets['targets']['proba'])
    roc_auc = metrics.auc(false_positive_rate,
                          true_positive_rate)

    plt.figure(figsize=(5, 3.9))
    lw = 2
    plt.plot(false_positive_rate,
             true_positive_rate,
             color='darkorange',
             lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    if STORE:
        plt.savefig(
            os.path.join(BASE_DIR,
                         'doc/final/img/roc.png'))
    else:
        plt.show()

if OUTPUT:
    plt.figure(figsize=(5, 3.9))
    datasets['new']['proba'].plot(kind='hist')

    if STORE:
        plt.savefig(
            os.path.join(BASE_DIR,
                         'doc/final/img/proba.png'))
    else:
        plt.show()

if OUTPUT:
    plt.figure(figsize=(5, 3.9))
    datasets['targets']['proba'].plot(kind='hist')

    if STORE:
        plt.savefig(
            os.path.join(BASE_DIR,
                         'doc/final/img/proba-targets.png'))
    else:
        plt.show()

if OUTPUT:
    plt.figure(figsize=(5, 3.9))
    datasets['targets']['target'].plot(kind='hist')

    if STORE:
        plt.savefig(
            os.path.join(BASE_DIR,
                         'doc/final/img/proba-true.png'))
    else:
        plt.show()

if OUTPUT:
    plt.figure(figsize=(5, 3.9))
    datasets['original']['target'].plot(kind='hist')

    if STORE:
        plt.savefig(
            os.path.join(BASE_DIR,
                         'doc/final/img/orig-true.png'))
    else:
        plt.show()

# Compare datasets
###########################

# country of origin
if OUTPUT:
    print('\n# Country of origin')

    for k, df in datasets.items():
        print(k)

        print(latex_table(
            df['country of origin'].value_counts()[:10].items()))

# birth date
if OUTPUT:
    print('\n# Birth date')

for k, df in datasets.items():

    if OUTPUT:
        print('\n', k)

        plt.figure(figsize=(5, 3.9))
        df['birth date'].groupby(
            (df['birth date'].dt.year, )).count().plot()
        print('Max: ', df['birth date'].max())
        print('Min: ', df['birth date'].min())

        if STORE:
            plt.savefig(
                os.path.join(
                    BASE_DIR,
                    'doc/final/img/birth-date-freq-%s.png' % k,
                ))
        else:
            plt.show()

# domestic relationship type
if OUTPUT:
    print('\n# Domestic relationship type')

    for k, df in datasets.items():
            print('\n', k)

            print(latex_table(
                df['domestic relationship type'].value_counts().items()))

            # print(df.groupby(
            #     ('domestic status', 'domestic relationship type')).size())

# domestic status
if OUTPUT:
    print('\n# Domestic status')

    for k, df in datasets.items():
        print('\n', k)
        print(latex_table(df['domestic status'].value_counts().items()))

# Earned dividends
if OUTPUT:
    print('\n# Earned dividends')

    for k, df in datasets.items():
        print('\n', k)

        plt.figure(figsize=(5, 3.9))
        df['earned dividends'].plot(kind='hist', logy=True)

        if OUTPUT:
            if STORE:
                plt.savefig(
                    os.path.join(
                        BASE_DIR,
                        'doc/final/img/earned-dividends-hist-%s.png' % k
                    ))
            else:
                plt.show()

# ethnicity
if OUTPUT:
    print('\n# Ethnicity')

    for k, df in datasets.items():
        print('\n', k)
        print(latex_table(df['ethnicity'].value_counts().items()))

# gender
if OUTPUT:
    print('\n# Gender')

    for k, df in datasets.items():
        print('\n', k)
        print(latex_table(df['gender'].value_counts().items()))

# job type
if OUTPUT:
    print('\n# Job type')

    for k, df in datasets.items():
        print('\n', k)
        print(latex_table(df['job type'].value_counts().items()))

# interest earned
if OUTPUT:
    print('\n# interest earned')

    for k, df in datasets.items():
        print('\n', k)

        plt.figure(figsize=(5, 3.9))
        df['interest earned'].plot(kind='hist', logy=True)

        if OUTPUT:
            if STORE:
                plt.savefig(
                    os.path.join(
                        BASE_DIR,
                        'doc/final/img/interest-earned-%s.png' % k
                    ))
            else:
                plt.show()

# monthly work
if OUTPUT:
    print('\n# Monthly work')

    for k, df in datasets.items():
        print('\n', k)

        plt.figure(figsize=(5, 3.9))
        df['monthly work'].plot(kind='hist', logy=True)

        if OUTPUT:
            if STORE:
                plt.savefig(
                    os.path.join(
                        BASE_DIR,
                        'doc/final/img/monthly-work-%s.png' % k
                    ))
            else:
                plt.show()

# profession
if OUTPUT:
    print('\n# Profession')

    for k, df in datasets.items():
        print('\n', k)
        print(latex_table(df['profession'].value_counts().items()))

# school level
if OUTPUT:
    print('\n# School level')

    for k, df in datasets.items():
        print('\n', k)
        print(latex_table(df['school level'].value_counts().items()))
