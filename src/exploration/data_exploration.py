import os
from itertools import tee

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pycountry
import seaborn as sns
from fuzzywuzzy import process
from sklearn import model_selection

from exploration.utils import BASE_DIR, latex_table

OUTPUT = True
STORE = False

# Output pgf
# matplotlib.use('pgf')
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pgf', FigureCanvasPgf)


pycountry.countries._load()
dataset = pd.read_csv(os.path.join(BASE_DIR, 'data', 'train.csv'))

datetime_columns = [
    'birth date',
]

numerical_columns = [
    'earned dividends',
    'interest earned',
    'monthly work',
]
dataset[numerical_columns].describe()

categorical_columns = [
    # 'id',
    # 'birth date'
    'job type',
    'school level',
    'domestic status',
    'profession',
    'domestic relationship type',
    'ethnicity',
    'gender',
    # 'earned dividends',
    # 'interest earned',
    # 'monthly work',
    'country of origin',
    # 'target',
]

# for col in categorical_columns:
#     print('\n', col)
#     print(dataset[col].value_counts())


def country_to_internal(txt):
    txt = txt.lower()

    if txt in ('u.s.', 'u.s.a.', 'u.s.a'):
        txt = 'us'
    elif txt == 'laos':
        txt = 'lao'
    elif txt == 'unknown':
        return 'unknown'
    elif txt == 'dr':  # this is not a country
        txt = 'do'

    try:
        ctry = pycountry.countries.lookup(txt)

    except LookupError:
        match = process.extractOne(txt, country_names)
        ctry = pycountry.countries.objects[country_names.index(match[0])]

        if match[1] < 80:

            try:
                sub = pycountry.subdivisions.lookup(txt)
                return sub.country.alpha_2
            except LookupError:
                print(txt, '~', ctry.name, '~', match[1])
                return 'unknown'

    return ctry.alpha_2

# ID
################################
if OUTPUT:
    print('\n# Id')
    print('Unique: ', dataset.shape[0] == dataset['id'].unique().size)
dataset.set_index('id', inplace=True)
# dataset = dataset.drop('id', axis=1)

# country of origin
################################
country_names = [obj.name for obj in pycountry.countries.objects]
# WTF are these dr?? democratic rep of smth?
# for i, r in dataset.iterrows():
#     if r['country of origin'] == 'dr':
#         print(r)
dataset['country of origin'].value_counts()
if OUTPUT:
    print('\n# Country')
    print(latex_table(dataset['country of origin'].value_counts().items()))

dataset['country of origin'] = dataset['country of origin'].map(
    country_to_internal)

# birth date
################################
# dataset['birth date'] = dataset['birth date'].map(
#     lambda x: parser.parse(x).date())
dataset['birth date'] = pd.to_datetime(dataset['birth date'])

# dataset['birth date'].plot.hist()
# dataset['birth date'].groupby((
#     dataset['birth date'].dt.year,
# )).count().plot(kind="bar")
plt.figure(figsize=(5, 3.9))
dataset['birth date'].groupby((
    dataset['birth date'].dt.year,
)).count().plot()
# dataset['birth date'].resample('5AS').count().plot(kind="bar")
# (dataset.set_index('birth date')
#     .resample('5AS')['target'].count().plot(kind="bar"))

# Show/save plot
if OUTPUT:
    print('\n# Birth date')
    print('Max: ', dataset['birth date'].max())
    print('Min: ', dataset['birth date'].min())

    if STORE:
        plt.savefig(os.path.join(BASE_DIR,
                                 'doc/report/img/birth_date_freq.pgf'))
    else:
        plt.show()

dataset['birth date'] = dataset['birth date'].map(lambda x: x.timestamp())

# domestic relationship type: ugly
################################
if OUTPUT:
    print('\n# Domestic relationship type')
    print(latex_table(
        dataset['domestic relationship type'].value_counts().items()))

    print(dataset.groupby(
        ('domestic status', 'domestic relationship type')).size())

# domestic status: d to divorce, several types of married
################################
if OUTPUT:
    print('\n# Domestic status')
    print(latex_table(dataset['domestic status'].value_counts().items()))

# earned dividends
################################
if OUTPUT:
    print('\n# Earned dividends')
    print(latex_table(dataset['earned dividends'].value_counts().items()))
dataset = dataset.drop('earned dividends', axis=1)
numerical_columns.remove('earned dividends')

# ethnicity: ugly
################################
if OUTPUT:
    print('\n# Ethnicity')
    print(latex_table(dataset['ethnicity'].value_counts().items()))

# gender: all female dataset
################################
if OUTPUT:
    print('\n# Gender')
    print(latex_table(dataset['gender'].value_counts().items()))
dataset = dataset.drop('gender', axis=1)
categorical_columns.remove('gender')

# job type
################################
if OUTPUT:
    print('\n# Job type')
    print(latex_table(dataset['job type'].value_counts().items()))

# interest earned
################################
if OUTPUT:
    print('\n# Interest earned')

plt.figure(figsize=(5, 3.9))
dataset['interest earned'].plot(kind='hist', logy=True)

# Show/save plot
plt.figure(figsize=(5, 3.9))
ax=dataset['interest earned'].plot(kind='hist', logy=True)
# ax.set_yscale(nonposy='mask')
# ax.set_xscale("log", nonposx='clip')

if OUTPUT:
    if STORE:
        plt.savefig(
            os.path.join(BASE_DIR,
                         'doc/report/img/interest_earned_freq.png'))
        plt.savefig(
            os.path.join(BASE_DIR,
                         'doc/report/img/interest_earned_freq.pgf'))
    else:
        plt.show()

# monthly work
################################
if OUTPUT:
    print('\n# Monthly work')

plt.figure(figsize=(5, 3.9))
dataset['monthly work'].plot(kind='hist')

# Show/save plot
if OUTPUT:
    if STORE:
        plt.savefig(os.path.join(BASE_DIR,
                                 'doc/report/img/monthly_work_freq.pgf'))
    else:
        plt.show()

# profession
################################
if OUTPUT:
    print('\n# Profession')
    print(latex_table(dataset['profession'].value_counts().items()))

# school level: ugly
################################
if OUTPUT:
    print('\n# School level')
    print(latex_table(dataset['school level'].value_counts().items()))

# normalize data
################################
# for col in numerical_columns:
# dataset[col] = preprocessing.normalize(
#     dataset[col].values.reshape(-1, 1))

# categorical data to dummy
################################
dataset = pd.get_dummies(dataset, columns=categorical_columns)

#####################
# Split
#####################
features = dataset.drop(['target'], axis=1)
target = dataset['target']

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    features,
    target,
    test_size=0.4,
    random_state=0
)


#####################
# Random stuff
#####################


def plot_corr(df, size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)


def windows(iterable, size):
    """Sliding window iterator"""
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)


def plot_corr_split(df, window_size=10, size=20, h=None, w=None):
    """Plot graphical correlation matrix split"""
    corr = df.corr()
    width = corr.shape[-1]

    for i in range(0, width, window_size):
        cols = corr.columns[i:min(i + window_size, width)]

        # Only show half of the correlation matrix it is diagonally
        # symmetric
        for j in range(i, width, window_size):
            rows = corr.columns[j:min(j + window_size, width)]

            sub_corr = corr[[*cols]].loc[[*rows]]

            if h is not None:
                fig, ax = plt.subplots(figsize=(w, h))
            else:
                fig, ax = plt.subplots(figsize=(size, size))

            # ax.matshow(sub_corr)
            # plt.xticks(range(window_size), cols)
            # plt.yticks(range(window_size), rows)

            ax = sns.heatmap(sub_corr,
                             xticklabels=cols,
                             yticklabels=rows,
                             ax=ax)

            # ax.figure.subplots_adjust(left=0.5, bottom=0.5)
            ax.figure.tight_layout()
            if OUTPUT:
                if STORE:
                    plt.savefig(
                        os.path.join(
                            BASE_DIR,
                            'doc/report/img/cross-matrix-%s-%s.pgf'
                            % (int(i/window_size), int(j/window_size)),
                        )
                    )
                else:
                    plt.show()


plot_corr_split(dataset, 34, h=10, w=12)

