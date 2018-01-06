import os
from datetime import date

import pandas as pd
import pycountry
from dateutil import parser
from fuzzywuzzy import process
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt

OUTPUT = False
STORE = False

# Output pgf
# matplotlib.use('pgf')
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pgf', FigureCanvasPgf)

try:
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except:
    BASE_DIR = os.getcwd()

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


def latex_table(data, pos='', table_spec=''):
    table = ''
    for row in data:
        if not table:
            if not table_spec:
                table_spec = 'c' * len(row)
            table_spec = '{' + table_spec + '}'
            if not pos:
                pos=''
            table = '    \\begin{{tabular}}{pos}{table_spec}\n'.format(
                pos=pos,
                table_spec=table_spec,
            )
        table += '        ' + ' & '.join(map(str, row)) + ' \\\\\n'
    table += '    \\end{tabular}\n'
    return table


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
                         'doc/report/img/interest_earned_freq.pgf'))
        plt.savefig(
            os.path.join(BASE_DIR,
                         'doc/report/img/interest_earned_freq.png'))
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
