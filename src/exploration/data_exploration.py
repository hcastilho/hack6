import os
from datetime import date

import pandas as pd
import pycountry
from dateutil import parser
from fuzzywuzzy import process
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt

# Output pgf
# matplotlib.use('pgf')
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pgf', FigureCanvasPgf)

# Fit A4
plt.figure(figsize=(11.69, 8.27))

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

for col in categorical_columns:
    print('\n', col)
    print(dataset[col].value_counts())


# ID
################################
print('\n# Id')
print('Unique: ', dataset.shape[0] == dataset['id'].unique().size)
dataset.set_index('id', inplace=True)
# dataset.drop('id', axis=1)

# birth date
################################
print('\n# Birth date')
# dataset['birth date'] = dataset['birth date'].map(
#     lambda x: parser.parse(x).date())
dataset['birth date'] = pd.to_datetime(dataset['birth date'])

print('Max: ', dataset['birth date'].max())
print('Min: ', dataset['birth date'].min())
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
# plt.savefig(os.path.join(BASE_DIR, 'doc/report/img/birth_date_freq.pgf'))
plt.show()

dataset['birth date'] = dataset['birth date'].map(lambda x: x.timestamp())

# job type
################################
print('\n# Job type')
print(dataset['job type'].value_counts())
for row in dataset['job type'].value_counts().items():
    line = ' & '.join(map(str, row)) + r' \\'
    print(line)

# school level: ugly
################################

# domestic status: d to divorce, several types of married
################################

# profession
################################

# domestic relationship type: ugly
################################

# ethnicity: ugly
################################

# gender: all female dataset
################################

# earned dividends
################################

# interest earned
################################

# monthly work
################################

# country of origin
################################
country_names = [obj.name for obj in pycountry.countries.objects]
# WTF are these dr?? democratic rep of smth?
# for i, r in dataset.iterrows():
#     if r['country of origin'] == 'dr':
#         print(r)

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

dataset['country of origin'] = dataset['country of origin'].map(
    country_to_internal)


# normalize data
################################
# for col in numerical_columns:
#     dataset[col] = preprocessing.normalize(dataset[col].values.reshape(-1, 1))

# categorical data to dummy
################################
dataset = pd.get_dummies(dataset, columns=categorical_columns)
