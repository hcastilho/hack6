import os

import pandas as pd
import pycountry
from dateutil import parser
from fuzzywuzzy import process
from sklearn import preprocessing

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
dataset.set_index('id', inplace=True)
# dataset.drop('id', axis=1)


# birth date
################################
dataset['birth date'] = dataset['birth date'].map(
    lambda x: int(parser.parse(x).timestamp()))

# job type
################################

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
