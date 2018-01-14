import os
import json
from random import randint
from pprint import pprint

import copy
import pytest
import pandas as pd
from hack6.modelling import DATA_DIR

df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
examples = df.sample(10, random_state=1)
examples = examples.drop(['id', 'target'], axis=1)
examples = [k.to_dict() for _id, k in examples.T.items()]
examples = [{'id': i, 'observation': k} for i, k in enumerate(examples)]

clean_examples = copy.deepcopy(examples[:9])
dirty_examples = copy.deepcopy(examples[:9])
for example in dirty_examples:
    example['id'] = 100 + example['id']
# 1. Missing Values (monthly work with a missing value (NaN), and gender with missing value (NaN))
dirty_examples[0]['observation']['monthly work'] = None
dirty_examples[0]['observation']['gender'] = None
# 2. Categories never seen before (e.g., Male inside Gender, New_Level inside school level)
dirty_examples[1]['observation']['gender'] = 'Male'
dirty_examples[1]['observation']['school level'] = 'geniOus'
# 3. Missing column (job type)
del dirty_examples[2]['observation']['job type']
# 4. school level as integer
dirty_examples[3]['observation']['school level'] = 111
# monthly work as str (non-numerical)
dirty_examples[4]['observation']['monthly work'] = 'r32g'
# monthly work as str (numerical)
dirty_examples[5]['observation']['monthly work'] = '40'
# birth date with different format (dd-mm-yyyy)
dirty_examples[6]['observation']['birth date'] = '01-01-1946'
# empty observation
dirty_examples[7]['observation'] = {}
# empty payload
dirty_examples[8] = {}


observations = dirty_examples


def test_missing_values(client):
    response = client.post('/predict', data=json.dumps(dirty_examples[0]),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_categories_never_seen(client):
    response = client.post('/predict', data=json.dumps(dirty_examples[1]),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_missing_column(client):
    response = client.post('/predict', data=json.dumps(dirty_examples[2]),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_school_level_as_integer(client):
    response = client.post('/predict', data=json.dumps(dirty_examples[3]),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_monthly_work_as_str(client):
    response = client.post('/predict', data=json.dumps(dirty_examples[4]),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_monthly_work_as_str_num(client):
    response = client.post('/predict', data=json.dumps(dirty_examples[5]),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_birth_date_diferent_format(client):
    response = client.post('/predict', data=json.dumps(dirty_examples[6]),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_empty_obs(client):
    response = client.post('/predict', data=json.dumps(dirty_examples[7]),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_empty_payload(client):
    response = client.post('/predict', data=json.dumps(dirty_examples[7]),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_root(client):
    response = client.get('/')
    assert response.status_code == 404


id_ = randint(1, 10 * 100)


def test_predict(client):
    data = {
        'id': id_,
        'observation': {
            'birth date': '1990-12-24',
            'job type': 'private',
            'school level': 'entry level college',
            'domestic status': 'single',
            'profession': 'C-level',
            'domestic relationship type': 'not living with family',
            'ethnicity': 'white and privileged',
            'gender': 'Female',
            'earned dividends': '0',
            'interest earned': '0',
            'monthly work': '160',
            'country of origin': 'u.s.',
        },
    }

    response = client.post('/predict',
                           data=json.dumps(data),
                           headers={'Content-Type': 'application/json'})
    print(response.data)
    assert response.status_code == 200

    # Allows resubmit
    response = client.post('/predict',
                           data=json.dumps(data),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_update(client):
    # does not exist
    response = client.post('/update',
                           data=json.dumps({
                               'id': -1,
                               'true_class': 1,
                           }),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 404

    # exists
    response = client.post('/update',
                           data=json.dumps({
                               'id': id_,
                               'true_class': 1,
                           }),
                           headers={'Content-Type': 'application/json'})
    print(response.data)
    assert response.status_code == 200


def test_list_db_contents(client):
    response = client.get('/list-db-contents')
    assert response.status_code == 200

    data = response.json
    # pprint(data)
    assert len(data) >= 1
