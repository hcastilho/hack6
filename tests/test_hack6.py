import copy
import os
import json
from random import randint
from pprint import pprint


id_ = randint(1, 10 * 100)
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


def test_missing_numerical_value(client):
    d = copy.deepcopy(data)
    d['observation']['monthly work'] = None
    response = client.post('/predict', data=json.dumps(d),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_missing_categorical_value(client):
    d = copy.deepcopy(data)
    d['observation']['gender'] = None
    response = client.post('/predict', data=json.dumps(d),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_categories_never_seen(client):
    d = copy.deepcopy(data)
    d['observation']['gender'] = 'Male'
    response = client.post('/predict', data=json.dumps(d),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200

    d = copy.deepcopy(data)
    d['observation']['school level'] = 'geniOus'
    response = client.post('/predict', data=json.dumps(d),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_missing_column(client):
    d = copy.deepcopy(data)
    del d['observation']['job type']
    response = client.post('/predict', data=json.dumps(d),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 400


def test_school_level_as_integer(client):
    d = copy.deepcopy(data)
    d['observation']['school level'] = 111
    response = client.post('/predict', data=json.dumps(d),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_monthly_work_as_str(client):
    d = copy.deepcopy(data)
    d['observation']['monthly work'] = 'r32g'
    response = client.post('/predict', data=json.dumps(d),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 400


def test_monthly_work_as_str_num(client):
    d = copy.deepcopy(data)
    d['observation']['monthly work'] = '40'
    response = client.post('/predict', data=json.dumps(d),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_birth_date_diferent_format(client):
    d = copy.deepcopy(data)
    d['observation']['birth date'] = '01-01-1946'
    response = client.post('/predict', data=json.dumps(d),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


def test_empty_obs(client):
    d = copy.deepcopy(data)
    d['observation'] = {}
    response = client.post('/predict', data=json.dumps(d),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 400


def test_empty_payload(client):
    response = client.post('/predict', data=json.dumps({}),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 400


def test_root(client):
    response = client.get('/')
    assert response.status_code == 404


def test_predict(client):
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
