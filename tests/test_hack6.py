import json
from random import randint
from pprint import pprint

import pytest


def test_root(client):
    response = client.get('/')
    assert response.status_code == 404


def test_predict(client):
    data = {
        'id': randint(1, 10*100),
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
    assert response.status_code == 200


    # Allows resubmit
    response = client.post('/predict',
                           data=json.dumps(data),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 200


#def test_update(client):
    response = client.post('/update',
                           data=json.dumps({
                               'id': 1,
                               'true_class': 1,
                           }),
                           headers={
                               'Content-Type': 'application/json'
                           })
    assert response.status_code == 200


def test_list_db_contents(client):
    response = client.get('/list-db-contents')
    assert response.status_code == 200

    data = response.json
    # pprint(data)
    assert len(data) >= 1
