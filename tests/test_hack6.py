import json
from random import randint
from pprint import pprint

import pytest


def test_root(client):
    response = client.get('/')
    assert response.status_code == 404


def test_predict(client):
    # TODO store observation
    # TODO check for true_class
    # TODO error handling
    data = {
        'id': randint(1, 10*100),
        'observation': [
            # '19723',
            '1990-12-24',
            'private',
            'entry level college',
            'single',
            'C-level',
            'not living with family',
            'white and privileged',
            'Female',
            '0',
            '0',
            '160',
            'u.s.',
            # '1'
        ],
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


def test_update(client):
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

    # data = response.json
    # pprint(data)
    assert len(data) >= 1

    assert False
