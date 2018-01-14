import json
import requests
from random import randint

from urllib.parse import urljoin

URL = 'https://peaceful-tor-58476.herokuapp.com/'

data = {
    'id': randint(1, 10 * 100),
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

response = requests.post(urljoin(URL, '/predict'),
                         data=json.dumps(data),
                         headers={'Content-Type': 'application/json'})
print(response)
print(response.content)
assert response.status_code == 200

# Allows resubmit
response = requests.post(urljoin(URL, '/predict'),
                         data=json.dumps(data),
                         headers={'Content-Type': 'application/json'})
assert response.status_code == 200

# def test_update(client):
response = requests.post(urljoin(URL, '/update'),
                         data=json.dumps({
                             'id': 1,
                             'true_class': 1,
                         }),
                         headers={'Content-Type': 'application/json'})
assert response.status_code == 200

response = requests.get(urljoin(URL, '/list-db-contents'))
assert response.status_code == 200

data = response.json
# pprint(data)
assert len(data) >= 1
