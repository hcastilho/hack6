import os
import logging
# import json
# import pickle
import flask
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from peewee import (SqliteDatabase,
                    PostgresqlDatabase,
                    Model,
                    IntegerField,
                    FloatField,
                    BooleanField,
                    TextField,
)
from playhouse.shortcuts import model_to_dict
from sklearn.externals import joblib

from hack6.modelling import MODEL_DIR

logger = logging.getLogger(__name__)

########################################
# Begin database stuff

if 'DATABASE_URL' in os.environ:
    db_url = os.environ['DATABASE_URL']
    dbname = db_url.split('@')[1].split('/')[1]
    user = db_url.split('@')[0].split(':')[1].lstrip('//')
    password = db_url.split('@')[0].split(':')[2]
    host = db_url.split('@')[1].split('/')[0].split(':')[0]
    port = db_url.split('@')[1].split('/')[0].split(':')[1]
    DB = PostgresqlDatabase(
        dbname,
        user=user,
        password=password,
        host=host,
        port=port,
    )
else:
    DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField(null=True)
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


columns = joblib.load(os.path.join(MODEL_DIR, 'columns.pkl'))
pipe = joblib.load(os.path.join(MODEL_DIR, 'best.pkl'))
dtypes = joblib.load(os.path.join(MODEL_DIR, 'dtypes.pkl'))


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


def get_or_create_prediction(_id):
    try:
        p = Prediction.get(Prediction.observation_id == _id)

    except Prediction.DoesNotExist:
        p = Prediction(observation_id=_id)

    return p


observation_keys = (
        'birth date',
        'job type',
        'school level',
        'domestic status',
        'profession',
        'domestic relationship type',
        'ethnicity',
        'gender',
        'earned dividends',
        'interest earned',
        'monthly work',
        'country of origin',
)

numeric_keys = (
    'earned dividends',
    'interest earned',
    'monthly work',
)


@app.route('/predict', methods=['POST'])
def predict():
    logger.debug(request.data)

    obs_dict = request.get_json()

    # Validate input
    for k in ('id', 'observation'):
        if k not in obs_dict:
            msg = "Computer says no! %s missing" % k
            logger.debug(msg)
            return flask.Response(msg, status=400)

    _id = obs_dict['id']
    observation = obs_dict['observation']

    for k in observation_keys:
        if k not in observation:
            msg = "Computer says no! %s missing" % k
            logger.debug(msg)
            return flask.Response(msg, status=400)

    for k in numeric_keys:
        try:
            int(observation[k])
        except ValueError:
            msg = "%s, %s is not remotely number-like" % (observation[k], k)
            logger.debug(msg)
            return flask.Response(msg, status=400)
        except TypeError:
            observation[k] = np.nan

    p = get_or_create_prediction(_id)
    p.observation = request.data
    p.save()

    # columns parameter ensures the df is in the right order
    obs = pd.DataFrame([observation], columns=columns)
    obs = obs.astype(dtypes)

    logger.debug(obs)

    proba = pipe.predict_proba(obs)[0, 1]

    p.proba = proba
    p.observation = request.data
    p.save()

    return jsonify({'proba': proba})


@app.route('/update', methods=['POST'])
def update():
    logger.debug(request.data)
    obs = request.get_json()

    # Validate input
    for input_ in ('id', 'true_class'):
        if input_ not in obs:
            return flask.Response("Computer says no! %s missing" % input_,
                                  status=400)

    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])

    except Prediction.DoesNotExist:
        return flask.Response("Nothing there to update, stahp!", status=404)

    p.true_class = obs['true_class']
    p.save()

    return jsonify(model_to_dict(p))


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(debug=True)

