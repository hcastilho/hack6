import os
# import json
# import pickle

import pandas as pd
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
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


columns = joblib.load(os.path.join(MODEL_DIR, 'columns.pkl'))
pipe = joblib.load(os.path.join(MODEL_DIR, 'pipe.pkl'))
dtypes = joblib.load(os.path.join(MODEL_DIR, 'dtypes.pkl'))


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json
    obs_dict = request.get_json()
    print(obs_dict)
    print(request)
    _id = obs_dict['id']
    observation = obs_dict['observation']
    # now do what we already learned in the notebooks about how to transform
    # a single observation into a dataframe that will work with a pipeline
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    # now get ourselves an actual prediction of the positive class
    proba = pipe.predict_proba(obs)[0, 1]
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    p.save()
    return jsonify({'proba': proba})


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    p = Prediction.get(Prediction.observation_id == obs['id'])
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

