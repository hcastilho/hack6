import json
import os

import pandas as pd
import sqlalchemy
from exploration.utils import BASE_DIR
from hack6.modelling import ForceDTypeDate

engine = sqlalchemy.create_engine(
    "postgresql+psycopg2://postgres:bestpwd@localhost/postgres")
db_data = pd.read_sql_table('prediction', engine)
# db_targets = db_data.dropna(axis=0, how='any')

orig_dataset = pd.read_csv(os.path.join(BASE_DIR, 'data', 'train.csv'))


samples = []
for index, row in db_data.iterrows():
    obs = json.loads(row['observation'])
    samples.append({
        'id': obs['id'],
        'target': row['true_class'],
        'proba': row['proba'],
        **obs['observation'],
    })


new_dataset = pd.DataFrame(samples)
new_targets = new_dataset.dropna(axis=0, how='any')

datasets = {
    'original': ForceDTypeDate().transform(orig_dataset.copy()),
    'new': ForceDTypeDate().transform(new_dataset),
    'targets': ForceDTypeDate().transform(new_targets),
}