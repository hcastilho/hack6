import pandas as pd
import numpy as np
import sqlalchemy
from sklearn import metrics


engine = sqlalchemy.create_engine(
    "postgresql+psycopg2://postgres:bestpwd@localhost/postgres")


df = pd.read_sql_table('prediction', engine)

print(df.dtypes)
print(df.shape())
print(df.describe())

df_clean = df.dropna(axis=0, how='any')
metrics.roc_auc_score(df_clean['true_class'], df_clean['proba'])
