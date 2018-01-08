from django.db import models


class Predict(models.Model):
    id = models.IntegerField(primary_key=True)
    observation = models.TextField()
    proba = models.FloatField()
    true_class = models.IntegerField(null=True)
