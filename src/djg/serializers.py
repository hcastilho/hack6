from rest_framework import serializers

from hack6_dj import models


class ObservationField(serializers.Field):

    def to_internal_value(self, data):
        pass


class PredictSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    observation = ObservationField()

    class Meta:
        model = models.Predict