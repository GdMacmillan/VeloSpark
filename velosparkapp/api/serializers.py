from rest_framework import serializers

from .models import Athlete

# class AthleteSerializer(serializers.ModelSerializer):
#
#     class Meta:
#         model = Athlete
#         fields = ('id', 'firstname', 'lastname', 'city', 'state')

class StravaUser(serializers.ModelSerializer):

    class Meta:
        model = Athlete
        fields = ('id', 'firstname', 'lastname', 'city', 'state')
