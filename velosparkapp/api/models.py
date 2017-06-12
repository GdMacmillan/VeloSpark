from django.conf import settings
from django.db import models
from django.contrib.postgres.fields import JSONField
from django.contrib.auth.models import AbstractUser
from django.utils import timezone


class StravaUser(AbstractUser):
    pass

class Athlete(models.Model):
    """
    Model of athlete data. This contains essential fields and behaviors of the athlete data we are storing from the Strava API.
    """
    id = models.IntegerField(primary_key=True)
    resource_state = models.PositiveSmallIntegerField(blank=True)
    firstname = models.CharField(blank=True, max_length=140)
    lastname = models.CharField(blank=True, max_length=140)
    profile_medium = models.SlugField(blank=True)
    profile = models.SlugField(blank=True)
    city = models.CharField(blank=True, max_length=140)
    state = models.CharField(blank=True, max_length=140)
    country = models.CharField(blank=True, max_length=140)
    sex = models.CharField(blank=True, max_length=10)
    friend = models.CharField(blank=True, max_length=10) # This should always be null
    follower = models.CharField(blank=True, max_length=10) # This should always be null
    premium = models.BooleanField(default=False, blank=True)
    created_at = models.DateTimeField(default=timezone.now, blank=True)
    # updated_at = models.DateTimeField(default=timezone.now, blank=True)

    # detailed athlete data model fields
    follower_count = models.PositiveSmallIntegerField(blank=True, default=0)
    friend_count = models.PositiveSmallIntegerField(blank=True, default=0)
    mutual_friend_count = models.PositiveSmallIntegerField(blank=True, default=0)
    athlete_type = models.PositiveSmallIntegerField(blank=True, default=0) # athlete's default sport type: 0=cyclist, 1=runner
    date_preference = models.CharField(blank=True, max_length=140)
    measurement_preference = models.CharField(blank=True, max_length=10)
    email = models.CharField(blank=True, max_length=140)
    ftp = models.PositiveSmallIntegerField(blank=True, default=0)
    weight = models.FloatField(blank=True, default=0)
    clubs = JSONField(blank=True, default="{}")
    bikes = JSONField(blank=True, default="{}")
    shoes = JSONField(blank=True, default="{}")
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    def __str__(self):
        return "Athlete: {0} {1}".format(self.firstname, self.lastname)

    def deserialize(self, profile_info):
        self.firstname = profile_info['firstname']
        self.lastname = profile_info['lastname']
        self.resource_state = profile_info['resource_state']
        self.city = profile_info['city']


class Map(models.Model):
    """
    Model of map object for each activity. This contains essential fields and behaviors of the athlete data we are storing from the Strava API.
    """
    id = models.IntegerField(primary_key=True)
    summary_polyline = models.TextField()
    resource_state = models.PositiveSmallIntegerField()

    def __str__(self):
        return "Map id: {}".format(self.id)


class Activity(models.Model):
    """
    Model of activity data. This contains essential fields and behaviors of the activity data we are storing from the Strava API.
    """
    id = models.IntegerField(primary_key=True)
    resource_state = models.PositiveSmallIntegerField()
    external_id = models.CharField(max_length=140)
    upload_id = models.IntegerField()
    athlete = models.ForeignKey(Athlete)
    name = models.CharField(max_length=140)
    distance = models.FloatField()
    moving_time = models.DurationField() # seconds
    elapsed_time = models.DurationField() # seconds
    total_elevation_gain = models.FloatField() # meters
    type = models.CharField(max_length=140)
    start_date = models.DateTimeField()
    start_date_local = models.DateTimeField()
    time_zone = models.CharField(max_length=140)
    start_latlng = JSONField() # array of latlng floats
    end_latlng = JSONField() # array of latlng floats
    achievment_count = models.PositiveSmallIntegerField()
    kudos_count = models.PositiveSmallIntegerField()
    comment_count = models.PositiveSmallIntegerField()
    athlete_count = models.PositiveSmallIntegerField()
    photo_count = models.PositiveSmallIntegerField()
    total_photo_count = models.PositiveSmallIntegerField()
    _map = models.ForeignKey(Map)
    trainer = models.BooleanField()
    commute = models.BooleanField()
    manual = models.BooleanField()
    private = models.BooleanField()
    flagged = models.BooleanField()
    average_speed = models.FloatField() # meters per second
    max_speed = models.FloatField() # meters per second
    average_watts = models.FloatField() # rides only
    max_watts = models.IntegerField() # rides with power meter
    weighted_average_watts = models.IntegerField() # rides with power  meter
    kilojoules = models.FloatField() # rides only
    device_watts = models.BooleanField() # true if watts are from a power meter, false if estimated
    has_heartrate = models.BooleanField() # true if recorded with heartrate
    average_heartrate = models.FloatField() # only if recorded with heartrate average over moving portion
    max_heartrate = models.PositiveSmallIntegerField() # only if recorded with has_heartrate

    # detailed activity data model fields
    calories = models.FloatField() # kilocalories, uses kilojoules for rides and speed/pace for runs
    description = models.TextField()
    # gear = models.ForeignKey(Gear) Leave out Gear model for now
    suffer_score = models.PositiveSmallIntegerField()
    has_kudoed = models.BooleanField()
    segment_effort = JSONField() # store segment_effort as jSON
    splits_metric = JSONField() # store splits_metric as jSON
    laps = JSONField() # store the laps as JSON
    best_efforts = JSONField() # store the best_efforts as JSON
    # photos = JSONField() Leave out Photo model for now
    device_name = models.CharField(max_length=140)
    embed_token = models.CharField(max_length=140)

    def __str__(self):
        return "Activty: {}".format(self.name)
