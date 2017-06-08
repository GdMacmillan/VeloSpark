from django.contrib import admin

from .models import Athlete, Activity, Map

admin.site.register(Athlete)
admin.site.register(Map)
admin.site.register(Activity)
