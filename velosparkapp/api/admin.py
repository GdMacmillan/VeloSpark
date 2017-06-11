from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import Athlete, Activity, Map, StravaUser

# inline admin descriptor for Athlete Model
class AthleteInline(admin.StackedInline):
    model = Athlete
    can_delete = False
    verbose_name_plural = 'athlete'


class UserAdmin(BaseUserAdmin):
    inlines = (AthleteInline, )


admin.site.register(StravaUser, UserAdmin)
# admin.site.unregister(User)
admin.site.register(Map)
admin.site.register(Activity)
