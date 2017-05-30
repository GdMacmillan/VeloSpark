from django.conf.urls import url
from django.contrib import admin

from .views import page

urlpatterns = (
    url(r'^admin/', admin.site.urls),
    url(r'^(?P<slug>[\w./-]+)/$', page, name='page'),
    url(r'^$', page, name='homepage'),
)


# https://www.strava.com/oauth/authorize?client_id=15880&response_type=code&redirect_uri=http://localhost&approval_prompt=force