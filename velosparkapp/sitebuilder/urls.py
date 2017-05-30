from django.conf.urls import url

from .views import page

urlpatterns = (
    url(r'^(?P<slug>[\w./-]+)/$', page, name='page'),
    url(r'^$', page, name='homepage'),
)


# https://www.strava.com/oauth/authorize?client_id=15880&response_type=code&redirect_uri=http://localhost&approval_prompt=force
