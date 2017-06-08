from django.conf.urls import url, include
from django.contrib import admin
from django.contrib.auth.views import logout_then_login

from .views import page

urlpatterns = (
    url(r'^admin/', admin.site.urls),
    url(r'^accounts/', include('allaccess.urls')),
    url(r'^logout/$', logout_then_login, name='logout'),
    url(r'^(?P<slug>[\w./-]+)/$', page, name='page'),
    url(r'^$', page, name='homepage'),
)


# https://www.strava.com/oauth/authorize?client_id=15880&response_type=code&redirect_uri=http://localhost&approval_prompt=force
