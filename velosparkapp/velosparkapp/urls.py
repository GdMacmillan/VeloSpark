from django.conf.urls import url, include
from django.contrib import admin
from django.contrib.auth.views import logout_then_login

# from .views import page
from . import views

urlpatterns = (
    url(r'^admin/', admin.site.urls),
    url(r'^accounts/', include('allaccess.urls')),
    url(r'^logout/$', logout_then_login, name='logout'),

    # url(r'^(?P<slug>[\w./-]+)/$', views.page, name='page'),
    # url(r'^$', views.page, name='homepage'),
    url(r'^(?P<slug>[\w./-]+)/$', views.MyTemplateView.as_view(), name='page'),
    url(r'^$', views.MyTemplateView.as_view(), name='homepage'),
)

# urlpatterns = (
    # url(r'^admin/', admin.site.urls),
    # url(r'^accounts/', include('allaccess.urls')),
    # url(r'^logout/$', logout_then_login, name='logout'),
#     url(r'^$', views.IndexView.as_view(), name='index'),
#     url(r'^/contact/$', views.ContactView.as_view(), name='contact'),
#     url(r'^/map/$', views.MapView.as_view(), name='map'),
#     url(r'^/activity/(?P<pk>[0-9]+)/$', views.ActivityView.as_view(), name='activity'),
#     url(r'^/profile/(?P<pk>[0-9]+)/$', views.ProfileView.as_view(), name='profile'),
#     url(r'^/profile/(?P<pk>[0-9]+)/results/$', views.RecResultsView.as_view(), name='results'),
# )
