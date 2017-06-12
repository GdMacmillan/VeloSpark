import json
import os

from django.conf import settings
from django.http import Http404
from django.shortcuts import render
from django.template import Template, Context
from django.template.loader_tags import BlockNode
from django.utils._os import safe_join

from allaccess.views import OAuthCallback

from api.models import Athlete
from api.models import StravaUser


def get_page_or_404(name):
    """Return page content as a Django template or raise 404 error."""
    try:
        file_path = safe_join(settings.SITE_PAGES_DIRECTORY, name)
    except ValueError:
        raise Http404('Page Not Found')
    else:
        if not os.path.exists(file_path):
            raise Http404('Page Not Found')

    with open(file_path, 'r') as f:
        page = Template(f.read())
    meta = None
    for i, node in enumerate(list(page.nodelist)):
        if isinstance(node, BlockNode) and node.name == 'context':
            meta = page.nodelist.pop(i)
            break
    page._meta = meta
    return page

def page(request, slug='index'):
    """Render the requested page if found."""
    file_name = '{}.html'.format(slug)
    page = get_page_or_404(file_name)
    context = {
        'slug': slug,
        'page': page,
    }
    if page._meta is not None:
    	meta = page._meta.render(Context())
    	extra_context = json.loads(meta)
    	context.update(extra_context)

    if request.user.is_authenticated():
        try:
            access = request.user.accountaccess_set.all()[0]
        except IndexError:
            access = None
        else:
            client = access.api_client

            context['info'] = client.get_profile_info(raw_token=access.access_token)
            # context['username'] = context['info']['username']
            # athlete = Athlete()
            profile_info = client.get_profile_info(raw_token=access.access_token)
            strava_user_id = getattr(request.user, 'id')
            strava_user = StravaUser.objects.get(pk=strava_user_id)
            # athlete = model(**strava_user.cleaned_data)

            athlete = None
            try:
                athlete = Athlete.objects.get(id=profile_info['id'])
            except Athlete.DoesNotExist:
                athlete = Athlete(id=profile_info['id'])
                athlete.user = StravaUser.objects.get(pk=strava_user_id)

            athlete.deserialize(profile_info)

            athlete.save()

            # athlete.user = strava_user
            # athlete.firstname = profile_info['firstname']
            # athlete.lastname = profile_info['lastname']
            # athlete.resource_state = profile_info['resource_state']



            # context['firstname'] = profile_info['firstname']
            # strava_user.firstname = profile_info['firstname']
            # strava_user.save()
            # athlete = Athlete.objects.get(pk=getattr(request.user, 'id'))
            # athlete.firstname =
            # athlete.save()

        # TODO: need to step get list of activities and put them in to database if they are not already there the following shows how to get one page of data from strava api:

        # if slug == 'test':
        #     params = {'page': 0}
        #     url = 'https://www.strava.com/api/v3/athlete/activities'
            # context['test'] = client.request('get', url, params=params)

    return render(request, 'page.html', context)
