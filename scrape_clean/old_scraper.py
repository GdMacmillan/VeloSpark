from stravalib.client import Client
from stravalib.util.limiter import RateLimitRule, RateLimiter
from geopy.geocoders import Nominatim
from collections import deque
from requests import HTTPError
from bs4 import BeautifulSoup
import os, re, datetime
import numpy as np
import pandas as pd

class DefaultRateLimiter(RateLimiter):
	"""
	Implements something similar to the default rate limit for Strava apps.
	To do this correctly we would actually need to change our logic to reset
	the limit at midnight, etc.  Will make this more complex in the future.
	Strava API usage is limited on a per-application basis using a short term,
	15 minute, limit and a long term, daily, limit. The default rate limit allows
	600 requests every 15 minutes, with up to 30,000 requests per day.
	"""

	def __init__(self):
		super(DefaultRateLimiter, self).__init__()
		self.rules.append(RateLimitRule(requests=40, seconds=60, raise_exc=False))
		self.rules.append(RateLimitRule(requests=30000, seconds=(3600 * 24), raise_exc=True))


class Strava_scraper(object):
	'''
	A strava scraper class.
	'''
	def __init__(self, client_secret, access_token):
		self.client_secret = client_secret
		self.access_token = access_token
		self.client = None
		self.athlete = None
		self.friends = None # list of my friends, dtype = stravalib object
		self.friend_ids = []
		self.friend_activities = []
		self.athlete_ids = [] # not used
		self.list_of_athletes = [] # not used

	def get_client(self):
		"""
		The get_client method create a client object for making requests to the strava API. The Client class accepts an access_token and a rate_limiter object. The method also populates a friends list
		Inputs: None
		Outputs: None
		"""
		self.client = Client(access_token=self.access_token, rate_limiter=DefaultRateLimiter())

		self.athlete = self.client.get_athlete() # Get Gordon's full athlete

		print "Client setup complete!"
		print
		self.friends = list(self.client.get_athlete_friends()) # Get athlete Gordon's friends
		print "Authenticated user's friends list complete!"
		print
		for friend in self.friends:
			self.friend_ids.append(friend.id)

	def _check_for_id(self, id):
		"""
		The _check_for_id method checks both the friends_ids and athlete_ids class attributes for the input id number.
		Inputs: id as integer
		Outputs: None
		"""
		return True if (id in self.friend_ids) or (id in self.athlete_ids) else False

	def get_n_athletes(self, n):
		"""
		The get_n_athletes method is deprecated because Strava no longer allows authenticated users to indiscriminantly pull activities for a particular user if they are not friends with the authenticated user. This means that by creating a large list of athletes without following them does not help get activities for those atheletes. It would take one parameter n which is the limit on the ammount of athletes to get.
		Input: n as integer
		Output: None
		"""
		athlete_deq = deque(self.friend_ids, maxlen=n)
		id_deq = deque(self.friend_ids, maxlen=n)
		num = 0
		while len(self.athlete_ids) + len(self.friend_ids) < n:
			athlete_id = id_deq.popleft()
			athlete = athlete_deq.popleft()
			for i in range(100): # try one hundred times
				while True:
					try:
						athletes_friends = self.client.get_athlete_friends(athlete_id)
					except ConnectionError:
						continue
					break
			for friend in athletes_friends:
				athlete_deq.append(friend)
				id_deq.append(friend.id)

			if not self._check_for_id(athlete_id):
				self.athlete_ids.append(athlete_id)
				self.list_of_athletes.append(athlete)
				firstname = re.sub(r'[^\x00-\x7F]+','', athlete.firstname)
				lastname = re.sub(r'[^\x00-\x7F]+','', athlete.lastname)
				print "athlete '{} {}' added to list position {}".format(firstname, lastname, num)
				num += 1

	def _get_state(self, latlng):
		if latlng:
			geoc = Nominatim()
			location = geoc.reverse(latlng)
			state = None
			try:
				state = location.raw['address']['state']
			 	return state
			except KeyError:
				pass


	def _get_activity(self, act_id, state):
		try:
			activity = self.client.get_activity(act_id) # get id with id = act_id from strava client
		except HTTPError:
			print "id:{}; client HTTP error when getting activity!!!".format(act_id)
			return None
		latlng = activity.start_latlng
		if not latlng:
			return None
		act_ath_id = activity.athlete.id
		firstname = re.sub(r'[^\x00-\x7F]+','', activity.athlete.firstname)
		lastname = re.sub(r'[^\x00-\x7F]+','', activity.athlete.lastname)
		act_state = self._get_state(list(latlng))
		if act_ath_id in self.friend_ids and act_state == state:
			print "activity id: {} belonging to {} {}, added to list".format(act_id, firstname, lastname)
			return activity
		else:
			print "activity {} not a gps coordinated activity or not in state.".format(act_id)
			return None

	def get_n_activities(self, start_id, end_id, state='Colorado', n=30000):
		"""
		The get_friends_activities method takes 2 parameters. The state which is the subset of the data to save to the self.activities attribute of the scraper class and n which is the max number of entries to add to the list. The default state is 'Colorado'.
		Input: state as string, n as int
		Output: None
		"""
		print "Getting activities starting with id: {}".format(start_id)
		print "_" * 50
		act_id = start_id
		while len(self.friend_activities) <= n and act_id >= end_id:
			activity = self._get_activity(act_id, state)
			if activity:
				self.friend_activities.append(activity)
			act_id -= 1

	def __repr__(self):
		return "This is {} {}'s strava scraper class".format(self.my_athlete.firstname, self.my_athlete.lastname)
