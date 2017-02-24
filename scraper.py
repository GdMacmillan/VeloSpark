from stravalib.client import Client
from stravalib.util.limiter import RateLimitRule, RateLimiter
from geopy.geocoders import Nominatim
from collections import deque
from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep

import os, re, datetime, requests
import numpy as np
import pandas as pd

# strava_email = os.environ['STRAVA_EMAIL']
# strava_password = os.environ['STRAVA_PASSWORD']

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
		self.activities = []
		self.activity_ids = []
		self.clubs = []
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

	def log_in_strava(self):
		"""
		The log_in_strava method uses a selenium webdriver to open and maintain a secure connect with Strava. It returns the driver object.
		Input: None
		Output: webdriver object
		"""
		chromeOptions = webdriver.ChromeOptions()
		prefs = {"profile.managed_default_content_settings.images":2}
		chromeOptions.add_experimental_option("prefs",prefs)

		driver = webdriver.Chrome('chromedriver.exe',chrome_options=chromeOptions)
		url = "https://www.strava.com/login"
		driver.get(url)
		user = driver.find_element_by_name('email')
		user.click()
		user.send_keys(strava_email)
		pwrd = driver.find_element_by_name('password')
		pwrd.click()
		pwrd.send_keys(strava_password)
		driver.find_element_by_id('login-button').click()
		sleep(10)
		return driver

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

	def _get_activity_by_id(self, act_id, state):
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

	def get_soup(self, driver, url):
		'''
		Helper function to get soup from a live url, as opposed to a local copy
		INPUT:
		-url: str
		OUTPUT: soup object
		'''
		driver.get(url)
		soup = BeautifulSoup(driver.page_source, 'html.parser')
		return soup

	def _make_interval_list(self):
		"""
		This helper function makes an interval list that returns a list of numbers cooresponding with a year and week number for the given year. It only returns a static list as of now but in the future could search farther back. It only goes back to week 1, 2014.
		"""
		now = datetime.datetime.now() # current date
		week_num = now.date().isocalendar()[1] # current week number
		yr_wk = {2014:52, 2015:53, 2016:52, 2017:week_num} # num of weeks each year only going back to 2014
		week_ints = [range(k * 100 + 1, k * 100 + v + 1) for k, v in yr_wk.iteritems()] # week ints in ugly nested lists
		new_week_ints = []
		for row in week_ints:
			new_week_ints.extend(row) # creates new_week_ints which is week ints flattened
		return new_week_ints

	def web_scrape_activities(self):
		"""
		This function when called will scrape strava data for athlete activity id's. It will only get those of people I follow. It will store them in a list
		page scraping example:
		https://www.strava.com/athletes/65920#interval?interval=201702&interval_type=week&chart_type=miles&year_offset=0
		where 65920 is athlete id
		201702 is the year and week num

		example:
		<div class="activity entity-details feed-entry" data-updated-at="1487545970" id="Activity-873023020" str-trackable-id="CgwIBTIICKyMpaADGAESBAoCCAE=">

		This is whats needed to grab friend activity id's. can't get BeautifulSoup to find the desired class at the moment
		"""
		driver = self.log_in_strava()
		week_ints = self._make_interval_list()

		activity_id_list = [] # need to fiill this thing

		# for athlete in athlete_list:
		# 	for yearweek num in week_ints:
		# 		get url html
		#
		# 		get_soup
		#
		# 		pull out div class="activity entity-details feed-entry"
		# 		append id's to activity_id_list

# url = "https://www.strava.com/athletes/7202879#interval?interval=201645&interval_type=week&chart_type=miles&year_offset=0"








	def main_caller(self):
		print "Getting client activities..."
		print
		self.activities.extend(list(self.client.get_activities()))
		print "Getting friend activities..."
		print
		self.activities.extend(list(self.client.get_friend_activities()))
		print "Getting athlete clubs..."
		print
		self.clubs.extend(self.client.get_athlete_clubs())
		club_ids = [club.id for club in self.clubs]
		print "Getting club activities..."
		print
		for club in club_ids:
			self.activities.extend(list(self.client.get_club_activities(club)))

		print "All done!"



	def __repr__(self):
		return "This is {} {}'s strava scraper class".format(self.my_athlete.firstname, self.my_athlete.lastname)
