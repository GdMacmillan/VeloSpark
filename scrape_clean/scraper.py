from stravalib.client import Client
from stravalib.util.limiter import RateLimitRule, RateLimiter
from collections import deque
from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep
from setup import write_list_to_csv
from urllib2 import HTTPError

import os, re, datetime, requests, csv
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
	def __init__(self, client_secret, access_token, strava_email, strava_password):
		self.client_secret = client_secret
		self.access_token = access_token
		self.strava_email = strava_email
		self.strava_password = strava_password
		self.client = None
		self.athlete = None
		self.friends = None # list of my friends, dtype = stravalib object
		self.activity_ids = [] # list of activity ids scraped from strava
		self.friend_ids = []
		self.activities = [] # list of activities
		self.clubs = [] # list of athlete clubs
		self.other_athletes = [] # list of other athlete objects unfollowed by client

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

		print "logging in..."
		print
		driver = webdriver.Chrome(chrome_options=chromeOptions)
		url = "https://www.strava.com/login"
		driver.get(url)
		user = driver.find_element_by_name('email')
		user.click()
		user.send_keys(self.strava_email)
		pwrd = driver.find_element_by_name('password')
		pwrd.click()
		pwrd.send_keys(self.strava_password)
		driver.find_element_by_id('login-button').click()
		sleep(10)
		print "complete!"
		return driver

	def _get_activity_by_id(self, act_id):
		try:
			activity = self.client.get_activity(act_id) # get id with id = act_id from strava client
			return activity
		except HTTPError:
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

	def _get_activities_from_page(self, soup):
		temp_act_id_list = []
		regex = re.compile('/activities/([0-9]*)') # compile regex function
		for link in soup.find_all('a'):
			text = link.get('href')
			try:
				act_id = regex.findall(text) # look for digits after '/activities/'. Stop upon any character not a number. only looking for 1st item found. should be unicode string.
				try: # only looking for integers 9 digits long
					temp_act_id_list.append(int(act_id[0])) # append number to small list
					# print act_id[0]
				except (IndexError, ValueError):
					continue
			except TypeError:
				continue
		return temp_act_id_list

	def web_scrape_activities(self, start_n=0, sleep=False, sleep_time=2):
		"""
		This function when called will scrape strava data for athlete activity id's. It will only get those of people I follow. It will store them in a list
		Example url:
		https://www.strava.com/athletes/2304253#interval?interval=201631&interval_type=week&chart_type=miles&year_offset=0
		where 2304253 is athlete id
		201631 is the year and week num

		This is whats needed to find and parse html from athlete pages and grab activity id's.
		Example tag:
		<a href="/activities/666921221">And the winning number is 400</a> ==$0
		"""
		driver = self.log_in_strava()
		week_ints = self._make_interval_list()

		print "scraping athletes"
		for ath_id in self.friend_ids[start_n:]: #starting on index 191, athlete 66299
			athlete_act_id_list = []
			for yearweek_int in week_ints:
				url = "https://www.strava.com/athletes/{}#interval?interval={}&interval_type=week&chart_type=miles&year_offset=0".format(str(ath_id),str(yearweek_int))
				soup = self.get_soup(driver, url)
				# self.activity_ids.extend(self._get_activities_from_page(soup))
				# print "added {}'s {} intervals to list".format(ath_id, yearweek_int)
				if sleep:
					sleep(np.random.exponential(1.0) * sleep_time) # pause for amount of sleep time before completing each loop
				athlete_act_id_list.extend(self._get_activities_from_page(soup))
			filename = "{}_act_ids.csv".format(ath_id)
			filepath = os.path.join('activity_files', filename)
			write_list_to_csv(athlete_act_id_list, filepath)

		self.activity_ids = set(self.activity_ids)

		print "All done!"

	def get_other_athletes(self, list_ath_ids):
		"""
		This utility function is provided to populate a list of other athletes. It requires a list of predifined athlete id's.
		Input: list_ath_ids as list
		Output: None
		"""
		print "Getting other athletes..."
		print
		for ath_id in list_ath_ids:
			if ath_id in self.friend_ids:
				continue
			else:
				athlete = self.client.get_athlete(ath_id)
				self.other_athletes.append(athlete)
		print "All done!"

	def load_activity_ids(self, act_id_csv_filename):
		"""
		This utility function should only be called to populate the class attribute 'activity_ids' from a csv when a new scraper has been instantiated
		"""
		with open(act_id_csv_filename) as f:
  			reader = csv.reader(f)
  			self.activity_ids = np.array(next(reader), dtype='int')

	def get_activities_main(self):
		"""
		This function when called after get client function will populate list attributes for class. This may be done when client wants all(last 200 for feeds) things associated with their athlete, friends, and clubs
		Input: None
		Output: None
		"""
		print "Getting client activities..."
		print
		self.activities.extend(list(self.client.get_activities())) # gets all
		print "Getting friend activities..."
		print
		self.activities.extend(list(self.client.get_friend_activities())) # only gets last 200 activities from users feed
		print "Getting athlete clubs..."
		print
		self.clubs.extend(self.client.get_athlete_clubs()) # gets all
		club_ids = [club.id for club in self.clubs]
		print "Getting club activities..."
		print
		for club in club_ids:
			self.activities.extend(list(self.client.get_club_activities(club))) # gets last 200 activities per club

		print "All done!"

	def get_activities_from_ids(self):
		requested_activity = None
		while len(self.activity_ids) > 0:
			requested_activity = self._get_activity_by_id(self.activity_ids[0])
			if requested_activity:
				self.activities.append(requested_activity)
			self.activity_ids = self.activity_ids[1:]

	def __repr__(self):
		return "This is {} {}'s strava scraper class".format(self.athlete.firstname, self.athlete.lastname)
