from setup import *
from scraper import *

my_scraper = Strava_scraper(client_secret, access_token, strava_email, strava_password)
my_scraper.get_client()
my_scraper.web_scrape_activities(start_n=376)
