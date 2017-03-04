from setup import *
from scraper import *

my_scraper = Strava_scraper(client_secret, access_token, strava_email, strava_password)
my_scraper.get_client()
my_scraper.load_activity_ids('act_ids_89358.csv')
my_scraper.get_activities_from_ids()

# collected_activity_ids = []
# for filename in os.listdir('activity_files'):
#     filepath = os.path.join('activity_files', filename)
#     with open(filepath) as f:
#         reader = csv.reader(f)
#         activity_ids = np.array(next(reader), dtype='int')
#         collected_activity_ids.extend(activity_ids)
#
# with open('data/act_ids_89358.csv') as f:
#     reader = csv.reader(f)
#     activity_ids = np.array(next(reader), dtype='int')
#     collected_activity_ids.extend(activity_ids)
