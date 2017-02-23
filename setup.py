from scraper import Strava_scraper
import os, csv, psycopg2
import pandas as pd
import numpy as np

# my user client secrete and access token. Not using access token for some reason. Not sure why I don't need it.
client_id = int(os.environ["STRAVA_CLIENT_ID"])
client_secret = os.environ["STRAVA_CLIENT_SECRET"]
access_token = os.environ["STRAVA_ACCESS_TOKEN"]

# conn = psycopg2.connect(dbname='rr_strava_tables', user='gmacmillan', host='localhost')

def main(scraper):
    start_id = 875708442
    end_id = start_id -1000000
    scraper.get_client()
    scraper.get_n_activities(start_id, end_id, n=15000)

def write_list_to_csv(my_list, filename):
    """
    This method will write a list of items to a csv filename
    Input: my_list as list, filename as string
    Output: none
    """
    my_file = open(filename, 'wb')
    wr = csv.writer(my_file)
    wr.writerow(my_list)

def create_athlete_df(ath_list):
    columns = ['id', 'resource_state', 'firstname', 'lastname', 'profile_medium', 'profile', 'city', 'state', 'country', 'sex', 'friend', 'follower', 'premium', 'created_at', 'updated_at']

    ath_feat_matrix = np.array([[getattr(athlete, atribute) for atribute in columns] for athlete in ath_list])

    ath_df = pd.DataFrame(ath_feat_matrix, columns=columns)
    return ath_df

def pickle_the_df(df, filename):
    df.to_pickle(filename)

if __name__ == '__main__':
    my_scraper = Strava_scraper(client_secret, access_token)
    main(my_scraper)
