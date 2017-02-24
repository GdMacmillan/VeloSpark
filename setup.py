# from scraper import Strava_scraper
# from old_scraper import Strava_scraper
import os, csv, psycopg2
import pandas as pd
import numpy as np
import re


# my user client secrete and access token. Not using access token for some reason. Not sure why I don't need it.
client_id = int(os.environ["STRAVA_CLIENT_ID"])
client_secret = os.environ["STRAVA_CLIENT_SECRET"]
access_token = os.environ["STRAVA_ACCESS_TOKEN"]

# conn = psycopg2.connect(dbname='rr_strava_tables', user='gmacmillan', host='localhost')

def scrape_n_activities1(scraper):
    """
    This is an older function used for scraping and gathering data before I really understood the API well. Doesn't really work.
    Input: Scraper class
    Output: None
    """
    start_id = 875708442  # used for old scraper
    end_id = start_id -1000000 # used for old scraper
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

def remap_athlete_datatypes(df, drop_identifying=True):
    """
    This method is to convert objects to numeric values when found. This pandas method is deprecated so may need to perform column operations explicitly in future.
    Input: df as DataFrame
    Output: df as DataFrame
    """

    df['city'] = df['city'].apply(lambda x: re.sub(r'[^\x00-\x7f]',r'',x) if x else 'unspecified city', 1)

    if drop_identifying:
        df.drop(['firstname', 'lastname', 'profile', 'profile_medium'], 1, inplace=True)

    otherdatatypes = {'id':'int', 'resource_state':'int'}

    for k, v in otherdatatypes.iteritems():
        df[k] = df[k].astype(v)

    return df.drop_duplicates(subset='id')

def create_athlete_df(ath_list):
    columns = ['id', 'resource_state', 'firstname', 'lastname', 'profile_medium', 'profile', 'city', 'state', 'country', 'sex', 'friend', 'follower', 'premium', 'created_at', 'updated_at']

    ath_feat_matrix = np.array([[getattr(athlete, atribute) for atribute in columns] for athlete in ath_list])

    ath_df = pd.DataFrame(ath_feat_matrix, columns=columns)
    # ath_df['friend'] = ath_df['friend'].apply(lambda x: 'no' if not x else x)
    # ath_df['follower'] = ath_df['follower'].apply(lambda x: 'no' if not x else x)
    ath_df = remap_athlete_datatypes(ath_df)
    return ath_df

def remap_activity_datatypes(df, convert_objects=True):
    """
    This method is to convert objects to numeric values when found. This pandas method is deprecated so may need to perform column operations explicitly in future.
    Input: df as DataFrame
    Output: df as DataFrame
    """
    str_cols = ['name', 'type', 'external_id']
    #
    for col in str_cols:
        df[col] = df[k].apply(lambda x: re.sub(r'[^\x00-\x7F]+','', str(x), 1))

    otherdatatypes = {'distance':'float', 'total_elevation_gain':'float', 'average_speed':'float', 'max_speed':'float'}

    for k, v in otherdatatypes.iteritems():
        df[k] = df[k].astype(v)

    if convert_objects:
        return df.convert_objects(convert_numeric=True)
    else:
        return df

def create_activity_df(act_list):
    columns = ['id', 'resource_state', 'external_id', 'upload_id', 'athlete', 'name', 'distance', 'moving_time', 'elapsed_time', 'total_elevation_gain', 'type', 'start_date', 'start_date_local', 'timezone', 'start_latlng', 'end_latlng', 'achievement_count', 'kudos_count', 'comment_count', 'athlete_count', 'photo_count', 'total_photo_count', 'map', 'trainer', 'commute', 'manual', 'private', 'flagged', 'average_speed', 'max_speed', 'average_watts', 'max_watts', 'weighted_average_watts', 'kilojoules', 'device_watts', 'has_heartrate', 'average_heartrate', 'max_heartrate']

    act_feat_matrix = np.array([[getattr(activity, atribute) for atribute in columns] for activity in act_list])

    act_df = pd.DataFrame(act_feat_matrix, columns=columns)
    act_ls = act_df['athlete'].tolist()
    act_df['athlete'] = pd.Series([ath.id for ath in act_ls])
    act_df['start_latlng'] = act_df['start_latlng'].apply(lambda x: list(x) if x else None, 1)
    act_df['end_latlng'] = act_df['end_latlng'].apply(lambda x: list(x) if x else None, 1)
    act_df['map'] = act_df['map'].apply(lambda x: {'id': x.id, 'summary_polyline': x.summary_polyline, 'resource_state': x.resource_state}, 1)
    return act_df

def pickle_the_df(df, filename):
    df.to_pickle(filename)



if __name__ == '__main__':
    # df.to_csv('path', header=True, index=False, encoding='utf-8') # utility function saves df to csv
    # my_scraper1 = Strava_scraper(client_secret, access_token)
    # main(my_scraper1)
    pass
