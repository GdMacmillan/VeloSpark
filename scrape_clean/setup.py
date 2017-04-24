from math import pi
import os, csv, json
import pandas as pd
import numpy as np
import reverse_geocoder as rg
import re, stravalib

# my user client secrete and access token. Not using access token for some reason. Not sure why I don't need it.

try:
    client_secret = os.environ["STRAVA_CLIENT_SECRET"]
    access_token = os.environ["STRAVA_ACCESS_TOKEN"]
    strava_email = os.environ['STRAVA_EMAIL']
    strava_password = os.environ['STRAVA_PASSWORD']
except:
    with open('aws/strava.json') as f:
    	data = json.load(f)
        client_secret = data["STRAVA_CLIENT_SECRET"]
        access_token = data["STRAVA_ACCESS_TOKEN"]
        strava_email = data['STRAVA_EMAIL']
        strava_password = data['STRAVA_PASSWORD']

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
    """
    This method creates an athletes DataFrame from a list of raw athlete objects from the stravalib api library. It uses remap_athlete_datatypes to get data into format safe for csv conversion.
    Input: ath_list as list
    Output: ath_df as DataFrame
    """
    columns = ['id', 'resource_state', 'firstname', 'lastname', 'profile_medium', 'profile', 'city', 'state', 'country', 'sex', 'friend', 'follower', 'premium', 'created_at', 'updated_at']

    ath_feat_matrix = np.array([[getattr(athlete, atribute) for atribute in columns] for athlete in ath_list])
    ath_df = pd.DataFrame(ath_feat_matrix, columns=columns)
    ath_df = ath_df.drop_duplicates(subset='id')
    ath_df = remap_athlete_datatypes(ath_df)
    return ath_df

def remap_activity_datatypes(df):
    """
    This method is to convert objects to numeric values when found. This pandas method is deprecated so may need to perform column operations explicitly in future.
    Input: df as DataFrame
    Output: df as DataFrame
    """
    str_cols = ['name', 'type']

    for col in str_cols:
        df[col] = df[col].apply(lambda x: re.sub(r'[^\x00-\x7F]+',r'',x) if x else 'unspecified {}'.format(col), 1)

    time_delta_cols = ['moving_time', 'elapsed_time']

    for col in time_delta_cols:
        df[col] = df[col].apply(lambda x: x.total_seconds(), 1)

    otherdatatypes = {'id':'int', 'distance':'float', 'total_elevation_gain':'float', 'achievement_count':'int', 'kudos_count':'int', 'comment_count':'int', 'athlete_count':'int', 'photo_count':'int', 'total_photo_count':'int', 'trainer':'bool', 'commute':'bool', 'manual':'bool', 'private':'bool', 'flagged':'bool', 'average_speed':'float', 'max_speed':'float', 'average_watts':'float', 'max_watts':'float', 'weighted_average_watts':'float', 'kilojoules':'float', 'device_watts':'bool', 'has_heartrate':'bool', 'average_heartrate':'float', 'max_heartrate':'float', 'athlete_id': 'int'}

    for k, v in otherdatatypes.iteritems():
        df[k] = df[k].astype(v)

    return df

def create_activity_df(act_list):
    """
    This method creates an activities DataFrame from a list of raw activity objects from the stravalib api library. It uses remap_activity_datatypes to get data into format safe for csv conversion.
    Input: act_list as list
    Output: act_df as DataFrame
    """

    columns = ['id', 'resource_state', 'external_id', 'upload_id', 'athlete', 'name', 'distance', 'moving_time', 'elapsed_time', 'total_elevation_gain', 'type', 'start_date', 'start_date_local', 'timezone', 'start_latlng', 'end_latlng', 'achievement_count', 'kudos_count', 'comment_count', 'athlete_count', 'photo_count', 'total_photo_count', 'map', 'trainer', 'commute', 'manual', 'private', 'flagged', 'average_speed', 'max_speed', 'average_watts', 'max_watts', 'weighted_average_watts', 'kilojoules', 'device_watts', 'has_heartrate', 'average_heartrate', 'max_heartrate']

    act_feat_matrix = np.array([[getattr(activity, atribute) for atribute in columns] for activity in act_list])
    act_df = pd.DataFrame(act_feat_matrix, columns=columns)

    act_df = act_df.drop_duplicates(subset='id')
    # create new column athlete containing athlete id
    act_df['athlete_id'] = pd.Series([athlete.id for athlete in act_df.athlete.values])
    # create new column map_id
    act_df['map_id'] = pd.Series([map.id if type(map) == stravalib.model.Map else None for map in act_df.map.values])
    # create new column map_summary_polyline
    act_df['map_summary_polyline'] = pd.Series([strava_map.summary_polyline if type(strava_map) == stravalib.model.Map else None for strava_map in act_df.map.values])
    # create new columns for start latitude, longitude and end latitude, longitude for the stravalib Latlon attribute
    act_df['start_lat'] = pd.Series([start[0] if type(start) == stravalib.attributes.LatLon else float('nan') for start in act_df.start_latlng.values])
    act_df['start_lng'] = pd.Series([start[1] if type(start) == stravalib.attributes.LatLon else float('nan') for start in act_df.start_latlng.values])
    act_df['end_lat'] = pd.Series([end[0] if type(end) == stravalib.attributes.LatLon else float('nan') for end in act_df.end_latlng.values])
    act_df['end_lng'] = pd.Series([end[1] if type(end) == stravalib.attributes.LatLon else float('nan') for end in act_df.end_latlng.values])

    # drop rows where athlete id is nan
    act_df = act_df[act_df['athlete_id'].fillna(0.0) > 0]
    # drop rows where gps data is null
    act_df = act_df[act_df['start_lat'].fillna(0.0) > 0]
    # drop columns that we don't potentially need
    act_df.drop(['athlete', 'upload_id', 'resource_state', 'external_id', 'start_latlng', 'end_latlng', 'map'], 1, inplace=True)

    act_df = remap_activity_datatypes(act_df)
    return act_df

# @rate_limited(1)
def get_state(lat, lng):
    results = rg.search((lat, lng))
    return results[0]['admin1']

def add_state_feature(act_df):
    act_df['state'] = pd.Series([get_state(start_lat, start_lng) for start_lat, start_lng in zip(act_df.start_lat.values, act_df.start_lng.values)])
    return act_df

def get_distance(lat1, lon1, lat2, lon2):
    radius = 6367000 # meters
    x = np.pi/180.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius*c

def add_closest_city_feature(act_df):
    colorado_cities = pd.read_csv('data/colorado_cities.csv', encoding='utf-8')

    x = pi/180.0
    lat1 = act_df.start_lat.values
    lon1 = act_df.start_lng.values
    lat2 = colorado_cities.city_lat.values.reshape((colorado_cities.shape[0],1))
    lon2 = colorado_cities.city_lng.values.reshape((colorado_cities.shape[0],1))


    a = (90.0-lat1)*(x)
    b = (90.0-lat2)*(x)
    theta = (lon2-lon1)*(x)
    c = np.arccos((np.cos(a)*np.cos(b)) + (np.sin(a)*np.sin(b)*np.cos(theta)))
    # cities = colorado_cities.city.values.reshape((colorado_cities.shape[0],1))
    cities = colorado_cities.city.values
    def f(x):
        return cities[x]
    func = np.vectorize(f)
    sorted_cities = func(np.argsort(c, axis=0))
    act_df['closest_city'] = pd.Series(sorted_cities[0, :])
    return act_df

def pickle_the_df(df, filename):
    df.to_pickle(filename)



if __name__ == '__main__':
    # df.to_csv('path', header=True, index=False, encoding='utf-8') # utility function saves df to csv
    # my_scraper = Strava_scraper(client_secret, access_token, strava_email, strava_password)

    # ssh -i ~/.ssh/my_key_pair_001.pem ubuntu@34.197.37.134
    #
    # scp -i ~/.ssh/my_key_pair_001.pem ubuntu@34.197.37.134:aws/act_df_needs_state_feature_20640.csv .
    #
    # ssh -i ~/.ssh/my_key_pair_001.pem ubuntu@52.91.211.58
    #
    # scp -i ~/.ssh/my_key_pair_001.pem act_df_needs_state_feature_20640.csv ubuntu@52.91.211.58:aws/
    #
    # scp -i ~/.ssh/my_key_pair_001.pem scrape_clean/setup.py ubuntu@52.91.211.58:aws/
    pass
