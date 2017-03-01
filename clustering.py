import pandas as pd
from sklearn.cluster import KMeans

def load_data():
    activities_df = pd.read_csv('data/activities_small.csv', encoding='utf-8')
    zero_dist_mask = activities_df['distance'] > 0
    activities_df = activities_df[zero_dist_mask]
    activities_df.drop('index', 1, inplace=True)
    runs_df = activities_df[activities_df.type == 'Run']
    rides_df = activities_df[activities_df.type == 'Ride']

    co_rides_mask = rides_df['state'] == 'Colorado'
    co_runs_mask = runs_df['state'] == 'Colorado'
    rides_co_df = rides_df[co_rides_mask].reset_index(drop=True)
    runs_co_df = runs_df[co_runs_mask].reset_index(drop=True)

    return runs_co_df, rides_co_df

def get_labels(runs_df, rides_df, n_clusters=50):
    X_runs = runs_df[['distance', 'total_elevation_gain', 'start_lat', 'start_lng', 'end_lat', 'end_lng']]
    X_rides = rides_df[['distance', 'total_elevation_gain', 'start_lat', 'start_lng', 'end_lat', 'end_lng']]
    kmeans_runs = KMeans(init='random', n_clusters=n_clusters, n_init=10).fit(X_runs)
    kmeans_rides = KMeans(init='random', n_clusters=n_clusters, n_init=10).fit(X_rides)
    runs_df['label'] = pd.Series(kmeans_runs.predict(X_runs), dtype='int')
    rides_df['label'] = pd.Series(kmeans_rides.predict(X_rides), dtype='int')
    return runs_df, rides_df

if __name__ == '__main__':
    runs_df, rides_df = load_data()
    runs_df, rides_df = get_labels(runs_df, rides_df)
