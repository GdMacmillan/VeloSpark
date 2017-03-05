import pandas as pd
from sklearn.cluster import KMeans

def load_data():
    # load data frame
    activities_df = pd.read_csv('data/activities_small.csv', encoding='utf-8')
    # remove any activites where distance = 0
    zero_dist_mask = activities_df['distance'] > 0
    activities_df = activities_df[zero_dist_mask]
    # activities_df.drop('index', 1, inplace=True) # possibly need to drip index depending on how df was stored in csv
    runs_df = activities_df[activities_df.type == 'Run']
    rides_df = activities_df[activities_df.type == 'Ride']

    # remove all entries not relevant to colorado
    co_rides_mask = rides_df['state'] == 'Colorado'
    co_runs_mask = runs_df['state'] == 'Colorado'
    co_rides_df = rides_df[co_rides_mask].reset_index(drop=True).fillna(0)
    co_runs_df = runs_df[co_runs_mask].reset_index(drop=True).fillna(0)

    return co_runs_df, co_rides_df

def get_labels(runs_df, rides_df, n_clusters_rides=550, n_clusters_runs=125):
    X_runs = runs_df[['distance', 'total_elevation_gain', 'moving_time', 'start_lat', 'start_lng']]
    X_rides = rides_df[['distance', 'total_elevation_gain', 'moving_time', 'start_lat', 'start_lng']]
    kmeans_runs = KMeans(init='k-means++', n_clusters=n_clusters_runs, n_init=10).fit(X_runs)
    kmeans_rides = KMeans(init='k-means++', n_clusters=n_clusters_rides, n_init=10).fit(X_rides)
    runs_df['label'] = pd.Series(kmeans_runs.predict(X_runs), dtype='int')
    rides_df['label'] = pd.Series(kmeans_rides.predict(X_rides), dtype='int')
    return runs_df, rides_df, kmeans_runs, kmeans_rides

if __name__ == '__main__':
    # runs_df, rides_df = load_data()
    # runs_df, rides_df = get_labels(runs_df, rides_df)
    pass
