import pandas as pd
from sklearn.cluster import KMeans

def load_data():
    activities_df = pd.read_csv('data/activities_small.csv', encoding='utf-8')
    zero_dist_mask = activities_df['distance'] > 0
    activities_df = activities_df[zero_dist_mask]
    activities_df.drop('index', 1, inplace=True)
    runs_df = activities_df[activities_df.type == 'Run']
    rides_df = activities_df[activities_df.type == 'Ride']
    return runs_df, rides_df

def get_labels(runs_df, rides_df, n_clusters):
    



if __name__ == '__main__':
    runs_df, rides_df = load_data()
