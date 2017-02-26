import pandas as pd
from sklearn.cluster import KMeans

def load_data():
    activities_df = pd.read_csv('data/activities_small.csv')
    zero_dist_mask = activities_df['distance'] == 0
    activities_df = activities_df[zero_dist_mask]
    runs_df = activities_df[activities_df.type == 'Run']
    rides_df = activities_df[activities_df.type == 'Ride']
    return runs_df, rides_df

if __name__ == '__main__':
    runs_df, rides_df = load_data()
