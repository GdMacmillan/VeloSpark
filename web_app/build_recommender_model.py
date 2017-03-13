from sklearn import preprocessing
from clustering import *
import pandas as pd
import numpy as np
import cPickle as pickle

def get_ratings(df):
    n_labels = len(df.label.unique())
    athlete_ids = np.array(sorted(df.athlete_id.unique()))
    n_athletes = len(athlete_ids)
    ath_labels = df.groupby(['athlete_id', 'label']).count()['id'].to_dict()
    ratings = np.zeros((n_athletes, n_labels))
    for k,v in ath_labels.iteritems():
        ratings[np.where(athlete_ids==k[0])[0][0], k[1]] = v

    # scale ratings matrix to between 0-5
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 5))
    ratings_scaled = min_max_scaler.fit_transform(ratings)
    return ratings_scaled

def get_idx_to_activities_dict(df):
    idx_to_activities = {}
    df_groupby = df.groupby(['label', 'id'])['label'].agg({'Frequency':'count'}).to_dict('series')['Frequency']
    for label in np.sort(df['label'].unique()):
        idx_to_activities[label] = df_groupby[label].index.values
    return idx_to_activities

def fast_similarity(ratings, kind='item', epsilon=1e-9):
    # epsilon -> small number for handling divide-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

if __name__ == '__main__':

    # load data from clustering.py
    co_runs_df, co_rides_df = load_data()

    # append labels and return clustering models from clustering.py
    co_runs_df, co_rides_df, runs_clusterer, rides_clusterer = get_labels(co_runs_df, co_rides_df)

    # generate ratings matrix's for both runs and rides
    co_runs_ratings = get_ratings(co_runs_df)
    co_rides_ratings = get_ratings(co_rides_df)

    # generate similarity matrix's for both runs and rides
    item_similarity_runs = fast_similarity(co_runs_ratings, kind='item')
    item_similarity_rides = fast_similarity(co_rides_ratings, kind='item')

    idx_to_runs = get_idx_to_activities_dict(co_runs_df)
    idx_to_rides = get_idx_to_activities_dict(co_rides_df)

    # Export index file to return runs
    with open('data/runs_mapper.pkl', 'wb') as f:
        pickle.dump(idx_to_runs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Export index file to return runs
    with open('data/rides_mapper.pkl', 'wb') as f:
        pickle.dump(idx_to_rides, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Export run similarity matrix via numpy save
    with open('data/item_similarity_runs.npy', 'wb') as f:
        np.save(f, item_similarity_runs)

    # Export ride similarity matrix via numpy save
    with open('data/item_similarity_rides.npy', 'wb') as f:
        np.save(f, item_similarity_rides)

    # Export run clustering model via pickle
    with open('data/runs_clusterer.pkl', 'wb') as f:
        pickle.dump(runs_clusterer, f)

    # Export ride clustering model via pickle
    with open('data/rides_clusterer.pkl', 'wb') as f:
        pickle.dump(rides_clusterer, f)
