import cPickle as pickle
import numpy as np
from clustering import get_labels, load_data
from build_recommender_model import get_ratings, fast_similarity, get_idx_to_activities_dict

def top_k_labels(similarity, mapper, label_idx, k=3):
    return [mapper[x] for x in np.argsort(similarity[label_idx,:])[:-k-1:-1]]

co_runs_df, co_rides_df = load_data()

co_runs_df, co_rides_df, runs_clusterer, rides_clusterer = get_labels(co_runs_df, co_rides_df)

co_runs_ratings = get_ratings(co_runs_df)

user_similarity_runs = fast_similarity(co_runs_ratings, kind='item')

idx_to_runs = get_idx_to_activities_dict(co_runs_df)

idx = 0

runs = top_k_labels(user_similarity_runs, idx_to_runs, idx)
