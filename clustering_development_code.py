import numpy as np # linear algebra
import pandas as pd # Data ETL, csv file I/O
import multiprocessing as mp
import polyline
import time
import hdbscan

from collections import defaultdict
from scipy.spatial.distance import pdist, squareform

df = pd.read_csv("data/activities_large.csv", parse_dates=["start_date", "start_date_local"])
# df = pd.read_csv("data/activities_small.csv", parse_dates=["start_date", "start_date_local"], encoding='utf-8')

df = df[df['commute'] == 0] # drop all where commute = true and store as new dataframe, df
df = df[df['state'] == 'Colorado'] # drop all where state != Colorado
df = df[df['type'] == 'Ride'] # drop everything that is not of type 'Ride' for now
df.dropna(subset=['map_summary_polyline'], inplace=True) # drop all where map_summary_polyline is nan
df.reset_index(drop=True, inplace=True)

data = df[['start_lat', 'start_lng']].values

algorithm = hdbscan.HDBSCAN(min_cluster_size=2) # create HDBSCAN cluster object to generate initial distances
labels = algorithm.fit_predict(data) # fit to data and generate initial labels
cluster_centers = np.zeros((len(labels), 7), dtype=object)

def get_key_to_indexes_ddict(labels):
    indexes = defaultdict(list)
    for index, label in enumerate(labels):
        indexes[label].append(index)
    return indexes

def polytrim(poly, diff):
    rands = np.random.choice(np.arange(0, len(poly)-2, 2), diff//2, replace=False)
    return np.delete(poly, np.hstack((rands,rands+1)))

def shorten_decoded_polyines(list_of_arrs):
    lens = list(map(len, list_of_arrs))
    diffs = [x - min(lens) for x in lens]
    new_arr_list = [polytrim(poly, diff) if diff != 0 else poly for poly, diff in zip(list_of_arrs, diffs)]
    return new_arr_list

def get_centroids_minus_1(v):
    chunk_dict = {}
    chunk_dict[-1] = {'indices':[], 'centroids':[]}

    orphan_indices = v
    orphan_poly_centroids = [[np.array(polyline.decode(df.iloc[v, 33])).flatten()] + list(df.iloc[v, 34:40].values) if isinstance(df.iloc[v, 33], str) else [df.iloc[v, 33]] + list(df.iloc[v, 34:40].values) for v in orphan_indices]

    chunk_dict[-1]['indices'].extend(orphan_indices)
    chunk_dict[-1]['centroids'].extend(orphan_poly_centroids)
    return chunk_dict

def get_centroids(k, v, final_v, chunk_dict, mean=False, list_of_arrs=None):
    if mean:
        poly_centroid = [np.mean(list_of_arrs, 0)] + list(np.mean(df.iloc[final_v, 34:38].values, 0)) + list(df.iloc[final_v[0], 38:40].values)
        orphan_indices = list(np.setdiff1d(v, final_v))
        orphan_poly_centroids = [[np.array(polyline.decode(df.iloc[v, 33])).flatten() if isinstance(df.iloc[v, 33], str) else df.iloc[v, 33]] + list(df.iloc[v, 34:40].values) for v in orphan_indices]
    else:
        poly_centroid = [np.array(polyline.decode(df.iloc[final_v, 33])).flatten() if isinstance(df.iloc[final_v, 33], str) else df.iloc[final_v, 33]] + list(df.iloc[final_v, 34:40].values)
        orphan_indices = v[1:]
        orphan_poly_centroids = [[np.array(polyline.decode(df.iloc[v, 33])).flatten()] + list(df.iloc[v, 34:40].values) if isinstance(df.iloc[v, 33], str) else [df.iloc[v, 33]] + list(df.iloc[v, 34:40].values) for v in orphan_indices]

    chunk_dict[-1]['indices'].extend(orphan_indices)
    chunk_dict[-1]['centroids'].extend(orphan_poly_centroids)
    chunk_dict[k] = {'indices': [final_v], 'centroid': poly_centroid}
    return chunk_dict


def reduce_clusters(chunk):
    chunk_dict = dict(item for item in chunk)  # Convert back to a dict
    chunk_dict[-1] = {'indices':[], 'centroids':[]} # initial empty list value for key = -1
    for k, v in chunk_dict.items():
        if k == -1:
            # don't try to cluster index's in the k=-1 bin
            continue
        X = df.iloc[v, [36, 37]] # Dataframe object with lats and longs
        n = X.shape[0]

        end_pts_arr = np.vstack((X.end_lat.values, X.end_lng.values))
        dsts_end_pts = squareform(pdist(end_pts_arr.T), checks=False)

        il1 = np.tril_indices(n) # lower triangle mask
        dsts_end_pts[il1] = -1

        pairs = np.argwhere((dsts_end_pts <= 0.01) & (dsts_end_pts > -1))
        idxs = np.array(sorted(set(pairs.flatten()))) # indices of v with the most similar endpoints
        if idxs.size != 0:
            new_v = np.array(v)[idxs]
            X = df.iloc[new_v, [33]] # Series object with map summary polyines
            n = X.shape[0]
            # maps = X.map_summary_polyline.values # array of polylines
            list_of_arrs = [np.array(polyline.decode(poly)).flatten() if isinstance(poly, str) else poly for poly in X.map_summary_polyline.values]
            list_of_arrs = shorten_decoded_polyines(list_of_arrs)
            df.iloc[new_v, [33]] = pd.Series(list_of_arrs, index=new_v) # store new polyline value arrays in the original dataframe
            dsts_maps = squareform(pdist(list_of_arrs), checks=False)
            
            il1 = np.tril_indices(n) # lower triangle mask
            dsts_maps[il1] = -1

            pairs = np.argwhere((dsts_maps <= 1.0) & (dsts_maps is not None) & (dsts_maps > -1))
            idxs = np.array(sorted(set(pairs.flatten()))) # indices of v with the most similar polylines
            if idxs.size != 0:
                final_v = np.array(new_v)[idxs]
                chunk_dict = get_centroids(k, v, final_v, chunk_dict, mean=True, list_of_arrs=list_of_arrs)
            else:
                final_v = v[0]
                chunk_dict = get_centroids(k, v, final_v, chunk_dict)
        else:
            final_v = v[0]
            chunk_dict = get_centroids(k, v, final_v, chunk_dict)
    return chunk_dict

start = time.time() # start time for function timing
# print("initial labels length: {}".format(len(labels)))
# init_mapper = get_key_to_indexes_ddict(labels)
# Break the mapper dict into 4 lists of (key, value) pairs
indexes_dict = get_key_to_indexes_ddict(labels)
minus_1 = indexes_dict.pop(-1)
items = list(indexes_dict.items())
chunksize = 4
chunks = [items[i:i + chunksize ] for i in range(0, len(items), chunksize)]
pool = mp.Pool(processes=4)
results = [pool.apply_async(reduce_clusters, args=(x,)) for x in chunks]
output = [p.get() for p in results]
output.append(get_centroids_minus_1(minus_1))

centroid_mapper = defaultdict(list)
new_label = np.max(labels)
for chunk in output:
    for k, v in chunk.items():
        np.put(labels, v['indices'], [k] * len(v['indices']))
        if k == -1:
            for idx, centroid in zip(v['indices'], v['centroids']):
                new_label += 1
                np.put(labels, [idx], [new_label])
                centroid_mapper[new_label] = centroid
        else:
            centroid_mapper[k] = v['centroid']

centroids = pd.DataFrame(centroid_mapper).transpose().values

end = time.time() # end time for function timing

print('The function ran for', end - start) # ran for 5248.5 seconds the for the large dataset, 77.5 seconds for the small dataset

# tests
# df.shape
# labels.shape # should be the same length as the dataframe
# np.unique(labels).shape
# centroids.shape # should be the same length as the unique array of labels
# np.setdiff1d(np.arange(np.max(labels)), list(centroid_mapper.keys())) # should be an empty array
# unique, counts = np.unique(labels, return_counts=True)
# test_dict = dict(zip(unique, counts)) # returns a test dict with the count of each label in the labels array
