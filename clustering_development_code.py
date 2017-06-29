import numpy as np # linear algebra
import pandas as pd # Data ETL, csv file I/O
import multiprocessing as mp
import polyline
import time
import hdbscan

from collections import defaultdict

# df = pd.read_csv("data/activities_large.csv", parse_dates=["start_date", "start_date_local"])
df = pd.read_csv("data/activities_small.csv", parse_dates=["start_date", "start_date_local"])

df = df[df['commute'] == 0] # drop all where commute = true and store as new dataframe, df
df = df[df['state'] == 'Colorado'] # drop all where state != Colorado
df = df[df['type'] == 'Ride'] # drop everything that is not of type 'Ride' for now
df.dropna(subset=['map_summary_polyline'], inplace=True) # drop all where map_summary_polyline is nan

data = df[['start_lat', 'start_lng']].values

algorithm = hdbscan.HDBSCAN(min_cluster_size=2) # create HDBSCAN cluster object to generate initial distances
labels = algorithm.fit_predict(data) # fit to data and generate initial labels

def get_key_to_indexes_ddict(labels):
    indexes = defaultdict(list)
    for index, label in enumerate(labels):
        indexes[label].append(index)
    return indexes

def polytrim(poly1, poly2):
    diff = len(poly1) - len(poly2)
    rands = np.random.choice(np.arange(0, len(poly1)-2, 2), diff//2, replace=False)
    return np.delete(poly1, np.hstack((rands,rands+1)))

def conditional_dist(poly1, poly2):
    return poly1, poly2
    # poly1 = np.array(polyline.decode(poly1[1])).flatten()
    # poly2 = np.array(polyline.decode(poly2[1])).flatten()
    # if len(poly1) > len(poly2):
    #     poly1 = polytrim(poly1, poly2)
    # elif len(poly1) < len(poly2):
    #     poly2 = polytrim(poly2, poly1)
    # some_dict[i1] = poly1
    # some_dict[i2] = poly2
    # return np.linalg.norm(np.subtract(poly1, poly2)) # return distance between polylines

vfunc = np.vectorize(conditional_dist)

some_dict = {}

def reduce_clusters(chunk):
    chunk_dict = dict(item for item in chunk)  # Convert back to a dict
    chunk_dict[-1] = [] # initial empty list value for key = -1
    for k, v in chunk_dict.items():
        if k == -1:
            # don't try to further cluster index's in the k=-1 bin
            continue
        X = df.iloc[v, [36, 37]] # Dataframe object with lats and longs
        n = X.shape[0]
        lats = X.end_lat.values # array of end latitudes
        lngs = X.end_lng.values # array of end longitudes

        dsts_end_pts = np.zeros((n, n)) # empty distance array
        # populate distance array with dists between end lats and longs
        for i, (lt, lg) in enumerate(zip(lats, lngs)):
            dsts_end_pts[i] = np.sqrt((lats - lt)**2 + (lngs - lg)**2)
        il1 = np.tril_indices(n) # lower triangle mask
        dsts_end_pts[il1] = -1

        pairs = np.argwhere((dsts_end_pts <= 0.01) & (dsts_end_pts > -1))
        idxs = np.array(sorted(set(pairs.flatten()))) # indices of v with the most similar endpoints
        if idxs.size != 0:
            new_v = np.array(v)[idxs]
            X = df.iloc[new_v, [33]] # Series object with map summary polyines
            n = X.shape[0]
            maps = list(X.to_records()) # array of polylines

            dsts_maps = np.zeros((n, n)) # empty distance array
            # populate distance array
            for poly in maps:
                dsts_maps[i] = vfunc(maps, poly)
            break
            il1 = np.tril_indices(n) # lower triangle mask
            dsts[il1] = -1

            pairs = np.argwhere((dsts_maps <= 1.0) & (dsts_maps is not None) & (dsts_maps > -1))
            idxs = np.array(sorted(set(pairs.flatten()))) # indices of v with the most similar polylines
            if idxs.size != 0:
                final_v = np.array(new_v)[idxs]
                chunk_dict[-1].extend(list(np.setdiff1d(v, final_v)))
                chunk_dict[k] = list(final_v)
        else:
            chunk_dict[-1].extend(v[1:])
            chunk_dict[k] = [v[0]]
    return chunk_dict

start = time.time() # start time for function timing

# mapper = get_key_to_indexes_ddict(labels)
# Break the mapper dict into 4 lists of (key, value) pairs
items = list(get_key_to_indexes_ddict(labels).items())
chunksize = 4
chunks = [items[i:i + chunksize ] for i in range(0, len(items), chunksize)]
pool = mp.Pool(processes=4)
results = [pool.apply_async(reduce_clusters, args=(x,)) for x in chunks]
output = [p.get() for p in results]


# the following code creates a mapper and updates the labels array with the correct labels to the indices specified by mapper. mapper may not be necessary as object of this function is to supply the output labels of the fit_predict method. using mapper might be necessary to create cluster centroids which can be stored in the appropriately named class attribute for this clusterer

mapper = defaultdict(list)
mapper.update(output[0])
for chunk in output:
    for k, v in chunk.items():
        np.put(labels, v, [k] * len(v))

    mapper[-1].extend(chunk.pop(-1, None))
    mapper.update(chunk)

end = time.time() # end time for function timing

print('The function ran for', end - start) # ran for 5248.5 seconds the for the large dataset, 77.5 seconds for the small dataset
