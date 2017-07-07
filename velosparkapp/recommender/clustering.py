import multiprocessing as mp
import pandas as pd
import numpy as np
import polyline
import copy_reg
import hdbscan
import types
import uuid
import time

from scipy.spatial.distance import pdist, squareform
# from sklearn.base import ClusterMixin
from collections import defaultdict
from functools import partial

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method) # add infrastructure to allow functions to be pickled registering it with the copy_reg standard library method

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

class PolylineClusterMaker(object):
    """A density based clustering module with filtering by location and path matching.

    Parameters
    ----------

    start_threshold : float, default: 1e-2
        Relative tolerance of the start location bin formation
    end_threshold : float, default: 1e-2
        Relative tolerance of the end location bin formation
    map_threshold : float, default: 1.0
        Relative tolerance of the map summar polyline bin formation
    verbose : int, default 0
        Verbosity mode.
    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.
    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    labels_ :
        Labels of each point
    inertia_ : float
        Sum of distances of samples to their closest cluster center. Does not calculate for label:-1
    Examples
    --------
    >>> from clustering import MapClustering
    >>> import numpy as np
    >>> df.head()
                  id                                     name  distance  moving_time  \
    0  467664896               Fat Bike in the snow at GM   12557.8       4111.0
    1  308543489  05/19/2015 Fruita, CO Horse Thief Bench   15941.8       3448.0
    2  432799746                               Lunch Ride   77276.3      10158.0
    4  308543507          05/20/2015 Fruita, CO Edge Loop   53577.9      14160.0
    5  718012436                  Lunch ride with Spencer   26395.2       4604.0

       elapsed_time  total_elevation_gain  type          start_date  \
    0        4193.0                 419.4  Ride 2016-01-10 19:13:47
    1        4356.0                 322.0  Ride 2015-05-20 00:41:46
    2       15286.0                 962.0  Ride 2015-11-14 19:21:32
    4       19563.0                1355.0  Ride 2015-05-20 15:55:46
    5        4971.0                 391.7  Ride 2016-09-19 17:04:45

         start_date_local        timezone      ...       max_heartrate  \
    0 2016-01-10 12:13:47  America/Denver      ...                 NaN
    1 2015-05-19 18:41:46  America/Denver      ...               148.0
    2 2015-11-14 12:21:32  America/Denver      ...                 NaN
    4 2015-05-20 09:55:46  America/Denver      ...               159.0
    5 2016-09-19 11:04:45  America/Denver      ...                 NaN

       athlete_id      map_id                               map_summary_polyline  \
    0     7360660  a467664896  w`gqFnpx`SDzEkDYw@wBApDdBjBGlD~@~@u@pCgJdD_AdM...
    1      113571  a308543489  scbnFjbgwSQjMzFhGnAzMJdEgBdBWtDpA`G|NnU`AxImAx...
    2       67365  a432799746  kf`sFhaiaS}KiBsAtFyI`ByeAjyBqBjNeJt@iC_EcGsAcn...
    4      113571  a308543507  ez|nFrrnvSvRzu@aKhOsw@|FzDxFoKpZlMz_@qBpL}PrCk...
    5      153343  a718012436  sojsFd_naSs@r@|BsCMgBxMaAzByD~b@{@nAsCKaPhBy@v...

       start_lat  start_lng  end_lat  end_lng     state  closest_city
    0      39.69    -105.15    39.69  -105.15  Colorado      Lakewood
    1      39.17    -108.83    39.18  -108.83  Colorado        Fruita
    2      39.98    -105.24    39.98  -105.24  Colorado       Boulder
    4      39.31    -108.71    39.31  -108.71  Colorado        Fruita
    5      40.04    -105.26    40.04  -105.26  Colorado       Boulder

    [5 rows x 40 columns]
    >>> clusterer = PolyClusterer()
    >>> clusterer.fit(df)
    >>> clusterer.labels_
    array([342,  16, 308, ..., 257,   9, 373])
    >>> clusterer.predict(#new_data#)
    ...
    >>> clusterer.cluster_centers_
    ...
    """


    def __init__(self, start_threshold=0.01, end_threshold=0.01, path_threshold=1.0):

        self.start_threshold=start_threshold
        self.end_threshold=end_threshold
        self.path_threshold=path_threshold
        self.algorithm = None
        self.labels_ = None
        self.cluster_centers_ = None

    def reduce_clusters(self, df, chunk):
        chunk_dict = dict(item for item in chunk)  # Convert back to a dict
        chunk_dict[-1] = {'indices':[], 'centroids':[]} # initial empty list value for key = -1
        for k, v in chunk_dict.items():
            if k == -1:
                # don't try to further cluster index's in the k=-1 bin
                continue
            X = df.iloc[v, [36, 37]] # Dataframe object with lats and longs
            n = X.shape[0]

            end_pts_arr = np.vstack((X.end_lat.values, X.end_lng.values))
            dsts_end_pts = squareform(pdist(end_pts_arr.T), checks=False)

            il1 = np.tril_indices(n) # lower triangle mask
            dsts_end_pts[il1] = -1

            pairs = np.argwhere((dsts_end_pts <= self.end_threshold) & (dsts_end_pts > -1))
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

                pairs = np.argwhere((dsts_maps <= self.path_threshold) & (dsts_maps is not None) & (dsts_maps > -1))
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

    def _build_clusters(self, X, method='fit'):
        data = X[['start_lat', 'start_lng']].values
        if method == 'fit':
            self.algorithm = hdbscan.HDBSCAN(min_cluster_size=2, prediction_data=True) # currently no other option are written for clustering using another method
            labels = self.algorithm.fit_predict(data) # fit to data and generate initial labels
        else:
            labels, strengths = hdbscan.approximate_predict(self.algorithm, data)

        indexes_dict = get_key_to_indexes_ddict(labels)
        minus_1 = indexes_dict.pop(-1)
        items = list(get_key_to_indexes_ddict(labels).items())
        chunksize = 4
        chunks = [items[i:i + chunksize ] for i in range(0, len(items), chunksize)]
        pool = mp.Pool(processes=4)
        results = [pool.apply_async(partial(self.reduce_clusters, X), args=(x,)) for x in chunks]
        output = [p.get() for p in results]
        output.append(get_centroids_minus_1(minus_1))

        cluster_mapper = defaultdict(list)
        if method == 'fit':
            new_label = np.max(labels)
        else:
            new_label = np.max(self.labels_)
        for chunk in output:
            for k, v in chunk.items():
                np.put(labels, v['indices'], [k] * len(v['indices']))
                if k == -1:
                    for idx, centroid in zip(v['indices'], v['centroids']):
                        new_label += 1
                        np.put(labels, [idx], [new_label])
                        cluster_mapper[new_label] = centroid
                else:
                    cluster_mapper[k] = v['centroid']
        if method == 'fit':
            self.labels_ = labels
            self.cluster_centers_ = pd.DataFrame(cluster_mapper).transpose().values
        else:
            cluster_centers_ = pd.DataFrame(cluster_mapper).transpose().values
            return labels, cluster_centers_

    def fit(self, X, y=None):
        """
        Inputs: X - dataframe containg lattiude and longitude information for original data, as well as encoded map summary polyline for each activity.
        """
        assert (X['map_summary_polyline'].isnull().sum() == 0), "all activities must have map_summary_polyline"
        assert (X['start_lat'].isnull().sum() == 0), "all activities must have start_lat"
        assert (X['start_lng'].isnull().sum() == 0), "all activities must have start_lng"
        assert (X['end_lat'].isnull().sum() == 0), "all activities must have end_lat"
        assert (X['end_lng'].isnull().sum() == 0), "all activities must have end_lng"

        global df
        df = X
        self._build_clusters(X)

    def predict(self, X, y=None):
        """
        Needs information
        """
        assert (self.labels_ is not None), "must run fit method before predict"

        return self.labels_

    def transform(self, X, y=None):
        """
        Needs information
        """
        assert (self.cluster_centers_ is not None), "must run fit method before transform"

        global df
        df = X
        return self._build_clusters(X, method='transform')

if __name__ == '__main__':

    # dataframe = pd.read_csv('data/activities_large.csv', encoding='utf-8')
    dataframe = pd.read_csv('data/activities_small.csv', encoding='utf-8')
    dataframe = dataframef[dataframe['commute'] == 0] # drop all where commute = true
    dataframe = dataframe[dataframe['state'] == 'Colorado'] # drop all where state != Colorado
    dataframe = dataframe[dataframe['type'] == 'Ride'] # drop everything that is not of type 'Ride'
    dataframe.dropna(subset=['map_summary_polyline'], inplace=True) # drop all where map_summary_polyline is nan
    dataframe.reset_index(drop=True, inplace=True)
    msk = np.random.rand(len(dataframe)) < 0.75
    train = dataframe[msk].reset_index(drop=True)
    test = dataframe[~msk].reset_index(drop=True)


    clusterer = PolylineClusterMaker()
    clusterer.fit(train)
    train_labels = clusterer.predict(train)
    test_labels, centroids = clusterer.transform(test)
