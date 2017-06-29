import multiprocessing as mp
import pandas as pd
import numpy as np
import polyline
import hdbscan
import uuid
import time

from sklearn.base import ClusterMixin
from collections import defaultdict
from functools import partial

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
    poly1 = np.array(polyline.decode(poly1)).flatten()
    poly2 = np.array(polyline.decode(poly2)).flatten()
    if len(poly1) > len(poly2):
        poly1 = polytrim(poly1, poly2)
    elif len(poly1) < len(poly2):
        poly2 = polytrim(poly2, poly1)
    return np.linalg.norm(np.subtract(poly1, poly2)) # return distance between polylines

class PolylineClusterMaker(ClusterMixin):
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


    def __init__(self, start_threshold=0.01, end_threshold=0.01, path_threshold=1.0, algorithm='hdbscan'):

        self.start_threshold=start_threshold
        self.end_threshold=end_threshold
        self.path_threshold=path_threshold
        self.algorithm = algorithm
        self.labels_ = None
        self.cluster_centers_ = None

    def reduce_clusters(self, df, vfunc, chunk):
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

            pairs = np.argwhere((dsts_end_pts <= self.end_threshold) & (dsts_end_pts > -1))
            idxs = np.array(sorted(set(pairs.flatten()))) # indices of v with the most similar endpoints
            if idxs.size != 0:
                new_v = np.array(v)[idxs]
                X = df.iloc[new_v, [33]] # Series object with map summary polyines
                n = X.shape[0]
                maps = X.map_summary_polyline.values # array of polylines

                dsts_maps = np.zeros((n, n)) # empty distance array
                # populate distance array
                for i, poly in enumerate(maps):
                    dsts_maps[i] = vfunc(poly, maps)
                il1 = np.tril_indices(n) # lower triangle mask
                dsts_maps[il1] = -1

                pairs = np.argwhere((dsts_maps <= self.path_threshold) & (dsts_maps is not None) & (dsts_maps > -1))
                idxs = np.array(sorted(set(pairs.flatten()))) # indices of v with the most similar polylines
                if idxs.size != 0:
                    final_v = np.array(new_v)[idxs]
                    # TODO: create array of cluster_centers_
                    # X = df.iloc[final_v, [33, 36, 37]].values
                    # X[:, 0]
                    # cluster_polyline =
                    chunk_dict[-1].extend(list(np.setdiff1d(v, final_v)))
                    chunk_dict[k] = list(final_v)
            else:
                chunk_dict[-1].extend(v[1:])
                chunk_dict[k] = [v[0]]
        return chunk_dict

    def _build_clusters(self, X, predict=True):
        vfunc = np.vectorize(conditional_dist)
        data = X[['start_lat', 'start_lng']].values

        if self.algorithm == 'hdbscan':
            algorithm = hdbscan.HDBSCAN(min_cluster_size=2)

        labels = algorithm.fit_predict(data) # fit to data and generate initial labels
        items = list(get_key_to_indexes_ddict(labels).items())
        chunksize = 4
        chunks = [items[i:i + chunksize ] for i in range(0, len(items), chunksize)]
        pool = mp.Pool(processes=4)
        results = [pool.apply_async(partial(self.reduce_clusters, X, vfunc), args=(x,)) for x in chunks]
        output = [p.get() for p in results]

        for chunk in output:
            for k, v in chunk.items():
                np.put(labels, v, [k] * len(v))

        self.labels_ = labels



        # The below code is the original _build_clusters function

        # if not predict:
        #     new_idxs = X[X.activity_id.isnull()].index
        #
        # lats = X.start_lat.values # array of start latitudes
        # lngs = X.start_lng.values # array of start longitudes
        #
        #
        # #block of code to time here
        # dsts = [] # empty distance array
        # # populate distance array
        # for (lt, lg) in izip(lats, lngs):
        #     dsts.append(np.sqrt((lats - lt)**2 + (lngs - lg)**2))
        # dsts = np.array(dsts) # convert to numpy array
        # il1 = np.tril_indices(dsts.shape[0]) # lower triangle mask
        # dsts[il1] = -1
        #
        # pairs = np.argwhere((dsts <= self.start_threshold) & (dsts > -1)) # all pairs of indices with distances between start locations shorter than threshold of 0.01
        # start_clusters = self.make_indexer(pairs)
        #
        # subset_pairs = []
        # for idx1, other_idxs in start_clusters.iteritems():
        #     for idx2 in other_idxs:
        #         poly1 = np.array(polyline.decode(X.ix[idx1, 'map_summary_polyline'])).flatten()
        #         poly2 = np.array(polyline.decode(X.ix[idx2, 'map_summary_polyline'])).flatten()
        #         dist = self.conditional_dist(poly1, poly2)
        #         if dist is not None and dist < self.polyline_threshold: # if distance between polylines is not None, add to list
        #             subset_pairs.append([idx1, idx2])
        #
        #
        # if predict:
        #     X['activity_id'] = 0 # set column named activity_id values to placeholder
        # final_clusters = self.make_indexer(subset_pairs) # make new indexer with clusters whose polylines match
        # # drop keys that are already in another clusters
        # copy_final_clusters = final_clusters.copy()
        # for idx1, other_idxs in copy_final_clusters.iteritems():
        #     for idx2 in other_idxs:
        #         final_clusters.pop(idx2, None)
        #
        # if predict:
        #     for idx1, other_idxs in final_clusters.iteritems():
        #         activity_id = uuid.uuid4().hex[:8] # 8 char activity key
        #         mask_arr = np.array([idx1] + other_idxs)
        #         X.loc[mask_arr, ['activity_id']] = activity_id
        # else:
        #     for idx1, other_idxs in final_clusters.iteritems():
        #         mask_arr = np.array([idx1] + other_idxs)
        #         if X.loc[mask_arr, 'activity_id'].isnull().all():
        #             activity_id = uuid.uuid4().hex[:8] # 8 char activity key
        #             X.loc[mask_arr, ['activity_id']] = activity_id
        #         else:
        #             activity_id = X.loc[mask_arr, 'activity_id'].dropna().values[0]
        #             X.loc[mask_arr, ['activity_id']] = activity_id
        #
        #
        # # set all remaining activities without a cluster assignment to a new a activity key
        #
        # if predict:
        #     cluster_center_idxs = np.concatenate((np.argwhere(X['activity_id'] == 0).ravel(), np.array(final_clusters.keys())))
        #     mask1 = X['activity_id'] == 0
        # else:
        #     cluster_center_idxs = np.concatenate((np.argwhere(X['activity_id'].isnull()).ravel(), np.array(final_clusters.keys())))
        #     mask1 = X['activity_id'].isnull()
        #
        # X.loc[mask1, ['activity_id']] = X.loc[mask1, ['activity_id']].apply(lambda x: uuid.uuid4().hex[:8], 1)
        #
        # cols = ['map_summary_polyline', 'start_lat', 'start_lng', 'end_lat', 'end_lng', 'activity_id']
        #
        # if predict:
        #     self.cluster_centers_ = X.loc[cluster_center_idxs, cols].reset_index()
        #     self.labels_ = X.activity_id.values
        # else:
        #     self.cluster_centers_ = self.cluster_centers_.append( X.loc[np.intersect1d(cluster_center_idxs, new_idxs, assume_unique=True), cols]).reset_index(drop=True)
        #     labels = X.dropna(subset = ['id']).activity_id.values
        #     return labels
        #

    def fit(self, X, y=None):
        """
        Inputs: X - dataframe containg lattiude and longitude information for original data, as well as encoded map summary polyline for each activity.
        """
        assert (X['map_summary_polyline'].isnull().sum() == 0), "all activities must have map_summary_polyline"
        assert (X['start_lat'].isnull().sum() == 0), "all activities must have start_lat"
        assert (X['start_lng'].isnull().sum() == 0), "all activities must have start_lng"
        assert (X['end_lat'].isnull().sum() == 0), "all activities must have end_lat"
        assert (X['end_lng'].isnull().sum() == 0), "all activities must have end_lng"


        n_samples = X.shape[0]
        self._build_clusters(X)

    def predict(self, X, y=None):
        """
        Needs information
        """
        assert (self.labels_ is not None), "must run fit method before predict"

        return self.labels_

    # def transform(self, X, y=None):
    #     assert (self.cluster_centers_ is not None), "must run fit method before transform"
    #     X = pd.concat([self.cluster_centers_, X], ignore_index=True)
    #
    #     return self._build_clusters(X, predict=False)

if __name__ == '__main__':

    df = pd.read_csv('data/activities_small.csv', encoding='utf-8')
    df = df[df['commute'] == 0] # drop all where commute = true
    df = df[df['state'] == 'Colorado'] # drop all where state != Colorado
    df = df[df['type'] == 'Ride'] # drop everything that is not of type 'Ride'
    df.dropna(subset=['map_summary_polyline'], inplace=True) # drop all where map_summary_polyline is nan
    # df.reset_index(drop=True, inplace=True)
    msk = np.random.rand(len(df)) < 0.75
    train = df[msk].reset_index(drop=True)
    test = df[~msk].reset_index(drop=True)


    clusterer = PolylineClusterMaker()
    clusterer.fit(df)
    train_labels = clusterer.predict(df)
    # test_labels = clusterer.transform(test)
