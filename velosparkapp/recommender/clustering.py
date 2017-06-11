from sklearn.base import ClusterMixin
from itertools import izip
import pandas as pd
import numpy as np
import polyline
import uuid

import time
from contextlib import contextmanager

@contextmanager
def measureTime(title):
    t1 = time.time()
    yield
    t2 = time.time()
    print '%s: %0.2f seconds elapsed' % (title, t2-t1)

class PolyClusterer(ClusterMixin):
    """
    A clustering tool to group activities by start and end latitude and longitude and then further by their polyline.
    """
    def __init__(self, start_threshold=0.01, end_threshold=0.01, polyline_threshold=1.0):
        """
        Input: None. **Possible attributes could be id schema and threshold values on which to define closeness of groups of acitivities
        """
        self.start_threshold=start_threshold
        self.end_threshold=end_threshold
        self.polyline_threshold=polyline_threshold
        self.labels_ = None
        self.cluster_centers_ = None

    def make_indexer(self, arr):
        indexer = {}
        for pair in arr:
            if pair[0] in indexer:
                indexer[pair[0]].append(pair[1])
            else:
                indexer[pair[0]] = [pair[1]]
        return indexer

    def polytrim(self, poly1, poly2):
        diff = len(poly1) - len(poly2)
        rands = np.random.choice(np.arange(0, len(poly1)-2, 2), diff/2, replace=False)
        return np.delete(poly1, np.hstack((rands,rands+1)))

    def conditional_dist(self, poly1, poly2):
        if np.linalg.norm(np.subtract(poly1[-2:], poly2[-2:])) <= self.end_threshold: # if end points are close by a certain threshold
            if len(poly1) > len(poly2):
                poly1 = self.polytrim(poly1, poly2)
            elif len(poly1) < len(poly2):
                poly2 = self.polytrim(poly2, poly1)
            return np.linalg.norm(np.subtract(poly1, poly2)) # return distance between polylines

    def _build_clusters(self, X, predict=True):
        if not predict:
            new_idxs = X[X.activity_id.isnull()].index

        lats = X.start_lat.values # array of start latitudes
        lngs = X.start_lng.values # array of end latitudes


        #block of code to time here
        dsts = [] # empty distance array
        # populate distance array
        for (lt, lg) in izip(lats, lngs):
            dsts.append(np.sqrt((lats - lt)**2 + (lngs - lg)**2))
        dsts = np.array(dsts) # convert to numpy array
        il1 = np.tril_indices(dsts.shape[0]) # lower triangle mask
        dsts[il1] = -1

        pairs = np.argwhere((dsts <= self.start_threshold) & (dsts > -1)) # all pairs of indices with distances between start locations shorter than threshold of 0.01
        start_clusters = self.make_indexer(pairs)

        subset_pairs = []
        for idx1, other_idxs in start_clusters.iteritems():
            for idx2 in other_idxs:
                poly1 = np.array(polyline.decode(X.ix[idx1, 'map_summary_polyline'])).flatten()
                poly2 = np.array(polyline.decode(X.ix[idx2, 'map_summary_polyline'])).flatten()
                dist = self.conditional_dist(poly1, poly2)
                if dist is not None and dist < self.polyline_threshold: # if distance between polylines is not None, add to list
                    subset_pairs.append([idx1, idx2])


        if predict:
            X['activity_id'] = 0 # set column named activity_id values to placeholder
        final_clusters = self.make_indexer(subset_pairs) # make new indexer with clusters whose polylines match
        # drop keys that are already in another clusters
        copy_final_clusters = final_clusters.copy()
        for idx1, other_idxs in copy_final_clusters.iteritems():
            for idx2 in other_idxs:
                final_clusters.pop(idx2, None)

        if predict:
            for idx1, other_idxs in final_clusters.iteritems():
                activity_id = uuid.uuid4().hex[:8] # 8 char activity key
                mask_arr = np.array([idx1] + other_idxs)
                X.loc[mask_arr, ['activity_id']] = activity_id
        else:
            for idx1, other_idxs in final_clusters.iteritems():
                mask_arr = np.array([idx1] + other_idxs)
                if X.loc[mask_arr, 'activity_id'].isnull().all():
                    activity_id = uuid.uuid4().hex[:8] # 8 char activity key
                    X.loc[mask_arr, ['activity_id']] = activity_id
                else:
                    activity_id = X.loc[mask_arr, 'activity_id'].dropna().values[0]
                    X.loc[mask_arr, ['activity_id']] = activity_id


        # set all remaining activities without a cluster assignment to a new a activity key

        if predict:
            cluster_center_idxs = np.concatenate((np.argwhere(X['activity_id'] == 0).ravel(), np.array(final_clusters.keys())))
            mask1 = X['activity_id'] == 0
        else:
            cluster_center_idxs = np.concatenate((np.argwhere(X['activity_id'].isnull()).ravel(), np.array(final_clusters.keys())))
            mask1 = X['activity_id'].isnull()

        X.loc[mask1, ['activity_id']] = X.loc[mask1, ['activity_id']].apply(lambda x: uuid.uuid4().hex[:8], 1)

        cols = ['map_summary_polyline', 'start_lat', 'start_lng', 'end_lat', 'end_lng', 'activity_id']

        if predict:
            self.cluster_centers_ = X.loc[cluster_center_idxs, cols].reset_index()
            self.labels_ = X.activity_id.values
        else:
            self.cluster_centers_ = self.cluster_centers_.append( X.loc[np.intersect1d(cluster_center_idxs, new_idxs, assume_unique=True), cols]).reset_index(drop=True)
            labels = X.dropna(subset = ['id']).activity_id.values
            return labels


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

    def transform(self, X, y=None):
        assert (self.cluster_centers_ is not None), "must run fit method before transform"
        X = pd.concat([self.cluster_centers_, X], ignore_index=True)

        return self._build_clusters(X, predict=False)

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


    clusterer = PolyClusterer()
    clusterer.fit(train)
    train_labels = clusterer.predict(train)
    test_labels = clusterer.transform(test)
