import pandas as pd
import numpy as np
import polyline

class Clusterer(Object):
    """
    A clustering tool to group activities by start and end latitude and longitude and then further by their polyline.
    """
    def __init__(self, start_threshold=):
        """
        Input: None. **Possible attributes could be id schema and threshold values on which to define closeness of groups of acitivities
        """
        self.clusterer

    def fit(self, X):
        """
        Inputs: X - dataframe containg lattiude and longitude information, as well as encoded map summary polyline for each activity.
        """
        self.build_clusterer(X)
