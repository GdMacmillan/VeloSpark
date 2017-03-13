import pandas as pd
import numpy as np
import polyline

class Activity_Labeler(object):

    def __init__(self, activity_dataframe):
        self.activity_dataframe = activity_dataframe

    def decode_and_create_arrays(self):
        self.polylines = activity_dataframe.map_summary_polyline.values

        func = vectorize(polyline.decode)
        self.lats, self.lngs

act1_lat_arr, act1_lng_arr = zip(*polyline.decode(act1_ply))
