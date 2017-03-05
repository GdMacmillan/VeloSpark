
from collections import Counter
from flask import Flask, request, render_template, session
import cPickle as pickle
import numpy as np
import pandas as pd # also temporary
from clustering import load_data, get_labels
import os
# sys.path.append('../')
#Initialize app

app = Flask(__name__, static_url_path = "", static_folder = "static")


#### temporary because i can't query a database at the momemnt ####
# load data from clustering.py
co_runs_df, co_rides_df = load_data()

# append labels and return clustering models from clustering.py
co_runs_df, co_rides_df, runs_clusterer, rides_clusterer = get_labels(co_runs_df, co_rides_df)

# defines function to load runs model
def load_runs():
    with open('data/item_similarity_runs.npy') as f:
        item_similarity_runs = np.load(f)

    with open('data/runs_clusterer.pkl') as f:
        runs_clusterer = pickle.load(f)

    with open('data/runs_mapper.pkl', 'rb') as f:
        runs_mapper = pickle.load(f)

    return item_similarity_runs, runs_clusterer, runs_mapper

# defines function to load rides model
def load_rides():
    with open('data/item_similarity_rides.npy') as f:
        item_similarity_rides = np.load(f)

    with open('data/rides_clusterer.pkl') as f:
        rides_clusterer = pickle.load(f)

    with open('data/rides_mapper.pkl', 'rb') as f:
        rides_mapper = pickle.load(f)

    return item_similarity_rides, rides_clusterer, rides_mapper

def top_k_labels(similarity, mapper, label_idx, k=3):
    return [mapper[x] for x in np.argsort(similarity[label_idx,:])[:-k-1:-1]]

def get_activity_data(activites, df):
    c1, c2, c3, c4 = [],[],[],[]
    for activity_id in activites[0]:
        activity = df[df['id'] == activity_id].values
        c1.append(activity[0, 1])
        c2.append(activity[0, 2] * 0.000621371)
        c3.append(activity[0, 5] * 3.28084)
        c4.append(activity[0, 39])
    return zip(c1, c2, c3, c4)


# Home page with options to predict rides or runs
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# How it works page describing the recommender
@app.route('/how_it_works', methods=['GET'])
def how_it_works():
    return render_template('how_it_works.html')

# Contact information page to link various social media
@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

# This is the form page where users fill out whether they would like bike or run recommendations
@app.route('/form/<activity>', methods=['GET', 'POST'])
def get_activity_predictors(activity):
    city_list = list(pd.read_csv('data/colorado_cities.csv').city.values)
    if activity == 'bike':
        displayed_activity = 'cycling'
    else:
        displayed_activity = 'running'

    session['activity'] = activity
    return render_template('form.html', city_list=city_list, displayed_activity=displayed_activity)

# This displays user inputs froms the form page
@app.route('/results', methods=['GET', 'POST'] )
def predict_activities():
    distance = int(request.form['distance'])
    elevation_gain = int(request.form['elevation_gain'])
    moving_time = int(request.form['moving_time'])
    result = distance*elevation_gain*moving_time
    city = request.form['city']
    activity = session['activity']
    # data = {'distance': distance, 'elevation_gain': elevation_gain, 'moving_time': moving_time, 'city': city}
    # item_similarity_rides, rides_clusterer, rides_mapper = load_rides()
    # out = 'The index is {}'
    # rides = top_k_labels(item_similarity_rides, rides_mapper, user_input)
    return render_template('results.html', result=result, city=city, activity=activity)

@app.route('/map', methods=['POST'] )
def go_to_map():
    return render_template('map.html')

app.secret_key = os.urandom(24)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
