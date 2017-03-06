from collections import Counter
from flask import Flask, request, render_template
import cPickle as pickle
import numpy as np
import pandas as pd # also temporary
from clustering import load_data, get_labels
#Initialize app
app = Flask(__name__)


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

return_to_homepage = '''
    <form action="/" >
        <input type="submit" value = "Return to homepage" />
    </form>
    '''

def get_activity_data(activites, df):
    c1, c2, c3, c4 = [],[],[],[]
    for activity_id in activites[0]:
        activity = df[df['id'] == activity_id].values
        c1.append(activity[0, 1])
        c2.append(activity[0, 2] * 0.000621371)
        c3.append(activity[0, 5] * 3.28084)
        c4.append(activity[0, 39])
    return zip(c1, c2, c3, c4)


# Home page with form on it to submit text
@app.route('/', methods=['GET','POST'])
def get_preference():
    # return '''
    #     <head>
    #         <meta charset="utf-8">
    #         <title>Index</title>
    #     </head>
    #     <body>
    #         <form action="/rides" method='POST' >
    #             <input type="submit" value = "Find Rides" />
    #         </form>
    #         <form action="/runs" method='POST' >
    #             <input type="submit" value = "Find Runs" />
    #         </form>
    #         <form action="/map" method='POST' >
    #             <input type="submit" value = "go to map" />
    #         </form>
    #     </body>
    #       '''
    return render_template('index.html')


@app.route('/rides', methods=['GET', 'POST'])
def get_ride_predictors():
    out = 'this is where you enter your ride preferences'
    predict = '''
        <form action="/predict_rides" method='POST' >
            <input type="number" name="user_input_rides" />
            <input type="submit" value = "Submit ride predictors" />
        </form>
        '''
    # return predict + return_to_homepage
    return render_template('rides.html')

@app.route('/runs', methods=['GET', 'POST'])
def get_run_predictors():
    out = 'this is where you enter your ride preferences'
    predict = '''
        <form action="/predict_runs" method='POST' >
            <input type="number" name="user_input_runs" />
            <input type="submit" value = "Submit run predictors" />
        </form>
        '''
    # return predict + return_to_homepage
    return render_template('runs.html')

@app.route('/predict_rides', methods=['GET', 'POST'] )
def predict_rides():
    user_input = int(request.form['user_input_rides'])

    item_similarity_rides, rides_clusterer, rides_mapper = load_rides()
    out = 'The index is {}'
    rides = top_k_labels(item_similarity_rides, rides_mapper, user_input)
    return render_template('predict_rides.html', data=get_activity_data(rides, co_rides_df)) + return_to_homepage

@app.route('/predict_runs', methods=['GET', 'POST'] )
def predict_runs():
    user_input = int(request.form['user_input_runs'])

    item_similarity_runs, runs_clusterer, runs_mapper = load_runs()
    out = 'The index is {}'
    runs = top_k_labels(item_similarity_runs, runs_mapper, user_input)
    return render_template('predict_runs.html', data=get_activity_data(runs, co_runs_df)) + return_to_homepage

@app.route('/map', methods=['POST'] )
def go_to_map():
    return render_template('map.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
