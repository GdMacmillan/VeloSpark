# *FindingEpics*
## (codename VeloSpark)

### A Colorado Ride/Run Recommender

Gordon MacMillan

Galvanize Data Science Immersive - Capstone Project - March 2017

## Introduction

Strava is a social network for athletes. It allows millions of users to upload activities and maintains a profile for athletes and any associated achievements, stats, photos, as well as connections to other athletes.

I'm a big fan of exercise like cycling and running. I wanted to do something with my newfound knowledge of API's and the wealth of data you can get with the nice one that Strava provides.

The VeloSpark app is what I came up with. Users can get recommendations that they might enjoy based on inputs to the web application. The app uses a clustering based approach to label an activity according to it's absolute location, total distance and elevation gain. The app then uses collaborative filtering to get a similarity of the input to other activities. Future use will allow users to connect with their strava account.

### Table of Contents
* [Project Scope](#h1)
* [Feature Engineering](#h2)
* [Exploratory Data Analysis](#h3)
* [Development](#h4)
* [Web Application](#h5)
* [Tools and Dependencies](#h6)
* [Acknowledgements](#h7)

## <a id="h1"></a> Project Scope

#### This project sets out to accomplish two things:
*   Provide users recommendations for rides or runs to do in the state of Colorado.
*   Create a nice stable interface for viewing those recommendations.

Future work might expand the areas for which recommendations are provided. It also may include a way for users to authenticate their Strava account and have recommendations provided based on their activity history.

## <a id="h2"></a> Data Engineering

Data was sourced from the [Strava V3 API](http://strava.github.io/api/) using a nice set of tools provided by [stravalib](http://pythonhosted.org/stravalib/). To start off I needed to create a dataframe with the source data for activities from the API. I had to write scraper code using Selenium and Beautiful Soup for activity ID's since the number of activities I could get from the API alone was not going to be enough. The activities I collected included everything from running to paddleboarding and yoga. For the purposes of this recommender I only used running and cycling activities. I performed most of the scraping and API calls using a class called Strava_scraper, found in `scraper.py`. I then cleaned and pre-processed data using functions found in `setup.py`.

Features of the activity dataframe:

    ['id', 'resource_state', 'external_id', 'upload_id', 'athlete', 'name', 'distance', 'moving_time', 'elapsed_time', 'total_elevation_gain', 'type', 'start_date', 'start_date_local', 'timezone', 'start_latlng', 'end_latlng', 'achievement_count', 'kudos_count', 'comment_count', 'athlete_count', 'photo_count', 'total_photo_count', 'map', 'trainer', 'commute', 'manual', 'private', 'flagged', 'average_speed', 'max_speed', 'average_watts', 'max_watts', 'weighted_average_watts', 'kilojoules', 'device_watts', 'has_heartrate', 'average_heartrate', 'max_heartrate']

The athlete was in the form of a stravalib object (basically json) so I converted that to just the 'athlete_id', not needing any other information about the athlete. Map is also an object containing route information such as the geospacial starting and ending location as well as a google polyline encoded string. I converted this to start_lat, start_lng, end_lat, end_lng and map_summary_polyline.

I dropped the following features:

    ['athlete', 'upload_id', 'resource_state', 'external_id', 'start_latlng', 'end_latlng', 'map']

Time delta columns `moving_time` and `elapsed_time` were converted to total seconds.

Other features were explicitly set to the desired to the desired datatype.

I also added two features
    * closest_city
    * state

With state, I could drop all activities not in Colorado. The closest_city feature is displayed with the results.


## <a id="h3"></a> Exploratory Data Analysis

I used a Jupyter notebook for most of the data exploration. This is shown in the notebook `eda_modeling.ipynb`

### Some Insights
The small dataset used for this write up had the following shape with number of activities and number of features:

    cycling: (5738, 40)
    running: (1215, 40)

Distance and elevation gain are distributed along an exponential curve shown here plotted in log space. I plotted in log space so it is a bit easier to see the distributions.

![image](web_app/static/images/Distribution.png)

Activities are spread out around Colorado. The majority are in the Denver/Boulder area.  

![image](web_app/static/images/Activity_distributions.png)

## <a id="h4"></a> Development

I organized the project into two main parts; Building what I call the recommender model which you will find in `build_recommender_model.py` and the web app called `recommender_app.py`.

### Building the recommender
To build the model, I took the following steps:
* Load pre-processed data from csv's into dataframes. Include only Colorado activities.
* Use k-means algorithm to cluster activities according to geographic starting latitude, longitude, total elevation gain, distance and moving time. This provides a label, which I used as analog to define the activity type. I tried to keep the number of activities per cluster to around 10.
* Calculate ratings to each user for each activity based on the number of times they have done an activity of a certain type.
* Create item similarity matrix for activities using numpy dot function.
* Generate index to activity dictionary for looking up activities based on labels returned by recommender.
* Pickle clustering model and activity data frame, similarity matrix and numpy save similarity matrix.

### Building the web application
After the models are built and stored in pickle files. I have the web application call those files rather than perform clustering and computing similarities in real time:
* The recommender starts with a landing page on which a user chooses whether they would like to go for a bike ride or a run. From here, I implement the recommender.
* The user goes to a form page where the activity they have clicked is stored as a session variable. They then fill out the form which is implemented as my solution to being a cold start. Each new user to the page gets just one activity type(label) on which their recommendations are built.
* The results page takes the label and finds k most similar labels from the similarity matrix. In this case k=5. The app maps these labels to the activities associated with them and combines them into a single array. The results are then displayed in order of increasing distance from the user specified location.
* The user may choose to click on a map which displays the activity map on a seperate page. It displays this using a query to the google maps API, as well as a bit of javascript which plots the latitude longitude points decoded from the google polyline string.

## <a id="h5"></a> Web Application: [VeloSpark](http://ec2-54-234-99-142.compute-1.amazonaws.com:8080)

To use the web application, click on the VeloSpark link and you will be directed to the homepage currently being hosted on Amazon Web Services EC2. This may change to a dedicated host if the recommender is improved and the ability to scale is seen as necessary.

![image](web_app/static/images/Homepage.png)

## <a id="h6"></a> Tools and Dependencies
The following are the packages I used and can be pip installed:

    flask
    pandas
    numpy
    sklearn
    scipy
    bs4
    selenium
    reverse_geocoder
    stravalib
    urllib2

## <a id="h7"></a> Acknowledgements
The data was sourced from Strava so all kudos for the engineering and development team behind their API. Also, the instructors at Galvanize for their vast body of knowledge and helpful tips.
