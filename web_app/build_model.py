from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import pickle

if __name__ == '__main__':
    # STEP 1
    # ==============================================
    # Load in the articles from their compressed pickled state
    data = pd.read_pickle('data/data.pkl')

    # data is a pandas df
    # Make X = our features
    X = data['content']

    # Make y = our labels
    y = data['section_name']

    # Convert the thing to a python list so sklearn doesn't freak out
    y_lst = list(y)


    # STEP 2
    # ==============================================
    # Process our data

    # Create our vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform our feature data
    vectorized_X = vectorizer.fit_transform(X)


    # STEP 3
    # ==============================================
    # Fit our model

    # Create a model
    clf = MultinomialNB()

    # Fit our model with our vecotrized_X and labels
    clf.fit(vectorized_X, y_lst)


    # STEP 4
    # ==============================================
    # Export model and vectorizer to use it later

    # Export our fitted model via pickle
    with open('data/model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # Export our vectorizer as well
    with open('data/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
