import cPickle as pickle
import pandas as pd

with open('data/vectorizer.pkl') as f:
    vectorizer = pickle.load(f)
with open('data/model.pkl') as f:
    model = pickle.load(f)

df = pd.read_csv('data/articles.csv')
X = vectorizer.transform(df['body'])
y = df['section_name']

print "Accuracy:", model.score(X, y)
print "Predictions:", model.predict(X)
