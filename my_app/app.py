from collections import Counter
from flask import Flask, request
import cPickle as pickle

#Initialize app
app = Flask(__name__)




# load the pickled model
with open('data/model.pkl') as f:
    model = pickle.load(f)

# load the vectorizer to transform X into a tfidf
with open('data/vectorizer.pkl') as f:
    vectorizer = pickle.load(f)

def dict_to_html(d):
    return '<br>'.join('{0}: {1}'.format(k, d[k]) for k in sorted(d))


# Home page with form on it to submit text
@app.route('/')
def get_text():
    return '''
        <form action="/predict" method='POST' >
            <input type="text" name="user_input" />
            <input type="submit" value = "Submit text" />
        </form>
        '''


# Predict on entered text (why does this have methods but the top one doesn't)
@app.route('/predict', methods=['POST'] )
def predict():
    # request the text from the form
    text = str(request.form['user_input'])

    # transform (vectorize) the text
    X = vectorizer.transform([text])

    # predict on the text
    y_pred = model.predict(X)

    out = 'The text: {0} <br><br>belongs to section: {1}'
    submit_for_new_query = '''
        <form action="/" >
            <input type="submit" value = "Enter new text" />
        </form>
        '''
    return out.format(text, y_pred) + submit_for_new_query


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
