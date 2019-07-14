import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from engineered_features import BasicTextAnalytics

# for some reason this only works in the virtual environment and not on my local machine.
# inserted this class at the end of the text to save the hassle

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Get categories excluding OHE and related vectors
    category_names = set(df.columns) - set({'id',
                                            'message',
                                            'original',
                                            'related',
                                            'genre',
                                            'genre_direct',
                                            'genre_news',
                                            'genre_social'})
    category_names = list(category_names)
    categories = df[category_names]
    category_means = categories.mean().sort_values(ascending=False)[1:11]
    low_cat_means = categories.mean().sort_values(ascending=True)[1:11]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
        # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=low_cat_means.index,
                    y=low_cat_means
                )
            ],

            'layout': {
                'title': 'Least Common Message Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_means.index,
                    y=category_means
                )
            ],

            'layout': {
                'title': 'Most Common Message Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    category_excludes = set({'id','message','original','related','genre','genre_direct','genre_news','genre_social'})
    category_names = []
    for column in df.columns:
        if column not in category_excludes:
            category_names.append(column)
    categories = df[category_names]
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(categories, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
