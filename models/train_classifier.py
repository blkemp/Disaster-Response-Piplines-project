import sys
# import libraries
import numpy as np
import pandas as pd
import nltk
import numpy as np
import pandas as pd
import sqlalchemy
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.externals import joblib 
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

class BasicTextAnalytics(BaseEstimator, TransformerMixin):
    '''
    Class for returning some basic numerical data for text analysis to include in 
    modelling. Such as: 
    - Number of sentences
    - Number of words
    - Number of nouns
    - Number of verbs
    - Number of adjectives
    A lot of the above were taken from ideas found here: 
    https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
    '''
    pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
    }

    # function to check and get the part of speech tag count of a words in a given sentence
    def check_pos_tag(self, text, flag):
        '''
        Returns the count of a given NL pos_tag, based on user selection. E.g. number of nouns.
        INPUTS
        text - the given text to analyse
        flag - pos family to analyse, one of 'noun', 'pron' , 'verb', 'adj' or 'adv'
        '''
        count = 0
        try:
            wiki = textblob.TextBlob(text)
            for tup in wiki.tags:
                ppo = list(tup)[1]
                if ppo in pos_family[flag]:
                    count += 1
        except:
            pass
        return count
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        trainDF = pd.DataFrame()
        trainDF['text'] = X
        trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
        trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
        trainDF['noun_count'] = trainDF['text'].apply(lambda x: self.check_pos_tag(x, 'noun'))
        trainDF['verb_count'] = trainDF['text'].apply(lambda x: self.check_pos_tag(x, 'verb'))
        trainDF['adj_count'] = trainDF['text'].apply(lambda x: self.check_pos_tag(x, 'adj'))
        trainDF['adv_count'] = trainDF['text'].apply(lambda x: self.check_pos_tag(x, 'adv'))
        trainDF['pron_count'] = trainDF['text'].apply(lambda x: self.check_pos_tag(x, 'pron'))
        
        return trainDF.drop('text',axis=1)

def load_data(database_filepath):
    # load from database
    engine = sqlalchemy.create_engine(database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    
    # split input and response variables
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    clean_tokens = [tok.lower().strip() for tok in tokens]

    return clean_tokens


def build_model():
    # Build and scale engineered features separate to Natural Language transformations
    # We create the preprocessing pipelines for both numeric and text data.

    pipeline_model = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, 
                                        ngram_range=(1, 2),
                                        max_features=5000,
                                        max_df=0.5)),
                ('tfidf', TfidfTransformer())
            ])),

            ('numerical_pipeline', Pipeline([
                ('analytics', BasicTextAnalytics()),
                ('norm', StandardScaler())
                ]))
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
    ])
    # specify parameters for grid search
    parameters = {
        'clf__estimator__min_samples_split' : [4,8,16,32],
        'clf__estimator__max_depth': [None,50,100]
    }

    # create grid search object
    # using f1 score rather than auc because of the significant imbalance in class distributions
    # per https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    cv = GridSearchCV(pipeline_model, parameters, scoring='f1_weighted', verbose=3)

    return cv

def evaluate_model(model, X_test, Y_test):
    # Note: this function used to have a variable "category_names" per the default
    # I'm not sure it was necessary so I have removed it.

    y_pred = model.predict(X_test)
    
    print('f1 score: {}'.format(f1_score(Y_test, y_pred, average='weighted')))
    pass


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)
        # Note: used to also include parameter category_names

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py /data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()