import sys
# import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sqlalchemy import create_engine
import nltk

def load_data(messages_filepath, categories_filepath):
    df = pd.read_csv(messages_filepath)
    df.merge(pd.read_csv(categories_filepath), on='id')

    #load categories data
    categories = pd.read_csv(categories_filepath)
    # use first row to extract a list of new column names for categories.
    row = categories.iloc[0,1:]
    category_colnames = [item[0] for item in np.array(row.str.split('-'))]
    categories.columns = ['id'] + category_colnames

    # set each value to be the last character of the string
    for column in categories:    
        if column != 'id':
            categories[column] = categories.loc[:,column].str[-1].astype('int')


def clean_data(df):
    pass


def save_data(df, database_filename):
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()