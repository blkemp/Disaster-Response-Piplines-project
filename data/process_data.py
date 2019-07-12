import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from sklearn.preprocessing import minmax_scale

def load_data(messages_filepath, categories_filepath):
    # read in messages data 
    df = pd.read_csv(messages_filepath)
    #OneHotEncode genre column
    df = df.merge(
        pd.get_dummies(df, columns=['genre'], drop_first=False).drop(['message','original'], axis=1),
        on='id')

    # load categories data
    categories = pd.read_csv(categories_filepath)

    # create a dataframe of the 36 individual category columns
    categories = pd.concat([categories, 
                            categories['categories'].str.split(';',expand=True)],
                            axis=1).drop('categories', axis=1)

    # use first row to extract a list of new column names for categories.
    row = categories.iloc[0,1:]
    category_colnames = [item[0] for item in np.array(row.str.split('-'))]
    categories.columns = ['id'] + category_colnames

    # set each value to be the last character of the string
    for column in categories:    
        if column != 'id':
            categories[column] = categories.loc[:,column].str[-1].astype('int')
    
    # merge category data with raw messages
    df = df.merge(categories, on='id')
    return df

def clean_data(df):
    # drop duplicates
    df.drop_duplicates(inplace=True)

    response_columns = set(df.columns) - set({'id','message','original','genre'})
    # Ensure response columns are binary
    for column in response_columns:
        a = np.array(df[column].values.tolist())
        df[column] = ((a > 0)*1).tolist()

    # drop columns with no variation     
    for column in response_columns:
        if len(df[column].value_counts()) < 2:
            df.drop(column, axis=1, inplace=True)
    
    # drop columns not used in analysis ## Deprecating for now just in case graphing is wanted later
    #drop_columns = set(df.columns) - set(['id','message'])
    #drop_columns -= response_columns
    #df.drop(drop_columns, axis=1, inplace=True)

    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('InsertTableName', engine, index=False) 

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