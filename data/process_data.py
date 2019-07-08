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
    messages = messages.merge(
        pd.get_dummies(messages, columns=['genre'], drop_first=False).drop(['message','original'], axis=1),
        on='id')

    # load categories data
    categories = pd.read_csv(categories_filepath)
    # use first row to extract a list of new column names for categories.
    row = categories.iloc[0,1:]
    category_colnames = [item[0] for item in np.array(row.str.split('-'))]
    categories.columns = ['id'] + category_colnames

    # set each value to be the last character of the string
    for column in categories:    
        if column != 'id':
            categories[column] = categories.loc[:,column].str[-1].astype('int')
    
    # merge category data with raw messages
    df.merge(categories, on='id')
    return df

def clean_data(df):
    # drop duplicates
    df.drop_duplicates(inplace=True)

    response_columns = set(df.columns) - set({'id','message','original','genre'})
    # Ensure response columns are binary
    for column in response_columns:
        a = np.array(df[response_columns].values.tolist())
        df[response_columns] = ((a > 0)*1).tolist()

    # drop columns with no variation     
    for column in response_columns:
        if len(df[column].value_counts()) < 2:
            df.drop(column, axis=1, inplace=True)
    
    # drop columns not used in analysis ## Deprecating for now just in case graphing is wanted later
    #drop_columns = set(df.columns) - set(['id','message'])
    #drop_columns -= response_columns
    #df.drop(drop_columns, axis=1, inplace=True)

    # Feature engineering
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    df['title_word_count'] = df['message'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    df['noun_count'] = df['message'].apply(lambda x: check_pos_tag(x, 'noun'))
    df['verb_count'] = df['message'].apply(lambda x: check_pos_tag(x, 'verb'))
    df['adj_count'] = df['message'].apply(lambda x: check_pos_tag(x, 'adj'))
    df['adv_count'] = df['message'].apply(lambda x: check_pos_tag(x, 'adv'))
    df['pron_count'] = df['message'].apply(lambda x: check_pos_tag(x, 'pron'))
    # Scale engineered features
    sc_features = ['word_count','title_word_count','noun_count','verb_count','adj_count',
                    'adv_count','pron_count']
    for feature in sc_features:
        df[feature] = minmax_scale(df[feature])

    return df

def check_pos_tag(x, flag):
    '''Function for returning number of pos_tag items in given text
    Inputs: 
    x = text
    flag = word type indicator for pos_tag reference, e.g. 'noun','verb', etc. 
    '''

    pos_family = {
            'noun' : ['NN','NNS','NNP','NNPS'],
            'pron' : ['PRP','PRP$','WP','WP$'],
            'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
            'adj' :  ['JJ','JJR','JJS'],
            'adv' : ['RB','RBR','RBS','WRB']
            }

    count = 0
    try:
        text_types = nltk.pos_tag(tokenize(df.message[1]))
        for word, word_pos_fam in text_types:
            if word_pos_fam in pos_family[flag]:
                count += 1
    except:
        pass
    return count

def save_data(df, database_filename):
    engine = create_engine(database_filename)
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