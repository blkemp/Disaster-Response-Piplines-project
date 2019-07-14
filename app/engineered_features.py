from sklearn.base import BaseEstimator, TransformerMixin

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