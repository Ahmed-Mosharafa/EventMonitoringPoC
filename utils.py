import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def window_making(tweets):
    pass

def remove_noise(tweet):

    #make tweet lower case
    lower = tweet.lower()

    # tokenize
    word_tokens = word_tokenize(lower)
    
    #remove tweets with less than 3 words
    if len(word_tokens) < 3:
        return ''

    #remove stop words
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    word_tokens = [w for w in word_tokens if not w in stop_words]

    #remove punctuation
    pass

    #remove web-lins
    pass

    #remove non-ASCII characters
    pass

    #remove retweets
    pass

    return ' '.join(word_tokens)



