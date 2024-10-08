import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import string
import ssl


class TweetPreprocessor:
    """
    A class used to preprocess tweets by normalizing text, removing stop words,
    punctuation, URLs, and other undesired elements.
    """


    def process_tweets(self, tweets):
        """
        Processes a list of tweets by normalizing each tweet's text and filtering out
        tweets with less than three tokens.

        (tweets : list of dict)  ->  list of dict   A list of normalized tweets with additional 'tokens' and 'normalized'fields.
        """
        normalized_tweets = []
        for tweet in tweets:
            tweet['tokens'] = self.normalize_tweet(tweet['text'])
            if len(tweet['tokens']) > 0:
                tweet['normalized'] = ' '.join(tweet['tokens'])
                normalized_tweets.append(tweet)
        return normalized_tweets
        #return window_making(normalized_tweets)

    def normalize_tweet(self, tweet):
        """
        Normalizes a single tweet by converting to lowercase, tokenizing, removing
        stop words, URLs, punctuation, non-ASCII characters, and empty tokens.

        (tweet : str) -> list A list of normalized tokens from the tweet.
        """

        tokenizer = TweetTokenizer()
        #make tweet lower case
        lower = tweet.lower()

        # tokenize
        word_tokens = tokenizer.tokenize(lower)

        #remove tweets with less than 3 words
        if len(word_tokens) < 3:
            return []

        try:
            stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
        
        word_tokens = [w for w in word_tokens if not w in stop_words]

        #remove web-lins
        word_tokens = [w for w in word_tokens if not w.startswith('http') and not w.startswith('www')]

        #remove punctuation
        word_tokens = [''.join(c for c in w if c not in string.punctuation or c == '#') for w in word_tokens]

        #remove non-ASCII characters
        word_tokens = [w.encode('ascii', 'ignore').decode() for w in word_tokens]

        #remove retweet
        if False:
            return []

        #remove empty tokens
        word_tokens = [w for w in word_tokens if w]

        return word_tokens
