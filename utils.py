import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import string

class TweetPreprocessor:

    def process_tweets(self, tweets):
        normalized_tweets = []
        for tweet in tweets:
            tweet_tokens = self.normalize_tweet(tweet)
            if len(tweet_tokens) > 0:
                normalized_tweets.append(tweet_tokens)
        print(normalized_tweets)
        return normalized_tweets
        #return window_making(normalized_tweets)

    def window_making(self, tweets):
        pass

    def normalize_tweet(self, tweet):

        tokenizer = TweetTokenizer()
        #make tweet lower case
        lower = tweet.lower()

        # tokenize
        word_tokens = tokenizer.tokenize(lower)

        #remove tweets with less than 3 words
        if len(word_tokens) < 3:
            return []

        #remove stop words
        #nltk.download('stopwords')
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
