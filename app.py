from flask import Flask
from utils.Dataloader import Dataloader
from utils.TweetPreprocessor import TweetPreprocessor
#app = Flask(__name__)

dataloader = Dataloader()
preprocessor = TweetPreprocessor()

#@app.route("/events")
#def fetch_events():

data = dataloader.events2012()
windows = dataloader.window_making(60, data)

for name, window in windows:
    tweets = window.to_dict('records')
    tweets = preprocessor.process_tweets(tweets)
    print(tweets)

    #return "<p>Hello, World!</p>"
