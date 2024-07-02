from flask import Flask

from utils.dataloader import Dataloader
from utils.tweet_preprocessor import TweetPreprocessor
from graph.contextual_knowledge import ContextualEmbeddings
from graph.structural_relation import StructuralRelation
from clustering.mcl_clustering import MarkovClustering

import numpy as np

#app = Flask(__name__)

dataloader = Dataloader()
preprocessor = TweetPreprocessor()

contextual = ContextualEmbeddings()
structural = StructuralRelation()
#@app.route("/events")
#def fetch_events():

data = dataloader.events2012()
windows = dataloader.window_making(2880, data)

for name, window in windows:
    tweets = window.to_dict('records')
    tweets = preprocessor.process_tweets(tweets)
    contextual_similarity = contextual.get_similarity_matrix(tweets)
    structural_similarity = structural.get_struct_relation_matrix(tweets)

    graph_matrix = 0.5 * (contextual_similarity + structural_similarity)

    print(graph_matrix)

    # Markov Clustering
    mcl = MarkovClustering(graph_matrix, tweets)
    mcl.apply_mcl_algorithm()
    mcl.plot_clusters()

    break

    #return "<p>Hello, World!</p>"
 