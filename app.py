from flask import Flask

from utils.Dataloader import Dataloader
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
windows = dataloader.window_making(60, data)

for name, window in windows:
    tweets = window.to_dict('records')
    tweets = preprocessor.process_tweets(tweets)
    # print(tweets[0])
    contextual_similarity = contextual.get_similarity_matrix(tweets)
    structural_similarity = structural.get_struct_relation_matrix(tweets)#TODO: structual similarity

    graph_matrix = 0.5 * (contextual_similarity + structural_similarity)

    # Markov Clustering
    mcl = MarkovClustering(graph_matrix, tweets)
    mcl.apply_mcl_algorithm()
    mcl.plot_clusters()

    # Get highest scores
    sorted_indices = np.argsort((graph_matrix * -1), axis=None)
    rows, columns = np.unravel_index(sorted_indices, graph_matrix.shape)

    sorted_indices_pairs = list(zip(rows, columns))

    for i in range(10):
        k, j = sorted_indices_pairs[i]
        print(tweets[k]['text'])
        print(tweets[j]['text'])
        print("\n")
    break

    #return "<p>Hello, World!</p>"
 