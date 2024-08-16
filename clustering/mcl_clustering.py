import networkx as nx
import markov_clustering as mc
import matplotlib.pyplot as plt

from collections import Counter
import numpy as np

class MarkovClustering:
    '''
    Class for performing Markov Clustering on a similarity graph derived from tweets.
    It includes methods to apply the clustering algorithm, plot clusters, and evaluate the results.
    '''
    def __init__(self, graph_matrix, tweets, threshold=0.29):
        '''
        Initializes the MarkovClustering class with the graph matrix, tweets, and a threshold.

        (graph_matrix: np.ndarray, tweets: list of dict, threshold: float) -> None
        '''
        self.graph_matrix = graph_matrix
        self.tweets = tweets
        self.threshold = threshold
        self.similarity_graph = None
        self.clusters = None

    def apply_mcl_algorithm(self):
        '''
        Applies the Markov Clustering Algorithm on the similarity graph.

        () -> list of list of int
        A list of clusters, where each cluster is a list of node indices representing the tweets.
        '''
        # Create a similarity graph
        similarity_graph = nx.Graph()
        matrix_length = len(self.tweets)

        for i in range(matrix_length):
            similarity_graph.add_node(i, tweet=self.tweets[i])
            for j in range(matrix_length):
                if i > j:  # Ensure we are in the lower triangular part
                    # print(f"Element at ({i}, {j}): {self.graph_matrix[i, j]}")
                    if self.graph_matrix[i, j] >= self.threshold:
                        similarity_graph.add_edge(i, j, weight=self.graph_matrix[i, j])

        self.similarity_graph = similarity_graph

        # Build adjacency matrix
        adjacency_matrix = nx.to_numpy_array(similarity_graph)

        # Apply the Markov Clustering Algorithm
        result = mc.run_mcl(adjacency_matrix)  # run MCL with default parameters
        self.clusters = mc.get_clusters(result)  # get the clusters
        self.clusters.sort(key=len, reverse=True)
        return self.clusters

    def plot_clusters(self):
        '''
        Plots the clusters by printing each cluster's tweets to the console.

        () -> None
        '''
        # Output the clusters
        for i, cluster in enumerate(self.clusters):
            print(f"Cluster {i + 1}:")
            for node in cluster:
                print(f"  - {self.similarity_graph.nodes[node]['tweet']}")

    def evaluation(self):
        '''
        Evaluates the clustering by calculating precision, recall, and F-score.

        () -> None
        Prints the precision, recall, and F-score of the clustering.
        '''
        evaluation_matrix = np.zeros((len(self.tweets), len(self.tweets)))
        all_tweets = []
        for i, cluster in enumerate(self.clusters):
            for node in cluster:
                all_tweets.append((self.similarity_graph.nodes[node]['tweet']['label'], i))
        
        for i, (t1l, t1p) in enumerate(all_tweets):
            for j in range(i+1,len(all_tweets)):
                # TP: 1, FN: 2, FP: 3, TN: 4
                (t2l, t2p) = all_tweets[j]
                score = 0
                if t1l == t2l and t1p == t2p:
                    score = 1
                elif t1l == t2l and t1p != t2p:
                    score = 2
                elif t1l != t2l and t1p == t2p:
                    score = 3
                elif t1l != t2l and t1p != t2p:
                    score = 4
                evaluation_matrix[i][j] = score

        unique, counts = np.unique(evaluation_matrix, return_counts=True)
        evals = dict(zip(unique, counts))

        precision = evals[1] / (evals[1] + evals[3])
        recall = evals[1] / (evals[1] + evals[2])
        fscore = 2 * (precision * recall) / (precision + recall)
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'fscore: {fscore}')

# Test usage
'''contextual = ContextualEmbeddings()
similarity_matrix = contextual.get_similarity_matrix(example_tweets)

mcl = MarkovClustering(similarity_matrix, example_tweets)
mcl.apply_mcl_algorithm()
mcl.plot_clusters()'''