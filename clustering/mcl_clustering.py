import networkx as nx
import markov_clustering as mc
import matplotlib.pyplot as plt


class MarkovClustering:

    def __init__(self, graph_matrix, tweets, threshold=0.5):
        self.graph_matrix = graph_matrix
        self.tweets = tweets
        self.threshold = threshold
        self.similarity_graph = None
        self.clusters = None

    def apply_mcl_algorithm(self):

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
        return self.clusters

    def plot_clusters(self):

        # Output the clusters
        for i, cluster in enumerate(self.clusters):
            print(f"Cluster {i + 1}:")
            for node in cluster:
                print(f"  - {self.similarity_graph.nodes[node]['tweet']}")

        # Visualize the graph (optional)
        pos = nx.spring_layout(self.similarity_graph)
        nx.draw(self.similarity_graph, pos, with_labels=True,
                labels=nx.get_node_attributes(self.similarity_graph, 'tweet'),
                node_size=500, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')
        plt.show()


# Test usage
'''contextual = ContextualEmbeddings()
similarity_matrix = contextual.get_similarity_matrix(example_tweets)

mcl = MarkovClustering(similarity_matrix, example_tweets)
mcl.apply_mcl_algorithm()
mcl.plot_clusters()'''