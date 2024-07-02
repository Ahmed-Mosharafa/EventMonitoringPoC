from sentence_transformers import SentenceTransformer
import numpy as np


class ContextualEmbeddings:

    def __init__(self):
        # Load pre-trained SentenceTransformer model
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def get_contextual_embeddings(self, tweets):
        # Calculate embeddings by calling model.encode()
        contextual_embeddings = self.model.encode(tweets)
        return contextual_embeddings

    def get_similarity_matrix(self, tweets):
        tweet_texts = [tweet['normalized'] for tweet in tweets]
        embeddings = self.get_contextual_embeddings(tweet_texts)
        # Calculate the embedding similarities
        similarity_matrix = self.model.similarity(embeddings, embeddings)
        print(similarity_matrix)
        return similarity_matrix

    def get_similarities(self, embedding1, embedding2):
        # Calculate the embedding similarities
        similarities = self.model.similarity(embedding1, embedding2)
        print("Similarities:", similarities)
        return similarities
