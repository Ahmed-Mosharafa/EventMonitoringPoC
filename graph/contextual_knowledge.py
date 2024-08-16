from sentence_transformers import SentenceTransformer
import numpy as np


class ContextualEmbeddings:

    def __init__(self):
        '''
        Class using pre-trained Senetence Transfomer models to get contextual similarity between tweets
        '''
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def get_contextual_embeddings(self, tweets):
        '''
        Initializes the ContextualEmbeddings class with a pre-trained model.

        () -> None
        '''
        # Calculate embeddings by calling model.encode()
        contextual_embeddings = self.model.encode(tweets)
        return contextual_embeddings

    def get_similarity_matrix(self, tweets):
        '''
        Computes a similarity matrix for all tweet embeddings.

        (tweets: list of dict) -> np.ndarray
        A numpy array representing the similarity matrix between tweet embeddings.
        '''
        tweet_texts = [tweet['normalized'] for tweet in tweets]
        embeddings = self.get_contextual_embeddings(tweet_texts)
        similarity_matrix = self.model.similarity(embeddings, embeddings)
        return similarity_matrix

    def get_similarities(self, embedding1, embedding2):
        '''
        Computes the similarity between two tweet embeddings.

        (embedding1: np.ndarray, embedding2: np.ndarray) -> float
        A float representing the similarity score between the two embeddings.
        '''
        # Calculate the embedding similarities
        similarities = self.model.similarity(embedding1, embedding2)
        return similarities
