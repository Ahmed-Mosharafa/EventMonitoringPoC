from sentence_transformers import SentenceTransformer

from utils import TweetPreprocessor

# The sentences to encode
test_tweets = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

model_name = 'all-mpnet-base-v2'


class ContextualEmbeddings:

    def __init__(self, model_name: str):
        # Load pre-trained SentenceTransformer model
        self.model = SentenceTransformer(model_name)

    def get_contextual_embeddings(self, tweets):
        # Preprocess tweets
        preprocessor = TweetPreprocessor()
        preprocessed_tweets = preprocessor.process_tweets(tweets)
        preprocessed_tweets = [' '.join(words) for words in preprocessed_tweets]
        print(preprocessed_tweets)

        # Calculate embeddings by calling model.encode()
        contextual_embeddings = self.model.encode(preprocessed_tweets)
        return contextual_embeddings

    def get_similarity_matrix(self, embeddings):
        # Calculate the embedding similarities
        similarity_matrix = self.model.similarity(embeddings, embeddings)
        return similarity_matrix

    def get_similarities(self, embedding1, embedding2):
        # Calculate the embedding similarities
        similarities = self.model.similarity(embedding1, embedding2)
        print("Similarities:", similarities)
        return similarities


# Instantiate the class with the model name
embedding_model = ContextualEmbeddings(model_name=model_name)

# Get embeddings
embeddings = embedding_model.get_contextual_embeddings(test_tweets)

'''# Print embeddings
for i, tweet in enumerate(tweets):
    print(f"Tweet: {tweet}")
    print(f"Embedding: {embeddings[i]}\n")'''

# Calculate and print similarity matrix
similarity_matrix = embedding_model.get_similarity_matrix(embeddings)
print("Similarity matrix:", similarity_matrix)