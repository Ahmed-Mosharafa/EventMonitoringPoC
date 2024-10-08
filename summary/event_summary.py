import spacy
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
from utils.tweet_preprocessor import TweetPreprocessor
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt


class EventSummarizer:
    '''
    Class for summarizing events based on tweet clusters. It includes methods
    for extracting top words from clusters, generating cluster summaries, and
    identifying the main topics of tweet clusters.
    '''

    def __init__(self, clusters, tweets):
        '''
        Initializes the EventSummarizer class with a TweetPreprocessor instance.

        (clusters: list, tweets: list of dict) -> None
        '''
        self.tweet_preprocessor = TweetPreprocessor()

    def get_top_words(self, clusters, tweets, top_n=7, ngram_range=(1, 3)):
        '''
        Extracts the top words from each tweet cluster using a term-document matrix.

        (clusters: list, tweets: list of dict, top_n: int, ngram_range: tuple) -> list of dict
        A list of dictionaries, each containing the top words for a cluster,
        the cluster ID, and the number of tweets in the cluster.
        '''
        # Summarize topics for each cluster
        clusters_top_words = []

        for i, cluster in enumerate(clusters):
            # Extract tweets in the cluster
            cluster_tweets = [tweets[node]['normalized'] for node in cluster]

            # Create a term-document matrix
            vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range)
            X = vectorizer.fit_transform(cluster_tweets)

            # Sum up the counts of each vocabulary word
            word_counts = np.asarray(X.sum(axis=0)).flatten()
            words = vectorizer.get_feature_names_out()

            # Get the top_n words
            top_words = [words[idx] for idx in word_counts.argsort()[-top_n:][::-1]]

            # Save the cluster summary
            cluster_top_words = {
                'cluster_id': i + 1,
                'top_words': top_words,
                'length': len(cluster)
            }
            clusters_top_words.append(cluster_top_words)

        return clusters_top_words

    def generate_summary(self, clusters, tweets):
        '''
        Extracts the top words from each tweet cluster using a term-document matrix.

        (clusters: list, tweets: list of dict, top_n: int, ngram_range: tuple) -> list of dict
        A list of dictionaries, each containing the top words for a cluster,
        the cluster ID, and the number of tweets in the cluster.
        '''
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        clusters_top_words = self.get_top_words(clusters, tweets)
        cluster_summaries = []

        for cluster_top_words in clusters_top_words:
            prompt = "I have a cluster summary described by the following keywords: " + str(
                cluster_top_words['top_words']) + ". Based on the previous keywords, what is the cluster title ?"

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            outputs = model.generate(input_ids, max_new_tokens=30)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

            cluster_summary = {
                'cluster_id': cluster_top_words['cluster_id'],
                'summary': summary,
                'length': cluster_top_words['length']
            }
            cluster_summaries.append(cluster_summary)

        return cluster_summaries

    def get_tweet_topics(self, clusters, tweets):
        '''
        Identifies the main topic of each tweet cluster using a topic classification model.

        (clusters: list, tweets: list of dict) -> list of dict
        A list of dictionaries, each containing the main topic for a cluster
        and the cluster ID. A pie chart of topic distribution is also displayed.
        '''
        topic_model = f"cardiffnlp/tweet-topic-latest-single"
        tokenizer = AutoTokenizer.from_pretrained(topic_model)

        model = AutoModelForSequenceClassification.from_pretrained(topic_model)
        class_mapping = model.config.id2label

        topic_list = ['arts_&_culture', 'business_&_entrepreneurs', 'pop_culture', 'daily_life', 'sports_&_gaming',
                      'science_&_technology']

        topics = []

        topic_counts = [0 for element in range(6)]

        for i, cluster in enumerate(clusters):

            # Extract tweets in the cluster
            cluster_tweets = [tweets[node]['normalized'] for node in cluster]

            cluster_topics = [0 for element in range(6)]

            for cluster_tweet in cluster_tweets:
                encoded_input = tokenizer(cluster_tweet, return_tensors='pt')
                output = model(**encoded_input)
                scores = output[0][0].detach().numpy()
                scores = softmax(scores)
                topic_index = np.argmax(scores)
                topic_counts[topic_index] = cluster_topics[topic_index] + 1
                cluster_topics[topic_index] = cluster_topics[topic_index] + 1

            cluster_topic = np.argmax(cluster_topics)
            topics.append({'cluster_id': i + 1, 'topic': topic_list[cluster_topic]})

        plt.pie(topic_counts, labels=topic_list)
        plt.show()

        return topics
