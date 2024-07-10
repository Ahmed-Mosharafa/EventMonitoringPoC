import spacy
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
from utils.tweet_preprocessor import TweetPreprocessor
import numpy as np
from scipy.special import softmax


class EventSummarizer:

    def __init__(self, clusters, tweets):
        self.tweet_preprocessor = TweetPreprocessor()

    def extract_named_entities(self, tweets):
        # Load spaCy English model
        nlp = spacy.load("en_core_web_sm")

        named_entities = []
        for tweet in tweets:
            doc = nlp(tweet)
            entities = [ent.text for ent in doc.ents]
            named_entities.append(' '.join(entities))
        return named_entities

    def get_top_words(self, clusters, tweets, top_n=7, ngram_range=(1, 3)):
        # Summarize topics for each cluster
        clusters_top_words = []

        for i, cluster in enumerate(clusters):
            # Extract tweets in the cluster
            cluster_tweets = [tweets[node]['normalized'] for node in cluster]

            # Extract named entities from the tweets
            cluster_entities = self.extract_named_entities(cluster_tweets)

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
                'top_words': top_words
            }
            clusters_top_words.append(cluster_top_words)

        return clusters_top_words

    def generate_summary(self, clusters, tweets):

        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        clusters_top_words = self.get_top_words(clusters, tweets)
        cluster_summaries = []

        for cluster_top_words in clusters_top_words:
            prompt = "I have a cluster summary described by the following keywords: " + str(
                cluster_top_words['top_words']) + ". Based on the previous keywords, what is the cluster summary?"

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            outputs = model.generate(input_ids, max_new_tokens=30)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

            cluster_summary = {
                'cluster_id': cluster_top_words['cluster_id'],
                'summary': summary
            }
            cluster_summaries.append(cluster_summary)

        return cluster_summaries

    def get_tweet_topics(self, clusters, tweets):

        topic_model = f"cardiffnlp/tweet-topic-latest-single"
        tokenizer = AutoTokenizer.from_pretrained(topic_model)

        model = AutoModelForSequenceClassification.from_pretrained(topic_model)
        class_mapping = model.config.id2label

        topic_list = ['arts_&_culture', 'business_&_entrepreneurs', 'pop_culture', 'daily_life', 'sports_&_gaming',
                      'science_&_technology']

        topics = []

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
                cluster_topics[topic_index] = cluster_topics[topic_index] + 1

            cluster_topic = np.argmax(cluster_topics)
            topics.append({'cluster_id': i + 1, 'topic': topic_list[cluster_topic]})
            # print(f"Cluster {i + 1} Topic: {topic_list[cluster_topic]}\n")

        return topics
