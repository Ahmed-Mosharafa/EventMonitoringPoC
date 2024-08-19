from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.tweet_preprocessor import TweetPreprocessor
from graph.contextual_knowledge import ContextualEmbeddings
from graph.structural_relation import StructuralRelation
from clustering.mcl_clustering import MarkovClustering
from summary.event_summary import EventSummarizer

app = Flask(__name__)
CORS(app)

# Beispieltext, der an Streamlit gesendet wird
current_events = []

current_query = ""

@app.route('/events', methods=['GET'])
def get_text():
    global current_events
    global current_query
    return jsonify({'events': current_events, 'query': current_query})

@app.route('/update-events', methods=['POST'])
def update_text():
    global current_events
    global current_query

    conversation = request.json["conversations"]
    current_query = request.json["query"]

    preprocessor = TweetPreprocessor()

    contextual = ContextualEmbeddings()
    structural = StructuralRelation()

    tweets = []

    for conv_dict in conversation:
        conv_text = conv_dict['dialogue']
        conv_id = conv_dict['id']
        tweets.append({'text': conv_text, 'id': conv_id})

    tweets = preprocessor.process_tweets(tweets)
    contextual_similarity = contextual.get_similarity_matrix(tweets)
    structural_similarity = structural.get_struct_relation_matrix(tweets)

    graph_matrix = 0.5 * (contextual_similarity + structural_similarity)

    # Markov Clustering
    mcl = MarkovClustering(graph_matrix, tweets)
    clusters = mcl.apply_mcl_algorithm()[:15]

    # mcl.plot_clusters()

    # Event Summaries
    summarizer = EventSummarizer(clusters, tweets)
    cluster_summaries = summarizer.generate_summary(clusters, tweets)

    current_events = cluster_summaries

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
