
# **Event Monitoring**

## **Overview**

This project is designed to detect topics within tweets, cluster the tweets based on these identified topics, and visualize the results. Leveraging  natural language processing (NLP) techniques, the system processes tweet data to determine the underlying topics, groups the tweets into clusters that share similar themes, and presents these clusters in a dashboard. The visualization helps in understanding the distribution of topics and their relative significance.

## **Features**

- **Topic Detection**: Automatically identifies the main topics within a set of tweets.
- **Tweet Clustering**: Groups tweets into clusters based on their detected topics using contextual similarity.
- **Cluster Visualization**: Displays the clusters on a 2D plane, with each cluster’s size reflecting the number of tweets within that topic.


## **Components**

### **1. Normalization of Tweets**

This part of the project involves preprocessing and tokenizing tweets to prepare them for further analysis. Each tweet is normalized, and tokenized fields are added for more accurate topic detection and similarity calculations.


### **2. DataLoader Class**

Handles loading and processing of datasets, including time-windowing for events.

### **3. Markov Clustering**

Performs clustering of tweets based on their similarity using the Markov Clustering Algorithm (MCL).


### **4. Contextual Similarity Calculation**

Computes contextual similarity between tweets using Sentence Transformers, allowing for more accurate clustering based on context.

### **5. Streamlit Dashboard**

This is the interface of the application where users can view with the clusters.

- **Key Features**:
  - Displays a list of hot topics.
  - Calculates and visualizes similarities between topics using a 2D plane.

## **Installation and Setup**

### **Requirements**

- Python 3.7+
- Streamlit
- Pandas
- Matplotlib
- scikit-learn
- SentenceTransformers
- Sent2Vec

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/Ahmed-Mosharafa/EventMonitoringPoC
   cd event-monitoring-dashboard
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset (`event2012.csv` or your equivalent data file) in the `datasets/` directory.

### **Running the Application**

To run the backend, execute the following command in your terminal:
```bash
python backend.py
```

To run the dashboard, execute the following command in your terminal:
```bash
streamlit run frontend.py
```

This will launch the Backend Server and the Streamlit application in your web browser, where you can explore the tweet clusters and their associated topics in combination with Automated Journalist App.
