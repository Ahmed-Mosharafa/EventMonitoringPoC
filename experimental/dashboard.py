import ast
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sent2vec.vectorizer import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import heapq
from sklearn import preprocessing as sp

# Set up the Streamlit page layout to be wide
st.set_page_config(layout="wide")

# Set the initial query for similarity comparison
query = "Various notable events and achievements involving public figures, entertainment, and current affairs."

# Load and parse the JSON data from 'test_file'
with open('test_file') as f:
    data = ast.literal_eval(f.read())

# Convert the JSON data to a pandas DataFrame for easy manipulation
data_df = pd.DataFrame(data)

# Display the title and hot topics on the Streamlit dashboard
st.title("Event Monitoring Dashboard")
st.write("Hot Topics")
st.write(data_df["summary"])  # Display summaries from the data

# Initialize the plot for visual representation
fig, ax = plt.subplots(figsize=(24, 12))

# Extract a list of titles (summaries) for further processing
list_titles = data_df["summary"].to_list()

# **Vectorization and Similarity Calculation**

# Initialize the Sent2Vec vectorizer
vectorizer = Vectorizer()

# Prepare the list of vectors by adding the query at the beginning
vectors_list = data_df["summary"].to_list()
vectors_list.insert(0, query)

# Generate vector embeddings for the list of summaries and the query
vectorizer.run(vectors_list)
vectors = vectorizer.vectors

# Compute cosine similarity between the query vector and all summary vectors
sim = []
for i, vector in enumerate(vectors):
    vectors[i] = vector.reshape(1, -1)  # Reshape vectors for similarity calculation
for vector in vectors:
    similarity = cosine_similarity(vector, vectors[0])[0][0]  # Compare each vector with the query
    sim.append(similarity)

# **Normalization of Similarity Values**

# Normalize the similarity values to the range [0, 1]
min_value = min(sim)
max_value = heapq.nlargest(2, sim)[-1]
normalized = [(x - min_value) / (max_value - min_value) for x in sim]

# Normalize cluster sizes to ensure proportional representation
cluster_size = data_df["length"].to_list()
normalized_cluster_size = [float(i) / max(cluster_size) for i in cluster_size]

# **Mapping Similarity to 2D Plane**

def map_to_2d_plane(numbers):
    """
    Map a list of normalized similarity values to a 2D plane using cosine and sine functions.
    """
    # Normalize the numbers to the range [0, 1]
    numbers = [float(x) for x in numbers]
    numbers = [(x - min(numbers)) / (max(numbers) - min(numbers)) for x in numbers]
    
    # Generate 2D coordinates for each normalized value
    coordinates = [(np.cos(2 * np.pi * x) * 100, np.sin(2 * np.pi * x) * 100) for x in numbers]
    
    # Scale the coordinates to fit within a specified range
    scaler = sp.MinMaxScaler(feature_range=(0, 30))
    scaled_data = scaler.fit_transform(coordinates)
    
    return scaled_data

# Generate the 2D coordinates based on normalized similarity values
coordinates = map_to_2d_plane(normalized)
positions = np.delete(coordinates, 0, 0)  # Remove the first position, which corresponds to the query

# **Radius Calculation Based on Cluster Size**

# Set maximum possible radius for the circles
max_radius = 350

# Calculate the radii of circles based on normalized cluster sizes
radii = [max_radius * (value / 75) for value in normalized_cluster_size]

# **Resolve Circle Overlaps**

def resolve_overlaps(positions, radii, max_iter=1000):
    """
    Resolve overlaps between circles by adjusting their positions iteratively.
    """
    def are_overlapping(p1, r1, p2, r2):
        """
        Check if two circles are overlapping.
        """
        distance = np.linalg.norm(np.array(p1) - np.array(p2))
        return distance < (r1 + r2)
    
    def move_apart(p1, r1, p2, r2):
        """
        Move two overlapping circles apart to resolve overlap.
        """
        direction = np.array(p1) - np.array(p2)
        distance = np.linalg.norm(direction)
        overlap = r1 + r2 - distance
        move_distance = overlap / 2
        direction_normalized = direction / distance
        p1_new = np.array(p1) + direction_normalized * move_distance
        p2_new = np.array(p2) - direction_normalized * move_distance
        return p1_new.tolist(), p2_new.tolist()
    
    # Convert positions to lists for easier manipulation
    positions = [pos.tolist() for pos in positions]
    
    # Iteratively resolve overlaps
    for _ in range(max_iter):
        moved = False
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if are_overlapping(positions[i], radii[i], positions[j], radii[j]):
                    positions[i], positions[j] = move_apart(positions[i], radii[i], positions[j], radii[j])
                    moved = True
        if not moved:
            break

    return positions

# Adjust positions to avoid overlap between circles
new_positions = resolve_overlaps(positions, radii)

# **Plotting the Circles**

# Draw circles and add text annotations for each summary
for (x, y), radius, summary in zip(new_positions, radii, list_titles):
    circle = plt.Circle((x, y), radius, color='blue', alpha=0.5)
    ax.add_patch(circle)
    fontsize = radius * 1.7  # Set the font size proportional to the circle radius
    ax.text(x, y, summary, ha='center', va='center', fontsize=fontsize, color='black')

# Draw a central circle representing the query
central_radius = 5
central_circle = plt.Circle((15, 15), central_radius, color='red', alpha=0.3)
ax.add_patch(central_circle)
ax.text(15, 15, query, ha='center', va='center', fontsize=central_radius, color='black')

# Set plot limits and aspect ratio to ensure proper scaling
grid_size = int(len(data) ** 0.5) + 1
spacing_factor = 9
x_limit = grid_size * spacing_factor
y_limit = (len(data) // grid_size + 1) * spacing_factor
ax.set_xlim(-spacing_factor, x_limit)
ax.set_ylim(-spacing_factor, y_limit)
ax.set_aspect('equal')  # Maintain aspect ratio
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')  # Turn off axis

# Display the plot in the Streamlit app
st.pyplot(fig)
