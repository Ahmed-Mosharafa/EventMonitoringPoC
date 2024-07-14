import ast
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sent2vec.vectorizer import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np
st.set_page_config(layout="wide")
from sklearn import preprocessing as sp
import heapq

# Example query with similarity to topic titles
query = "Various notable events and achievements involving \n public figures, entertainment, and current affairs."
#open and read JSON
with open('test_file') as f:
    data = ast.literal_eval(f.read())
    
#Set Page structure 
st.title("Event Monitoring dashboard")
data_df = pd.DataFrame(data)
st.write("Hot Topics")
st.write(data_df["summary"])
fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(24, 12))
n = len(data)
grid_size = int(n**0.5) + 1


list_titles = data_df["summary"].to_list()

# Calculate vector similarity
vectorizer = Vectorizer()
vectors_list = data_df["summary"].to_list()
vectors_list.insert(0,query)
vectorizer.run(vectors_list)
vectors = vectorizer.vectors
# st.write(vectors_list)

# vectorizer.run([quer])
# query_embedding = vectorizer.vectors

sim = []
for i,vector in enumerate(vectors):
    vectors[i] = vector.reshape(1,-1)

for vector in vectors:
    similarity = cosine_similarity(vector, vectors[0])[0][0]
    sim.append(similarity)
# transformed = [s ** 3 for s in sim]

min_value = min(sim)
max_value=  heapq.nlargest(2, sim)[-1]
# st.write(min_value)
# st.write(max_value)

normalized = [(x - min_value) / (max_value - min_value) for x in sim]
# new_max = 10000000000000
    # Expand the normalized values to the new range [0, new_max]
# transformed = [x * new_max for x in normalized]

# min_value_new = min(transformed)
# max_value_new=  max(transformed) 
# normalized_new = [(x - min_value_new) / (max_value_new - min_value_new) for x in transformed]

# transformed = [np.log1p(s) / np.log(10) for s in sim]

# st.write(normalized)


#normalize cluster sizes
cluster_size = data_df["length"].to_list()
normalized_cluster_size = [float(i)/max(cluster_size) for i in cluster_size]

#initialize cluser positions
positions = []
spacing_factor = 9
# for i in range(n):


def map_to_2d_plane(numbers):
    numbers = [float(x) for x in numbers]
    numbers = [(x - min(numbers)) / (max(numbers) - min(numbers)) for x in numbers]
    
    # Map each value to a coordinate in the 2D plane
    coordinates = [(np.cos(2 * np.pi * x)*100, np.sin(2 * np.pi * x)*100) for x in numbers]
    min_value_x = coordinates[0][0]
    min_value_y = coordinates[0][1]
    for coo in coordinates:
        if coo[0] < min_value_x:
            min_value_x = coo[0]
        if coo[1] < min_value_y:
            min_value_x = coo[1]
    scaler = sp.MinMaxScaler(feature_range=(0, 30))
    model = scaler.fit(coordinates)
    scaled_data = model.transform(coordinates)
    # for coo in coordinates:
        
        
    return scaled_data


coordinates = map_to_2d_plane(normalized)
# positions = coordinates[1:, 1:]
positions = np.delete(coordinates, 0, 0)

# st.write(positions)

# for i,similarity in enumerate(normalized):
#     if i == 0 :
#         continue
#     # distance_x = math.sqrt(value ** 2 + 0 ** 2)
#     a =0
#     b = 30
#     x_map= a + similarity *(b-a) 
#     y_map = a + (1-similarity)*(b-a) 
#     x = x_map 
#     y = y_map 
#     # x = (i % grid_size) * spacing_factor  # 3 is an arbitrary spacing factor to avoid overlap
#     # y = (i // grid_size) * spacing_factor
#     positions.append((x, y))
# st.write(positions)

# Calculate radius based on cluster size
max_radius = 350
radii = [max_radius * (value / 75) for value in normalized_cluster_size]
# print(len(radii))
# print("hey")
# print(len(positions))

# def get_max_font_size(ax, x, y, radius, text):
#     fontsize = radius * 2  # Start with a reasonable font size
#     while fontsize > 5:  # Minimum font size
#         bbox = ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color='black').get_window_extent(renderer=fig.canvas.get_renderer())
#         # Adjust the font size if text is out of circle bounds
#         if bbox.width / 2 <= radius and bbox.height / 2 <= radius:
#             break
#         fontsize -= 1
#     return fontsize
#Main loop drawing and positioning the circles

def resolve_overlaps(positions, radii, max_iter=1000):
    def are_overlapping(p1, r1, p2, r2):
        distance = np.linalg.norm(np.array(p1) - np.array(p2))
        return distance < (r1 + r2)
    
    def move_apart(p1, r1, p2, r2):
        direction = np.array(p1) - np.array(p2)
        distance = np.linalg.norm(direction)
        overlap = r1 + r2 - distance
        move_distance = overlap / 2
        direction_normalized = direction / distance
        p1_new = np.array(p1) + direction_normalized * move_distance
        p2_new = np.array(p2) - direction_normalized * move_distance
        return p1_new.tolist(), p2_new.tolist()
    
    positions = [pos.tolist() for pos in positions]
    
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
# st.write(positions.pop(0))
# st.write(radii)
new_positions = resolve_overlaps(positions, radii)

# st.write(new_positions)
for (x, y), radius,summary in zip(new_positions,radii,list_titles):
    
    circle = plt.Circle((x, y), radius, color='blue', alpha=0.5)
    ax.add_patch(circle)
    # fontsize = get_max_font_size(ax, x, y, radius, summary)
    fontsize = radius * 1.7
    ax.text(x, y, f'{summary}', ha='center', va='center', fontsize=fontsize, color='black')

# Add a central circle
central_radius = 5  # Radius of the central circle
central_circle = plt.Circle((15, 15), central_radius, color='red', alpha=0.3)
ax.add_patch(central_circle)

# Add text in the central circle
central_text = query
# central_fontsize = get_max_font_size(ax, 0, 0, central_radius, central_text)
ax.text(15, 15, central_text, ha='center', va='center', fontsize=central_radius , color='black')

# Set limits and aspect
x_limit = grid_size * spacing_factor
y_limit = (n // grid_size + 1) * spacing_factor
# ax.set_xlim(-5, grid_size * 3)
# ax.set_ylim(-5, (n // grid_size + 1) * 3)
ax.set_xlim(-spacing_factor, x_limit)
ax.set_ylim(-spacing_factor, y_limit)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')  

# Display the plot
st.pyplot(fig)