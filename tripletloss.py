from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# Load tokenizer and model from Hugging Face's Transformers for code embeddings
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")

# Define directory and get list of Java files
extracted_dir = r'C:\Users\ATABDELLATIF\Documents\GitHub\DATASETS\Demo'
all_files = [os.path.join(extracted_dir, f) for f in os.listdir(extracted_dir) if f.endswith('.java')]

# Function to get CodeBERT embeddings for a file
def get_embedding_for_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
    outputs = codebert_model(**inputs)
    return outputs.pooler_output.detach().numpy()  # Convert tensor to numpy for compatibility

# Compute embeddings for all files
embeddings = {file: get_embedding_for_file(file) for file in all_files}
#ref: https://github.com/FSMVU-Tubitak-1001-Kod-Analiz/Triplet-net-keras/blob/triplet-for-sourcecode/Triplet%20NN%20Test%20on%20MNIST.ipynb
# Function to find positive (similar) and negative (dissimilar) examples for an anchor
def find_positive_negative_Tripletloss(anchor_embedding, embeddings, all_files):
    # Calculate cosine distances between anchor and all other embeddings
    distances = {file: cosine(anchor_embedding, embeddings[file]) for file in all_files}
    # Sort files based on distance to find the most similar and dissimilar ones
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    positive = sorted_distances[1][0]  # Closest file
    negative_candidates = sorted_distances[len(sorted_distances)//2:]  # Distant half as candidates for negative
    negative = random.choice(negative_candidates)[0]  
    return positive, negative
def find_positive_negative_knn(anchor_embedding, embeddings):
    # Convert embeddings to a list of vectors for NearestNeighbors
    embedding_list = list(embeddings.values())
    file_list = list(embeddings.keys())
    
    # Fit the NearestNeighbors model
    neigh = NearestNeighbors(n_neighbors=len(embedding_list), metric='cosine')
    neigh.fit(embedding_list)
    
    # Find the nearest neighbors for the anchor embedding
    distances, indices = neigh.kneighbors([anchor_embedding], n_neighbors=len(embedding_list))
    
    # Find the positive and negative examples (excluding the anchor itself which is at index 0)
    positive_index = indices[0][1]  # The closest neighbor (index 1)
    negative_index = indices[0][-1]  # The furthest neighbor
    positive = file_list[positive_index]
    negative = file_list[negative_index]
    
    return positive, negative

# Generate triplets using k-NN
triplets = []
for anchor_file in all_files:
    anchor_embedding = embeddings[anchor_file]
    positive, negative = find_positive_negative_knn(anchor_embedding, embeddings)
    triplets.append((anchor_file, positive, negative))

# Print out the triplets
for triplet in triplets:
    print(triplet)
# Generate triplets (anchor, positive, negative)
triplets = []
for anchor_file in all_files:
    anchor_embedding = embeddings[anchor_file]
    positive, negative = find_positive_negative_Tripletloss(anchor_embedding, embeddings, all_files)
    triplets.append((anchor_file, positive, negative))

# Define triplet loss function
def triplet_loss(alpha=0.4):
    # Custom loss for training
    def loss(y_true, y_pred):
        # Split the predictions into anchor, positive, and negative parts
        total_length = y_pred.shape.as_list()[-1]
        anchor = y_pred[:, 0:int(total_length*1/3)]
        positive = y_pred[:, int(total_length*1/3):int(total_length*2/3)]
        negative = y_pred[:, int(total_length*2/3):int(total_length*3/3)]
        # Calculate squared distances
        pos_dist = K.sum(K.square(anchor - positive), axis=1)
        neg_dist = K.sum(K.square(anchor - negative), axis=1)
        # Compute triplet loss
        basic_loss = pos_dist - neg_dist + alpha
        return K.maximum(basic_loss, 0.0)
    return loss

# Create shared network for processing embeddings
def create_shared_network(input_shape):
    # Define a neural network to process the input embeddings
    input = Input(shape=input_shape)
    x = Dense(256, activation='relu')(input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    return Model(inputs=[input], outputs=[x])

# Prepare input tensors for training
input_shape = (768,)  # Shape of CodeBERT embeddings
shared_network = create_shared_network(input_shape)

# Reshape embeddings and create arrays for training
anchor_embeddings = np.array([embeddings[triplet[0]].reshape(-1) for triplet in triplets])
positive_embeddings = np.array([embeddings[triplet[1]].reshape(-1) for triplet in triplets])
negative_embeddings = np.array([embeddings[triplet[2]].reshape(-1) for triplet in triplets])

# Define the model with triplet architecture
anchor_input = Input(shape=input_shape, name='anchor_input')
positive_input = Input(shape=input_shape, name='positive_input')
negative_input = Input(shape=input_shape, name='negative_input')

# Process each input through the shared network
encoded_anchor = shared_network(anchor_input)
encoded_positive = shared_network(positive_input)
encoded_negative = shared_network(negative_input)

# Concatenate the outputs for the final model
merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)

# Compile and train the model
model.compile(loss=triplet_loss(alpha=0.4), optimizer='adam')
y_dummy = np.empty((anchor_embeddings.shape[0],))  # Dummy labels for training
model.fit([anchor_embeddings, positive_embeddings, negative_embeddings], y_dummy, epochs=10, batch_size=32)