from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pandas as pd

#Load dataset
dataset = load_dataset('chargoddard/WebInstructSub-prometheus', split='train')

# Remove specified columns
dataset = dataset.remove_columns(['model_name'])

# Choose the top 1000 rows
dataset = dataset.select(range(1000))

# Concatenate 'instruction' and 'generation' columns
def concatenate_columns(example):
    example['text'] = example['instruction'] + " " + example['generation']
    return example

dataset = dataset.map(concatenate_columns)

# Assume 'text' is the column containing the concatenated text data
texts = dataset['text']

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode texts
embeddings = model.encode(texts, convert_to_tensor=True)

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(embeddings.cpu().numpy()) #computes the pairwise cosine similarity between all the text embeddings.
cosine_sim_matrix

# Set similarity threshold
threshold = 0.95

# Create mask for values above threshold
similarity_mask = cosine_sim_matrix > threshold  #creates a boolean matrix
similarity_mask

# Group similar texts
groups = []
visited = set()

for i, row in enumerate(similarity_mask):
    if i not in visited:
        group = np.where(row)[0]
        groups.append(group)
        visited.update(group)

# Deduplicate texts
unique_texts_indices = [group[0] for group in groups]
deduplicated_dataset = dataset.select(unique_texts_indices)

# Remove specified columns
deduplicated_dataset = deduplicated_dataset.remove_columns(['text'])

# Convert to pandas DataFrame
df = deduplicated_dataset.to_pandas()

# Save the DataFrame to a CSV file
csv_file_path = "WebInstructSub-prometheus_dataset_1000.csv"
df.to_csv(csv_file_path, index=False)
