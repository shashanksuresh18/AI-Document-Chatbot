import faiss
import numpy as np
import json
import openai
import os

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment

# Load the FAISS index
index = faiss.read_index("document_index.faiss")

# Load metadata
with open("metadata.json", "r") as meta_file:
    metadata = json.load(meta_file)

# Function to generate embeddings for a query
def get_query_embedding(query):
    response = openai.Embedding.create(
        input=[query],
        model="text-embedding-ada-002"
    )
    return np.array(response["data"][0]["embedding"]).reshape(1, -1)

# Get user query
query = input("Enter your search query: ")

# Generate embedding for the query
query_embedding = get_query_embedding(query)

# Perform the search
k = 3  # Number of results to retrieve
distances, indices = index.search(query_embedding, k)

# Display the results
print("\nTop results:")
for i, idx in enumerate(indices[0]):
    if idx == -1:
        continue
    result_metadata = metadata[idx]
    print(f"{i+1}. File: {result_metadata['file_name']}")
    print(f"   Chunk Index: {result_metadata['chunk_index']}")
    print(f"   Chunk Text: {result_metadata['chunk_text']}")
    print(f"   Distance: {distances[0][i]:.2f}")
