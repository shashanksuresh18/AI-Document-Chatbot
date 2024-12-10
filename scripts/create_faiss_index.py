import os
import openai
import faiss
import numpy as np
import json

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment

# Function to split text into smaller chunks
def chunk_text(text, chunk_size=150):
    sentences = text.split(".")
    chunks = ['.'.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

# Function to generate embeddings for a batch of chunks
def get_embeddings_batch(chunks):
    try:
        response = openai.Embedding.create(
            input=chunks,  # Pass all chunks as a list
            model="text-embedding-ada-002"
        )
        return [data["embedding"] for data in response["data"]]
    except Exception as e:
        print(f"Error while generating embeddings: {e}")
        return []

# Create FAISS index and metadata
dimension = 1536  # Size of the embedding vector
index = faiss.IndexFlatL2(dimension)  # FAISS index
metadata = []  # To store metadata (file name and chunk details)

# Process all text files in the data/ folder
data_folder = "data"
if not os.path.exists(data_folder):
    print(f"Error: '{data_folder}' folder not found.")
    exit()

for file_name in os.listdir(data_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(data_folder, file_name)
        print(f"Processing file: {file_name}")
        
        with open(file_path, 'r') as file:
            document_text = file.read()
        
        chunks = chunk_text(document_text)
        embeddings = get_embeddings_batch(chunks)
        
        if embeddings:
            index.add(np.array(embeddings))  # Add embeddings to the FAISS index
            
            # Add metadata for each chunk
            for i, chunk in enumerate(chunks):
                metadata.append({
                    "file_name": file_name,
                    "chunk_index": i,
                    "chunk_text": chunk
                })

# Save the FAISS index to a file
faiss.write_index(index, "document_index.faiss")
print("FAISS index created and saved as 'document_index.faiss'.")

# Save metadata to a JSON file
with open("metadata.json", "w") as meta_file:
    json.dump(metadata, meta_file)
print("Metadata saved as 'metadata.json'.")
