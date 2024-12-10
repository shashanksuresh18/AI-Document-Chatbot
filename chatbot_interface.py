import openai
import numpy as np
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)

class ChatbotInterface:
    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks

    def answer_query(self, query, top_k=3):
        try:
            # Generate embedding for the query
            response = openai.Embedding.create(
                input=[query],
                model="text-embedding-ada-002"
            )
            query_embedding = response["data"][0]["embedding"]
            logging.info(f"Query embedding type: {type(query_embedding)}")
            logging.info(f"Query embedding shape: {np.array([query_embedding]).shape}")

            # Search in FAISS index
            distances, indices = self.index.search(np.array([query_embedding]), top_k)
            logging.info(f"FAISS search distances: {distances}")
            logging.info(f"FAISS search indices: {indices}")

            # Retrieve relevant chunks
            retrieved_chunks = [self.chunks[i] for i in indices[0]]
            logging.info(f"Retrieved chunks: {retrieved_chunks}")

            # Concatenate chunks as context
            context = " ".join(retrieved_chunks)

            # Use GPT to generate a response
            gpt_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Updated to ChatCompletion API
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Context: {context}\n\nAnswer the question: {query}"}
                ],
                max_tokens=150
            )
            chatbot_response = gpt_response["choices"][0]["message"]["content"].strip()

            return context, chatbot_response
        except Exception as e:
            logging.error(f"Error during query processing: {e}")
            return "", f"Error generating response: {e}"
