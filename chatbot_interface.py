import openai

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

            # Search in FAISS index
            distances, indices = self.index.search([query_embedding], top_k)

            # Retrieve relevant chunks
            retrieved_chunks = [self.chunks[i] for i in indices[0]]

            # Concatenate chunks as context
            context = " ".join(retrieved_chunks)

            # Use GPT to generate a response
            gpt_response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Context: {context}\n\nAnswer the question: {query}",
                max_tokens=150
            )
            chatbot_response = gpt_response["choices"][0]["text"].strip()

            return context, chatbot_response
        except Exception as e:
            return "", f"Error generating response: {e}"
