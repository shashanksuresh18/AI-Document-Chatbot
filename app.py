import os
import streamlit as st
import openai
import faiss
import numpy as np
from chatbot_interface import ChatbotInterface

# Set OpenAI API Key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Helper functions
def chunk_text(text, chunk_size=150):
    sentences = text.split(".")
    chunks = ['.'.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

def get_embeddings_batch(chunks):
    try:
        response = openai.Embedding.create(
            input=chunks,
            model="text-embedding-ada-002"
        )
        return [data["embedding"] for data in response["data"]]
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return []

# Streamlit UI
st.title("AI-Powered Document Search & Chatbot")
st.write("Upload a document, ask questions, and get relevant answers!")

# File upload section
uploaded_file = st.file_uploader("Upload a document (.txt only)", type=["txt"])
if uploaded_file:
    document_text = uploaded_file.read().decode("utf-8")
    st.success("File uploaded successfully!")

    # Create embeddings
    st.info("Processing document and generating embeddings...")
    chunks = chunk_text(document_text)
    embeddings = get_embeddings_batch(chunks)

    if embeddings:
        # Initialize FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        st.success("FAISS index created successfully!")

        # Save embeddings and document text for chatbot
        chatbot = ChatbotInterface(index, chunks)

        # Query input section
        user_query = st.text_input("Enter your search query:")
        if user_query:
            context, response = chatbot.answer_query(user_query)
            st.write("### Chatbot Response:")
            st.write(response)
