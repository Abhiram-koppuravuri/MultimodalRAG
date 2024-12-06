import streamlit as st
import weaviate
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import numpy as np
from weaviate.auth import AuthApiKey
from weaviate.util import generate_uuid5
from weaviate.classes.query import MetadataQuery
import textwrap

# Weaviate configuration
userdata = {
    'URL': 'https://jqx2z1tkqs2cvohiq6nvxw.c0.asia-southeast1.gcp.weaviate.cloud',
    'APIKEY': 'XhatAAoFshT4XPsQu83xTkUAvYAi0XyASDgv'
}
URL = userdata.get('URL')
APIKEY = userdata.get('APIKEY')

# Connect to Weaviate
client = weaviate.Client(
    url=URL,
    auth_client_secret=AuthApiKey(APIKEY)
)

# Hugging Face models
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
chat_model_name = "facebook/bart-base"

# Load embedding model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
hf_model = AutoModel.from_pretrained(embedding_model_name).to(device)

# Load chat model
hf_tokenizer_gen = AutoTokenizer.from_pretrained(chat_model_name)
hf_model_gen = AutoModelForSeq2SeqLM.from_pretrained(chat_model_name).to(device)

# Function to generate embeddings
def get_embedding_hf(text: str) -> np.ndarray:
    inputs = hf_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = hf_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states.mean(dim=1)
    return embeddings.cpu().numpy()[0]

# Function to query Weaviate
def search_multimodal(query: str, limit: int = 3):
    query_vector = get_embedding_hf(query)
    response = (
        client.query
        .get("RAGESGDocuments", [
            "content_type", "url", "audio_path", "transcription",
            "source_document", "page_number", "paragraph_number",
            "text", "image_path", "description"
        ])
        .with_near_vector({"vector": query_vector})
        .with_limit(limit)
        .with_additional(["distance"])
        .do()
    )
    return response.get("data", {}).get("Get", {}).get("RAGESGDocuments", [])


# Function to generate response using BART
def generate_response(query: str, context: str) -> str:
    prompt = f"""
    You are an AI assistant specializing in the Manufacturing Execution System (MES) for the chemical industry.
    Use the following context to answer the user's question.

    Context:
    {context}

    User Question: {query}
    """
    inputs = hf_tokenizer_gen(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = hf_model_gen.generate(inputs["input_ids"], max_length=1024, num_return_sequences=1, temperature=0)
    return hf_tokenizer_gen.decode(outputs[0], skip_special_tokens=True)

# Streamlit app
st.title("RAG-based MES Assistant")
st.write("Enter a query to retrieve relevant documents and generate a response.")

# User input
user_query = st.text_input("Enter your query:")

if user_query:
    # Step 1: Retrieve relevant data
    search_results = search_multimodal(user_query)
    context = ""
    for item in search_results:
        content_type = item.get("content_type")
        if content_type == "audio":
            context += f"Audio Transcription from {item.get('url')}: {item.get('transcription')}\n\n"
        elif content_type == "text":
            context += f"Text from {item.get('source_document')} (Page {item.get('page_number')}, Paragraph {item.get('paragraph_number')}): {item.get('text')}\n\n"
        elif content_type == "image":
            context += f"Image Description from {item.get('source_document')} (Page {item.get('page_number')}, Path: {item.get('image_path')}): {item.get('description')}\n\n"

    # Step 2: Generate response
    response = generate_response(user_query, context)

    # Step 3: Display results
    st.subheader("AI Response")
    st.write(response)

    st.subheader("Retrieved Sources")
    for item in search_results:
        content_type = item.get("content_type")
        st.write(f"Type: {content_type}")
        if content_type == "audio":
            st.write(f"URL: {item.get('url')}")
            st.write(f"Transcription: {item.get('transcription')[:100]}...")
        elif content_type == "text":
            st.write(f"Source: {item.get('source_document')}, Page: {item.get('page_number')}")
            st.write(f"Text: {item.get('text')[:100]}...")
        elif content_type == "image":
            st.write(f"Description: {item.get('description')}")
        st.write(f"Distance to query: {item.get('_additional', {}).get('distance'):.3f}")
        st.write("---")
