# data_utils.py
import streamlit as st
from langchain_google_community import GoogleDriveLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import google_credentials, bedrock_embeddings, INDEX_NAME, pinecone_client

def data_ingestion(folder_id):
    if google_credentials is None:
        st.error("Google Drive credentials are not available.")
        return []

    loader = GoogleDriveLoader(
        folder_id=folder_id,
        credentials=google_credentials,
        recursive=False
    )
    try:
        documents = loader.load()
        # Corrected chunking strategy
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        return docs
    except Exception as e:
        st.error(f"Error loading documents from Google Drive: {e}")
        return []

def update_vector_store(docs, namespace):
    try:
        PineconeVectorStore.from_documents(
            docs,
            embedding=bedrock_embeddings,
            index_name=INDEX_NAME,
            namespace=namespace
        )
        st.success(f"✅ Documents successfully updated in namespace '{namespace}'.")
    except Exception as e:
        st.error(f"An error occurred while updating the vector store: {e}")