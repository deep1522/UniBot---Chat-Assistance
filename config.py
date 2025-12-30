import os
import json
import streamlit as st
import boto3
from dotenv import load_dotenv

# Import the correct pinecone package as determined by the previous error resolution
from pinecone import Pinecone as PineconeClient

from langchain_community.embeddings import BedrockEmbeddings
from google.oauth2 import service_account

# --- Environment Variable Loading ---
# Load variables from .env file immediately on script startup.
# This ensures os.getenv() can access them below.
load_dotenv()

# --- Helper Function to Read Configuration ---
def load_env_var(key, default=None):
    """
    Loads a variable from Streamlit secrets (for deployment) or environment variables (for local dev).
    Priority: Streamlit Secrets > Environment Variables > Default Value.
    """
    return st.secrets.get(key) or os.getenv(key) or default

# --- Configuration Variable Assignment ---
AWS_ACCESS_KEY_ID = load_env_var("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = load_env_var("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME = load_env_var("AWS_DEFAULT_REGION", "us-east-1")
PINECONE_API_KEY = load_env_var("PINECONE_API_KEY")
GOOGLE_CREDS_JSON_STR_OR_PATH = load_env_var("GOOGLE_APPLICATION_CREDENTIALS")

# Define constants used across the application
INDEX_NAME = "langchain" # Or load from environment if preferred

# --- Service Client Initializations (Cached) ---

@st.cache_resource
def get_bedrock_client():
    """Initializes and returns the AWS Bedrock runtime client."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        st.error("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        return None
    try:
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION_NAME,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        return client
    except Exception as e:
        st.error(f"Error initializing Bedrock client: {e}")
        return None

@st.cache_resource
def get_bedrock_embeddings(_client):
    """Initializes and returns Bedrock embeddings model."""
    if _client:
        return BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=_client)
    return None

@st.cache_resource
def get_pinecone_client():
    """Initializes and returns the Pinecone client."""
    if not PINECONE_API_KEY:
        st.error("Pinecone API key not found. Please set PINECONE_API_KEY environment variable.")
        return None
    try:
        return PineconeClient(api_key=PINECONE_API_KEY)
    except Exception as e:
        st.error(f"Error initializing Pinecone client: {e}")
        return None

@st.cache_resource
def get_google_credentials():
    """Loads Google credentials from a JSON string or file path."""
    if not GOOGLE_CREDS_JSON_STR_OR_PATH:
        st.error("Google credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
        return None

    try:
        # Option 1: Try loading as a direct JSON string
        credentials_dict = json.loads(GOOGLE_CREDS_JSON_STR_OR_PATH)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    except json.JSONDecodeError:
        # Option 2: If not a JSON string, assume it's a file path
        if os.path.exists(GOOGLE_CREDS_JSON_STR_OR_PATH):
            credentials = service_account.Credentials.from_service_account_file(GOOGLE_CREDS_JSON_STR_OR_PATH)
        else:
            st.error(f"Google credentials file not found at path: {GOOGLE_CREDS_JSON_STR_OR_PATH}")
            return None
    except Exception as e:
        st.error(f"Error loading Google credentials: {e}")
        return None
    
    return credentials

# --- Instantiate Clients for Export ---
# These instances will be imported by other modules (graph_workflow.py, data_utils.py)
bedrock_client = get_bedrock_client()
pinecone_client = get_pinecone_client()
bedrock_embeddings = get_bedrock_embeddings(bedrock_client)
google_credentials = get_google_credentials()