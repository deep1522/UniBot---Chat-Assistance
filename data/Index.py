import json
import os
import streamlit as st
import boto3
from dotenv import load_dotenv

# Import necessary libraries from LangChain and providers
from google.oauth2 import service_account
from langchain_google_community import GoogleDriveLoader
from langchain_aws import BedrockEmbeddings  # Correct, non-deprecated import
from langchain_community.llms.bedrock import Bedrock
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever

# Import Pinecone client
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec
import time

# Load environment variables from a .env file for local development
load_dotenv()

# --- AWS & Pinecone Setup ---
def get_aws_credentials():
    """Fetches AWS credentials from Streamlit secrets or local environment variables."""
    aws_access_key_id = st.secrets.get("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = st.secrets.get("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    return aws_access_key_id, aws_secret_access_key

def get_pinecone_api_key():
    """Fetches Pinecone API key from Streamlit secrets or local environment variables."""
    return st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")

aws_access_key_id, aws_secret_access_key = get_aws_credentials()
aws_region_name = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Set up the Bedrock runtime client and the embeddings model
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Initialize Pinecone
PINECONE_API_KEY = get_pinecone_api_key()
pc = PineconeClient(api_key=PINECONE_API_KEY)
INDEX_NAME = "langchain"  # Your Pinecone index name

# --- User Database and Google Drive Configuration ---
USERS_DATA = {
    "senior@example.com": {"password": "seniorpassword", "role": "senior"},
    "regular@example.com": {"password": "regularpassword", "role": "regular"}
}

# IMPORTANT: Ensure these IDs match your Google Drive folders
FOLDER_IDS = {
    "public": "1uMxXe08q2VJ06sSxajkOXXTy7OZy8lQf",
    "regular": "1ZIXet5KWj8H9YLxqJ7RR0SORfaBppGXz",
    "senior": "1oFoZj_lqz3O9H_H08JmU7EalXcb5KyRA"
}

# --- Google Drive Credentials Setup ---
def setup_google_credentials():
    """Handles Google credentials for both cloud (JSON string) and local (file path) environments."""
    credentials_source = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_source:
        st.error("`GOOGLE_APPLICATION_CREDENTIALS` not found in secrets or environment.")
        return None
    try:
        # Try loading from JSON string first
        credentials_dict = json.loads(credentials_source)
        return service_account.Credentials.from_service_account_info(credentials_dict)
    except json.JSONDecodeError:
        # Fallback to loading from file path
        if os.path.exists(credentials_source):
            return service_account.Credentials.from_service_account_file(credentials_source)
        st.error("Google credentials are not a valid JSON string and the file path was not found.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred with Google credentials: {e}")
        return None

# --- Core Application Functions ---
def data_ingestion(gdrive_credentials, folder_id):
    """Loads documents from a specific Google Drive folder and splits them into chunks."""
    if not gdrive_credentials:
        st.error("Google Drive credentials failed to load. Ingestion skipped.")
        return []
    
    print(f"--- Starting ingestion for folder: {folder_id} ---")
    try:
        loader = GoogleDriveLoader(folder_id=folder_id, credentials=gdrive_credentials)
        documents = loader.load()
        if not documents:
            st.warning(f"No documents found in folder {folder_id}. Ensure the service account has 'Viewer' permissions on this folder and its files.")
            return []
        
        print(f"--- Found {len(documents)} document(s) in folder {folder_id}. Splitting text. ---")
        # Using a larger chunk size for better retrieval context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        print(f"--- Created {len(docs)} chunks. ---")
        return docs
    except Exception as e:
        st.error(f"Failed to load documents from Google Drive folder {folder_id}. Error: {e}")
        return []

def get_vector_store(docs, namespace):
    """Creates or updates a Pinecone namespace with document chunks."""
    if not docs:
        st.info(f"No new documents to process for namespace '{namespace}'.")
        return
    try:
        # This check is more robust for Pinecone's API response
        index_list = pc.list_indexes().indexes
        if INDEX_NAME not in [index['name'] for index in index_list]:
            st.info(f"Pinecone index '{INDEX_NAME}' not found. Creating it now...")
            pc.create_index(name=INDEX_NAME, dimension=1024, metric="cosine", spec=ServerlessSpec(cloud="aws", region=aws_region_name))
            while not pc.describe_index(INDEX_NAME).status["ready"]: time.sleep(1)
        
        print(f"--- Upserting {len(docs)} chunks into namespace '{namespace}' ---")
        PineconeVectorStore.from_documents(docs, embedding=bedrock_embeddings, index_name=INDEX_NAME, namespace=namespace)
        st.success(f"✅ Namespace '{namespace}' updated successfully.")
    except Exception as e:
        st.error(f"Error updating Pinecone namespace '{namespace}': {e}")

def get_llama2_llm():
    """Initializes the Llama 3 model from Bedrock."""
    return Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})

prompt_template = """
You are an assistant that provides comprehensive and truthful answers based ONLY on the provided context.
Your goal is to use all relevant information from the documents to answer the user's query as thoroughly as possible.

<context>
{context}
</context>

Question: {question}

Instructions:
- If the question can be answered from the provided context, include ALL relevant details.
- Do NOT use any information from outside of the provided context.
- If the context does not contain any information relevant to the question, you MUST reply with "I do not have that information.".
- Your answer must be factual and directly supported by the context.

Assistant:
"""


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, retriever, query):
    """Performs the RAG chain to get a response from the LLM."""
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        answer = qa.invoke({"query": query})
        return answer['result'], answer['source_documents']
    except Exception as e:
        st.error(f"An error occurred during the search: {e}")
        return "An error occurred. Please check the application logs.", []

# --- User Authentication ---
def login_user(email, password):
    if email in USERS_DATA and USERS_DATA[email]["password"] == password:
        st.session_state.authenticated = True
        st.session_state.username = email
        st.session_state.user_role = USERS_DATA[email]["role"]
        st.rerun()

def logout_user():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# --- Main Application UI ---
def main():
    st.set_page_config(page_title="UNIBOT")
    st.header("UNIBOT - AWS Bedrock + Pinecone")

    # Initialize session state variables
    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if 'last_response' not in st.session_state: st.session_state.last_response = None

    if not st.session_state.authenticated:
        st.subheader("Login")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                login_user(email, password)
    else:
        st.sidebar.button("Logout", on_click=logout_user)
        st.sidebar.write(f"Logged in as: **{st.session_state.username}** ({st.session_state.user_role})")

        user_question = st.text_input("Here to help with your queries...")

        if st.button("Search"):
            if not user_question.strip():
                st.warning("⚠️ Please enter a question.")
                st.session_state.last_response = None
            else:
                with st.spinner("Processing..."):
                    # Build the list of retrievers based on user role
                    retrievers = []
                    namespaces_to_search = ["public_docs"]
                    if st.session_state.user_role == "senior":
                        namespaces_to_search.append("senior_docs")
                    elif st.session_state.user_role == "regular":
                        namespaces_to_search.append("regular_docs")
                    
                    for ns in namespaces_to_search:
                        retrievers.append(
                            PineconeVectorStore.from_existing_index(INDEX_NAME, bedrock_embeddings, namespace=ns)
                            .as_retriever(search_type="similarity", search_kwargs={"k": 5})
                        )

                    ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=[1.0] * len(retrievers))
                    llm = get_llama2_llm()
                    
                    response, sources = get_response_llm(llm, ensemble_retriever, user_question)
                    
                    st.session_state.last_response = response
                    st.session_state.last_question = user_question
                    st.session_state.last_sources = sources

        # Display the last response if it exists
        if st.session_state.get("last_response") is not None:
            st.write(f"**Question:** {st.session_state.last_question}")
            st.write(st.session_state.last_response)
            
            with st.expander("Show Sources 🕵️", expanded=True):
                sources = st.session_state.get("last_sources", [])
                if sources:
                    for doc in sources:
                        st.info(f"**Source:** `{doc.metadata.get('source', 'Unknown')}`")
                        st.text(doc.page_content[:400] + "...")
                else:
                    st.info("No source documents were returned for this query.")
            
            st.write("✅ Anything Else?")

        with st.sidebar:
            st.title("Update Documents")
            if st.button("Update All Documents"):
                with st.spinner("Processing all documents... This may take a while."):
                    gdrive_credentials = setup_google_credentials()
                    if gdrive_credentials:
                        # Ingest each folder into its corresponding namespace
                        get_vector_store(data_ingestion(gdrive_credentials, FOLDER_IDS["public"]), "public_docs")
                        get_vector_store(data_ingestion(gdrive_credentials, FOLDER_IDS["regular"]), "regular_docs")
                        get_vector_store(data_ingestion(gdrive_credentials, FOLDER_IDS["senior"]), "senior_docs")
                        st.sidebar.success("✅ Document update process complete.")

if __name__ == "__main__":
    main()

