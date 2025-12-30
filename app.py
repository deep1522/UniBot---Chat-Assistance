import streamlit as st
import os
from streamlit_cognito_auth import CognitoAuthenticator
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="UNIBOT - AWS Bedrock",
    layout="wide"
)

# --- Project-Specific Imports ---
from config import AWS_REGION_NAME
from data_utils import data_ingestion, update_vector_store
from graph_workflow import compile_workflow

# --- Cache Compiled Graph ---
@st.cache_resource
def load_compiled_graph():
    """Compiles the LangGraph workflow once and caches it."""
    return compile_workflow()

app_graph = load_compiled_graph()

# =============================================================================
# Authentication & Role Management Functions
# =============================================================================

# Get Cognito credentials from environment variables or Streamlit secrets
try:
    COGNITO_POOL_ID = os.getenv("COGNITO_POOL_ID")
    COGNITO_APP_CLIENT_ID = os.getenv("COGNITO_APP_CLIENT_ID")
    COGNITO_APP_CLIENT_SECRET = os.getenv("COGNITO_APP_CLIENT_SECRET")
    
    # --- DEBUGGING: Print loaded variables ---
    print(f"DEBUG: COGNITO_POOL_ID: {COGNITO_POOL_ID}")
    print(f"DEBUG: COGNITO_APP_CLIENT_ID: {COGNITO_APP_CLIENT_ID}")
    print(f"DEBUG: COGNITO_APP_CLIENT_SECRET is loaded: {bool(COGNITO_APP_CLIENT_SECRET)}")
    # --- END DEBUGGING ---

    authenticator = CognitoAuthenticator(
        COGNITO_POOL_ID,
        COGNITO_APP_CLIENT_ID,
        COGNITO_APP_CLIENT_SECRET
    )
except (ValueError, KeyError) as e:
    st.error(f"Missing Cognito configuration: {e}. Please set your secrets.")
    st.stop()


def get_user_role_from_cognito():
    """Checks user's Cognito group and assigns a role."""
    user_groups = authenticator.get_user_cognito_groups()
    if 'UnibotSeniorEmployees' in user_groups:
        return "senior"
    elif 'UnibotRegularEmployees' in user_groups:
        return "regular"
    else:
        # Default role for users not in a specific group
        return "regular"

def login_user():
    """Handles the login process via the Cognito Authenticator."""
    try:
        is_logged_in = authenticator.login()
        # --- DEBUGGING: Print login status ---
        print(f"DEBUG: Login status returned by authenticator.login(): {is_logged_in}")
        # --- END DEBUGGING ---
        if is_logged_in:
            st.session_state.authenticated = True
            st.session_state.user_role = get_user_role_from_cognito()
            st.session_state.username = authenticator.get_username()
        else:
            st.session_state.authenticated = False
    except Exception as e:
        st.error(f"An error occurred during login: {e}")
        st.session_state.authenticated = False

def logout_user():
    """Clears session state and logs the user out via the authenticator."""
    authenticator.logout()
    st.session_state.authenticated = False
    st.session_state.user_role = ""
    st.session_state.username = ""
    st.rerun()

# =============================================================================
# UI Helper Functions
# =============================================================================

def handle_search():
    """Sets a flag when search button is clicked and stores the question."""
    if st.session_state.user_question_input:
        st.session_state.search_clicked = True
        st.session_state.user_question = st.session_state.user_question_input
        st.session_state.user_question_input = ""

# =============================================================================
# Main Application UI
# =============================================================================

def main():
    st.header("UNIBOT - AWS Bedrock + Pinecone")

    # --- Initialize Session State ---
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'last_response' not in st.session_state:
        st.session_state['last_response'] = ""
    if 'last_question' not in st.session_state:
        st.session_state['last_question'] = ""
    if 'search_clicked' not in st.session_state:
        st.session_state.search_clicked = False
    if 'user_role' not in st.session_state:
        st.session_state['user_role'] = "regular" # Set a default role

    # --- Authentication Gate ---
    if not st.session_state.authenticated:
        login_user()
    else:
        # --- Sidebar Controls ---
        with st.sidebar:
            st.title("Controls")
            st.write(f"Logged in user: **{st.session_state.username}**")
            st.write(f"Assigned role: **{st.session_state.user_role}**")
            st.button("Logout", on_click=logout_user, use_container_width=True)
            st.divider()
            st.title("Update Documents")
            st.info("Sync documents from Google Drive to Pinecone.")
            if st.button("Start Data Sync", use_container_width=True):
                with st.spinner("Processing documents..."):
                    st.write("Ingesting senior documents...")
                    senior_folder_id = "YOUR_SENIOR_FOLDER_ID"
                    senior_docs = data_ingestion(senior_folder_id)
                    if senior_docs:
                        update_vector_store(senior_docs, "senior_docs")
                    
                    st.write("Ingesting regular documents...")
                    regular_folder_id = "YOUR_REGULAR_FOLDER_ID"
                    regular_docs = data_ingestion(regular_folder_id)
                    if regular_docs:
                        update_vector_store(regular_docs, "regular_docs")

                    st.success("Document sync complete.")

        # --- Main Chat Interface ---
        st.text_input(
            "Ask a question about your documents:",
            key="user_question_input",
            on_change=handle_search
        )
        st.button("Search", on_click=handle_search)

        if st.session_state.search_clicked:
            st.session_state.search_clicked = False
            with st.spinner("Searching and generating answer..."):
                current_question = st.session_state.user_question
                graph_input = {
                    "query": current_question,
                    "user_role": st.session_state.user_role,
                    "documents": []
                }
                try:
                    result = app_graph.invoke(graph_input)
                    response = result.get("generation", "Error: No generation found in result.")
                    st.session_state['last_response'] = response
                    st.session_state['last_question'] = current_question
                except Exception as e:
                    st.error(f"An error occurred during graph execution: {e}")

        if st.session_state['last_response']:
            with st.chat_message("user"):
                st.write(st.session_state['last_question'])
            with st.chat_message("assistant"):
                st.write(st.session_state['last_response'])

if __name__ == "__main__":
    main()
