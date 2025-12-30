# graph_workflow.py

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_community.llms.bedrock import Bedrock
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate

# Import clients and configs from config.py
from config import bedrock_client, bedrock_embeddings, INDEX_NAME

# --- LLM and Prompt Definition ---

def get_llama2_llm():
    return Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock_client, model_kwargs={'max_gen_len': 512})

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

# --- Graph State Definition ---

class GraphState(TypedDict):
    query: str
    user_role: str
    documents: List[Document]
    generation: str

# --- Graph Nodes ---

def retrieve_regular_docs(state: GraphState) -> GraphState:
    """Retrieves documents from the regular namespace."""
    query = state["query"]
    vectorstore_regular = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=bedrock_embeddings, namespace="regular_docs"
    )
    # Increased k from 5 to 10 for better context retrieval
    retriever_regular = vectorstore_regular.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    regular_docs = retriever_regular.invoke(query)
    return {"documents": regular_docs}

def retrieve_senior_docs(state: GraphState) -> GraphState:
    """Retrieves documents from the senior and regular namespaces."""
    query = state["query"]
    
    # Retrieve from both senior and regular namespaces for a comprehensive search
    vectorstore_senior = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=bedrock_embeddings, namespace="senior_docs"
    )
    vectorstore_regular = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=bedrock_embeddings, namespace="regular_docs"
    )

    # Increased k from 5 to 10 for better context retrieval
    senior_retriever = vectorstore_senior.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    regular_retriever = vectorstore_regular.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    senior_docs = senior_retriever.invoke(query)
    regular_docs = regular_retriever.invoke(query)

    # Combine and deduplicate documents from both retrievers
    all_docs = senior_docs + regular_docs
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    return {"documents": list(unique_docs)}

def generate_answer(state: GraphState) -> GraphState:
    """Generates an answer using the LLM based on the retrieved documents."""
    query = state["query"]
    documents = state["documents"]
    
    # Create context string from retrieved documents
    context_str = "\n\n".join([doc.page_content for doc in documents])
    
    # Format the prompt
    prompt_input = {"context": context_str, "question": query}
    formatted_prompt = PROMPT.format(**prompt_input)
    
    # Invoke LLM
    llm = get_llama2_llm()
    generation = llm.invoke(formatted_prompt)
    return {"generation": generation}

# --- Conditional Edge Logic ---

def decide_next_step(state: GraphState) -> str:
    """Determines whether to fetch senior documents based on user role."""
    if state["user_role"] == "senior":
        return "retrieve_senior"
    else:
        # The 'regular' role is the default and only other role
        return "retrieve_regular"

# --- Graph Assembly ---

def compile_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve_regular", retrieve_regular_docs)
    workflow.add_node("retrieve_senior", retrieve_senior_docs)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("retrieve_regular")
    workflow.add_conditional_edges(
        "retrieve_regular", decide_next_step,
        {"retrieve_senior": "retrieve_senior", "generate": "generate"}
    )
    workflow.add_edge("retrieve_senior", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()