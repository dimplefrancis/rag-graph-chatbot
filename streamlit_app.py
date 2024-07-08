import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Explicitly set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API")  # Note the change here
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

# Check if all required keys are loaded
required_keys = ["OPENAI_API", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
missing_keys = [key for key in required_keys if not os.getenv(key)]

if missing_keys:
    st.error(f"The following environment variables are missing: {', '.join(missing_keys)}. Please check your .env file.")
    st.stop()

# Set page title
st.set_page_config(page_title="Wimbledon 2024 Chatbot")

# Title
st.title("Wimbledon 2024 Chatbot")

# Debug logging
st.sidebar.write("Environment variables loaded:")
st.sidebar.write(f"OPENAI_API: {os.getenv('OPENAI_API')[:5]}...{os.getenv('OPENAI_API')[-5:]}")
st.sidebar.write(f"NEO4J_URI: {os.getenv('NEO4J_URI')}")
st.sidebar.write(f"NEO4J_USERNAME: {os.getenv('NEO4J_USERNAME')}")
st.sidebar.write(f"NEO4J_PASSWORD: {'*' * len(os.getenv('NEO4J_PASSWORD', ''))}")

# Model selection
model_name = st.sidebar.selectbox(
    "Choose a model",
    ("gpt-3.5-turbo", "gpt-4-turbo-preview")
)

# Display selected model
st.sidebar.write(f"Currently using: {model_name}")

# Initialize Neo4j Graph and vector store
@st.cache_resource
def initialize_resources(model):
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API"))
    neo4j_vector = Neo4jVector(
        embedding=embeddings,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    llm = ChatOpenAI(temperature=0, model_name=model, openai_api_key=os.getenv("OPENAI_API"))
    return graph, neo4j_vector, llm

graph, neo4j_vector, llm = initialize_resources(model_name)

# Function to ingest PDF
def ingest_pdf(file):
    with st.spinner("Processing PDF..."):
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(file.getbuffer())
        
        # Load and process the PDF
        loader = PyMuPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        split_documents = text_splitter.split_documents(documents)
        
        # Add to vector store
        neo4j_vector.add_documents(split_documents)
        
        # Remove temporary file
        os.remove("temp.pdf")
        
        st.success(f"Ingested {len(split_documents)} document chunks")

# PDF Ingestion Section
st.header("Update Knowledge Base")
pdf_file = st.file_uploader("Upload PDF file", type="pdf")
if pdf_file:
    ingest_pdf(pdf_file)

# Chat Interface
st.header("Chat with Wimbledon Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about Wimbledon?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from chatbot
    docs = neo4j_vector.similarity_search(prompt, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    response_prompt = f"Based on the following context, answer the question about Wimbledon. If the answer is not in the context, say 'I don't have enough information to answer that question.'\n\nContext: {context}\n\nQuestion: {prompt}\n\nAnswer:"
    response = llm.invoke(response_prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response.content)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})