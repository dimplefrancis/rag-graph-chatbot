import streamlit as st
import os
#from dotenv import load_dotenv
#from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_community.vectorstores.neo4j_vector import Neo4jVector
#from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.document_loaders import PyMuPDFLoader

# Load environment variables
#load_dotenv()

# Set page config
st.set_page_config(
    page_title="Wimbledon 2024 Chatbot",
    page_icon="🎾",
    layout="wide"
)

# Function to get local image path
def get_image_path(image_name):
    return os.path.join("Images", image_name)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f0f0;
    }
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
        color: #4B0082;
        text-align: center;
    }
    .custom-button, .stButton > button {
        display: inline-block;
        padding: 10px 24px;
        font-size: 20px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        outline: none;
        color: #fff !important;
        background-color: #4B0082 !important;
        border: none !important;
        border-radius: 5px !important;
        width: 100%;
    }
    .custom-button:hover, .stButton > button:hover {
        background-color: #3a006c !important;
    }
    .custom-button:active, .stButton > button:active {
        background-color: #2d0052 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

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
        password=os.getenv("NEO4J_PASSWORD"),
    )
    llm = ChatOpenAI(temperature=0.8, model_name=model, openai_api_key=os.getenv("OPENAI_API"))
    return graph, neo4j_vector, llm

# Function to ingest PDF
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    split_documents = text_splitter.split_documents(documents)
    return split_documents

def ingest_pdf(neo4j_vector, split_documents):
    neo4j_vector.add_documents(split_documents)
    return len(split_documents)

# Function to check if input is a greeting
def is_greeting(text):
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    return any(greeting in text.lower() for greeting in greetings)

# Function to generate response
def generate_response(prompt, neo4j_vector, llm):
    if is_greeting(prompt):
        return "Hello! Welcome to the Wimbledon 2024 chatbot. How can I assist you with information about the tournament based on the Ticket Holders Handbook?"
    
    try:
        docs = neo4j_vector.similarity_search(prompt, k=3)
        if not docs:
            st.warning("No relevant documents found in the database.")
            return "I couldn't find any specific information about that in the Ticket Holders Handbook. Is there something else about Wimbledon 2024 you'd like to know?"

        context = "\n".join([doc.page_content for doc in docs])
        response_prompt = f"""Based solely on the following context from the Wimbledon 2024 Ticket Holders Handbook, answer the question. If the answer is not in the context, politely say that you don't have that specific information in the handbook and ask if there's anything else you can help with regarding Wimbledon 2024.

        Context: {context}

        Question: {prompt}

        Answer:"""
        
        response = llm.invoke(response_prompt)
        return "Based on the information in the Wimbledon 2024 Ticket Holders Handbook: " + response.content

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Details of the error have been logged for the developers to investigate.")
        print(f"Error in generate_response: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your request. Please try asking your question again, or ask about something else related to Wimbledon 2024."
    
# Landing Page
def show_landing_page():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{get_image_path('background.jpg')}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<h1 class='big-font'>Wimbledon 2024</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("WimbleChat", key="wimblechat-button"):
            st.session_state.page = 'chat'
            st.rerun()
    with col2:
        st.markdown(
            """
            <a href="https://www.wimbledon.com" target="_blank" style="text-decoration:none;">
                <button class="custom-button">
                    Official Wimbledon Site
                </button>
            </a>
            """, 
            unsafe_allow_html=True
        )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(get_image_path("aerial_view.jpg"), use_column_width=True)
    with col2:
        st.image(get_image_path("trophy.jpg"), use_column_width=True)
    with col3:
        st.image(get_image_path("centre_court.jpg"), use_column_width=True)

# Chatbot Page
def show_chat_page():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: none;
            background-color: #f0f0f0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.image(get_image_path("wimbledon_logo.jpg"), width=150)
    with col2:
        st.title("WIMBLECHAT")
    with col3:
        if st.button("Home", key="home-button"):
            st.session_state.page = 'landing'
            st.rerun()

    model_name = st.sidebar.selectbox(
        "Choose a model",
        ("gpt-3.5-turbo", "gpt-4-turbo-preview")
    )
    st.sidebar.write(f"Currently using: {model_name}")

    st.sidebar.info(
        "This chatbot provides information based solely on the Wimbledon 2024 Ticket Holders Handbook. "
        "For the most up-to-date and comprehensive information, please visit the official Wimbledon website."
    )

    graph, neo4j_vector, llm = initialize_resources(model_name)

    pdf_path = "data/Ticket Holders Handbook 2024.pdf"
    if os.path.exists(pdf_path):
        split_documents = load_and_split_pdf(pdf_path)
        num_chunks = ingest_pdf(neo4j_vector, split_documents)
        st.sidebar.write(f"PDF ingested: {num_chunks} chunks")
    else:
        st.sidebar.error("PDF file not found. Please check the file path.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about Wimbledon 2024?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt, neo4j_vector, llm)

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Main app logic
def main():
    if st.session_state.page == 'landing':
        show_landing_page()
    elif st.session_state.page == 'chat':
        show_chat_page()

if __name__ == "__main__":
    main()