import os
from flask import Flask, request, jsonify, render_template
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Access environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
neo4j_uri = os.getenv('NEO4J_URI')
neo4j_username = os.getenv('NEO4J_USERNAME')
neo4j_password = os.getenv('NEO4J_PASSWORD')

# Set environment variables for other components
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["NEO4J_URI"] = neo4j_uri
os.environ["NEO4J_USERNAME"] = neo4j_username
os.environ["NEO4J_PASSWORD"] = neo4j_password

# Initialize Neo4j Graph
graph = Neo4jGraph()

# Initialize OpenAI embeddings and ChatOpenAI
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")

# Initialize Neo4jVector
neo4j_vector = Neo4jVector(
    embedding=embeddings,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)

# Initialize RetrievalQAWithSourcesChain
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=neo4j_vector.as_retriever()
)

# Initialize LLMGraphTransformer
llm_transformer = LLMGraphTransformer(llm=llm)

def is_greeting(text):
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    return any(greeting in text.lower() for greeting in greetings)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json['query']
    
    if is_greeting(user_query):
        return jsonify({
            "answer": "Hello! Welcome to the Wimbledon Championships chatbot. How can I assist you today? Feel free to ask any questions about the Wimbledon Championships or the content of the Ticket Holders Handbook."
        })
    
    try:
        response = chain.invoke({"question": user_query}, return_only_outputs=True)
        if response['answer'].strip() == "I don't know.":
            return jsonify({
                "answer": "I'm sorry, I don't have enough information to answer that question. Could you please ask something related to the Wimbledon Championships or the content of the Ticket Holders Handbook?"
            })
        return jsonify(response)
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({
            "answer": "I'm sorry, I encountered an error while processing your question. Could you please try again or rephrase your question?"
        })

@app.route('/ingest_pdf', methods=['POST'])
def ingest_pdf():
    print("Received request to ingest PDF")
    file_path = request.json['file_path']
    print(f"File path: {file_path}")
    
    try:
        print("Starting PDF load")
        loader = PyMuPDFLoader(file_path)
        print("PyMuPDFLoader initialized")
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from PDF")
        
        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        split_documents = text_splitter.split_documents(documents)
        print(f"Split into {len(split_documents)} chunks")
        
        # Convert to graph documents and add to Neo4j
        graph_documents = llm_transformer.convert_to_graph_documents(split_documents)
        graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
        print("Added documents to Neo4j graph")
        
        # Add to vector store
        neo4j_vector.add_documents(split_documents)
        print("Added documents to vector store")
        
        return jsonify({"message": f"Ingested {len(split_documents)} document chunks"})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)