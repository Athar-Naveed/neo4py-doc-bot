from PIL import Image
import base64
from io import BytesIO
import faiss
import os
import glob
from dotenv import load_dotenv
import pymupdf4llm
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.gemini import Gemini
from functools import lru_cache

# Load environment variables once
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set constants
EMBEDDING_DIMENSION = 768
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "models/gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"

# System prompt
SYSTEM_PROMPT = """
You are a highly knowledgeable and helpful assistant specialized in Neo4py and related topics. You provide clear, concise, and detailed answers about Neo4py, focusing on accuracy and professionalism.

Casual Interaction: Respond naturally to casual greetings and small talk (e.g., "Hello," "How are you?") without diving into technical details about Neo4py unless specifically asked.

Technical Queries: When the user asks about Neo4py, analyze the prompt thoroughly and respond with in-depth, accurate, and structured information in Markdown format. Your answers should be:

- Detailed: Provide explanations, code snippets, and best practices where relevant.
- Concise and Clear: Avoid unnecessary information while covering the topic comprehensively.
- Helpful: Offer step-by-step guidance if needed, focusing exclusively on Neo4py and avoiding unrelated topics.
- Unknown Information: If you do not know the answer, admit it honestly and avoid speculation.

Tone: Maintain a professional yet approachable tone, ensuring that responses are tailored to the context of the user's queries.
"""

# Helper function to convert image to base64 (cached to avoid repeated conversions)
@lru_cache(maxsize=1)
def image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

# Initialize session state
chat_history = []
index = None
pdfs_loaded = False

# Read PDF with error handling
def read_pdf(file_path):
    try:
        return pymupdf4llm.to_markdown(file_path)
    except Exception as e:
        raise Exception(f"Error reading {file_path}: {str(e)}")

# Load PDFs and create vector index
def load_pdfs_from_folder():
    global index, pdfs_loaded
    
    # Get all PDF files from the data folder
    pdf_files = glob.glob("data/*.pdf")
    
    if not pdf_files:
        raise Exception("No PDF files found in the data folder!")
    
    # Read all PDF files and create document objects
    documents = []
    for file_path in pdf_files:
        text = read_pdf(file_path)
        if text:
            documents.append(Document(text=text, metadata={"filename": os.path.basename(file_path)}))
    
    if not documents:
        raise Exception("Could not extract text from any PDFs in the data folder!")
    
    # Create vector store
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    # Initialize embedding model
    embed_model = GeminiEmbedding(model_name=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)
    
    # Configure global settings
    Settings.embed_model = embed_model
    
    # Initialize default LLM - try Groq first
    try:
        Settings.llm = Groq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
    except Exception as e:
        print(f"Failed to initialize Groq: {str(e)}. Falling back to Gemini.")
        try:
            Settings.llm = Gemini(model=GEMINI_MODEL, api_key=GOOGLE_API_KEY)
        except Exception as e2:
            raise Exception(f"Failed to initialize any LLM. Please check your API keys. Error: {str(e2)}")
    
    # Create and store index
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
    pdfs_loaded = True
    
    return len(documents)

# Get response with Groq primary and Gemini fallback
def get_bot_response(user_input):
    global index, chat_history
    
    if index is None:
        return "PDFs have not been loaded yet! Please check the data folder."
    
    # Build context from recent messages (last 5 interactions)
    context_str = ""
    if chat_history:
        recent = chat_history[-10:]  # 5 interactions = 10 messages
        for i in range(0, len(recent), 2):
            if i+1 < len(recent):
                context_str += f"### Previous Interaction:\n**User**: {recent[i]['content']}\n**Assistant**: {recent[i+1]['content']}\n\n"
    
    # Combine system prompt, context, and current question
    full_query = f"{SYSTEM_PROMPT}\n\n{context_str}\n### New Question:\n{user_input}"
    
    # Try Groq first, fall back to Gemini
    try:
        Settings.llm = Groq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
        query_engine = index.as_query_engine(response_mode="compact")
        return str(query_engine.query(full_query))
    except Exception as e:
        print(f"Groq query failed: {str(e)}. Falling back to Gemini.")
        try:
            Settings.llm = Gemini(model=GEMINI_MODEL, api_key=GOOGLE_API_KEY)
            query_engine = index.as_query_engine(response_mode="compact")
            return str(query_engine.query(full_query))
        except Exception as e2:
            return f"Both Groq and Gemini failed. Error: {str(e2)}"

# Clear chat history
def clear_chat_history():
    global chat_history
    chat_history = []

# Check if PDFs are loaded
def is_pdfs_loaded():
    return pdfs_loaded

# Get chat history
def get_chat_history():
    return chat_history