import asyncio
import os
import streamlit as st
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import numpy as np

# Initialize models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

def load_pdf(file):
    """Extract text from a PDF file."""
    reader = PdfReader(file)
    return "".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text, chunk_size=600):
    """Split text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    return splitter.split_text(text)

def retrieve_context(query, top_k=1):
    """Retrieve relevant documents from ChromaDB."""
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]

def query_llama3(user_query):
    """Handles user queries while retrieving context and past chat history."""
    system_prompt = """
    You are an AI clone of Rahul Pakhare, a consultant with 12+ years of experience.
    Respond naturally and concisely, engaging like a real person.
    Do not provide false information.
    If you don’t know the answer, simply respond:
    "Apologies, I am an AI clone of Rahul and don't have all the details. Stay tuned for my latest version!" No additional sentences are required.
    """
    past_chat_history = memory.load_memory_variables({}).get("chat_history", [])[-8:]
    retrieved_context = retrieve_context(user_query)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Past Chat: {past_chat_history}\nDB Context: {retrieved_context}\n\nQuestion: {user_query}")
    ]
    response = chat.invoke(messages)
    memory.save_context({"input": user_query}, {"output": response.content})
    return response.content

#UI starts from here
# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Streamlit UI
st.title("Rahul's AI Chatbot")

#Hide sidebar for public access starts from here

# Define user authentication
user_authenticated = False  # Change this based on authentication logic

if not user_authenticated:
    # Hide the sidebar completely
    hide_sidebar_style = """
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
    """
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)
else:
    st.sidebar.header("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        pdf_text = load_pdf(uploaded_file)
        chunks = chunk_text(pdf_text)
        embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]
        collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings
        )
        st.sidebar.success("You are ready to use this chatbot now!")
#Hide sidebar for public access starts ends here

# Apply custom CSS to style user and bot messages
st.markdown(
    """
    <style>
    .chat-container { max-height: 400px; overflow-y: auto; }
    .user-message { text-align: right; background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 5px 0; }
    .bot-message { text-align: left; background-color: #E8E8E8; padding: 10px; border-radius: 10px; margin: 5px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display Chat History
#st.subheader("Chat History")
chat_container = st.container()
with chat_container:
    for chat_message in st.session_state.chat_history:
        role = chat_message["role"]
        if role == "user":
            st.markdown(
                f'<div class="user-message">👤 {chat_message["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bot-message">🤖 {chat_message["content"]}</div>',
                unsafe_allow_html=True,
            )

# Function to process user input
def process_input():
    user_query = st.session_state.user_input.strip()
    if user_query:
        response = query_llama3(user_query)  # Call your chatbot function

        # Save chat history in session state
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "bot", "content": response})

        # Clear input box (reset value)
        st.session_state.user_input = ""

# Move Input Box and Button Below Chat
st.markdown("---")  # Adds a separator
st.text_input("Ask a question:", key="user_input", on_change=process_input)

