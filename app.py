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

import streamlit as st
import openai

# ✅ Set page config
st.set_page_config(page_title="Rahul's AI Clone Chatbot", layout="wide")


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

def evaluate_response(user_query, generated_response, context):
    """Evaluate response using semantic similarity."""
    response_embedding = semantic_model.encode(generated_response, convert_to_tensor=True)
    context_embedding = semantic_model.encode(context, convert_to_tensor=True)
    return util.pytorch_cos_sim(response_embedding, context_embedding)[0][0].item()
    
def query_llama3(user_query):
    """Handles user queries while retrieving context and past chat history."""
    system_prompt = """
    You are an AI clone of Rahul Pakhare, a consultant with 12+ years of experience.
    Respond naturally and concisely, engaging like a real person.
    Do not provide false information.
    If you don’t know the answer, simply respond:
    "Apologies, I am an AI clone of Rahul and don't have all the details. Stay tuned for my latest version!" No additional sentences are required.
    You can discuss general topics, but avoid speculative or misleading responses.
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
    
# Streamlit UI
st.title("Rahul's AI Chatbot")

#Sidebar Hide code for public
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
#Sidebar Code hide ends here

#CHAT UI


# ✅ Set up Groq API key (Replace with actual key)
GROQ_API_KEY = "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"
openai.api_key = GROQ_API_KEY

# ✅ Updated AI Query Function
def query_llama3(user_input):
    """Fetch response from Groq Llama 3 API with better error handling"""
    try:
        response = openai.ChatCompletion.create(
            model="llama-3-8b",
            messages=[{"role": "user", "content": user_input}],
            max_tokens=150,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"]
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ✅ UI for Chatbot
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ✅ Chat Display
for message in st.session_state["messages"]:
    role, content = message["role"], message["content"]
    if role == "user":
        st.markdown(f"<div style='text-align: right; background-color: #DCF8C6; padding: 10px; margin: 5px; border-radius: 10px; width: 60%; float: right;'><b>👤 You:</b> {content}</div><div style='clear: both;'></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; background-color: #E0E0E0; padding: 10px; margin: 5px; border-radius: 10px; width: 60%;'><b>🤖 AI:</b> {content}</div><div style='clear: both;'></div>", unsafe_allow_html=True)

# ✅ Input Box
user_query = st.text_input("Type a message...", key="input", placeholder="Ask me anything...")

# ✅ Send Button
if st.button("Send"):
    if user_query:
        # Store user message
        st.session_state["messages"].append({"role": "user", "content": user_query})
        
        # Get AI response
        ai_response = query_llama3(user_query)
        
        # Store AI message
        st.session_state["messages"].append({"role": "assistant", "content": ai_response})
        
        # Refresh to show messages
        st.experimental_rerun()
