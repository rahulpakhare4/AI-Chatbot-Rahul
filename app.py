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
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Initialize models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="YOUR_GROQ_API_KEY")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

# Function to retrieve context
def retrieve_context(query, top_k=1):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]

# Function to process chatbot query
def query_llama3(user_query):
    system_prompt = """
    You are an AI clone of Rahul Pakhare, a consultant with 12+ years of experience.
    Respond naturally and concisely, engaging like a real person.
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

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("Rahul's AI Chatbot")

# Display Chat History
chat_container = st.container()
with chat_container:
    for chat_message in st.session_state.chat_history:
        role = chat_message["role"]
        message_class = "user-message" if role == "user" else "bot-message"
        icon = "👤" if role == "user" else "🤖"
        st.markdown(
            f'<div class="{message_class}">{icon} {chat_message["content"]}</div>',
            unsafe_allow_html=True,
        )

# Custom CSS for styling text input and button
st.markdown(
    """
    <style>
    .chat-container { max-height: 400px; overflow-y: auto; }
    .user-message { text-align: right; background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 5px 0; }
    .bot-message { text-align: left; background-color: #E8E8E8; padding: 10px; border-radius: 10px; margin: 5px 0; }
    .input-container { display: flex; align-items: center; }
    .stTextInput { flex-grow: 1; margin-right: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# User Input Form
with st.form(key="chat_form"):
    col1, col2 = st.columns([4, 1])  # Adjust width for input and button
    with col1:
        user_query = st.text_input("Ask a question:", key="user_input")
    with col2:
        submit_button = st.form_submit_button("Send")

# Process Input
if submit_button and user_query:
    response = query_llama3(user_query)

    # Save chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "bot", "content": response})

    # Rerun to update UI
    st.rerun()
