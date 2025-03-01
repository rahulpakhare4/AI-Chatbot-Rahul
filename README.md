# Rahul's Personal Chatbot 🤖📄

Rahul's AI Clone Chatbot – A Streamlit-based chatbot that allows users to upload PDFs, process them using Llama 3 (Groq), and chat with Rahul's AI clone through the chatbox. The chatbot retrieves contextual information from ChromaDB for more accurate responses.

I have already uploaded a PDF of my bio and deployed my AI clone on Streamlit. You can check it out at the link below. (Note: I will hide the Groq API key after a few days of pushing the project.)

**Please find Streamlit link here: ** https://ai-chatbot-rahul.streamlit.app/

## Features
✅ Upload PDF and extract text  
✅ Store embeddings using ChromaDB  
✅ Retrieve relevant context before answering  
✅ Use Llama 3 (Groq) for responses  
✅ Evaluate responses using semantic similarity  
✅ Memory-based chat history  
✅ Streamlit UI for easy interaction  

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py

-----------------------------------------------------------------------

## **Push Future Updates**
Whenever you make changes, run:

```bash
git add .
git commit -m "Updated features"
git push origin main
