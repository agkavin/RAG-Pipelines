# 📚 Chat with PDF - RAG-powered Document Chat System  

A Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and chat with them. This project uses a combination of FAISS for vector search, Ollama LLM for generating responses, and Gradio for the user interface.  

---

## Overview  
This project enables users to interact with PDF documents by uploading them and asking questions about their contents. The system processes the uploaded documents, extracts relevant text, stores vector embeddings, and retrieves relevant chunks for answering user queries using a large language model (LLM).  

---

## Features  
- **PDF Document Upload**: Easily upload and process PDF files.  
- **Document-based Q&A**: Ask questions and get context-aware responses from the uploaded PDFs.  
- **Retrieval-Augmented Generation (RAG)**: Enhances response quality using document-based retrieval.  
- **Memory-efficient Vector Search**: Uses FAISS for storing and searching embeddings.  
- **Gradio UI**: Simple and user-friendly interface for interaction.  

---

## How It Works  
1. **Upload a PDF File**: The system processes the file, extracts text, and chunks it into smaller sections.  
2. **Text Embedding & Indexing**: The extracted text is embedded using `nomic-embed-text` and stored in a FAISS index.  
3. **User Queries**: When a user asks a question, the system retrieves the most relevant text chunks using Maximum Marginal Relevance (MMR).  
4. **LLM Processing**: The retrieved context is passed to the LLM (`llama3.1`) for generating responses.  
5. **Chatbot Response**: The generated answer is displayed to the user in a conversational format.  

---

## Tech Stack  
- **Python** - Core programming language  
- **LangChain** - Framework for building AI-driven applications  
- **FAISS** - Vector search for efficient retrieval  
- **Ollama** - Local LLM for response generation  
- **PyMuPDF** - PDF document processing  
- **Gradio** - Web-based user interface  

---

## Installation
1. Clone the Repository
   ```sh
   git clone https://github.com/agkavin/RAG-Workshop.git
2. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   cd RAG_for_pdfs
3. Run the application:
   ```sh
   python3 app.py
    
