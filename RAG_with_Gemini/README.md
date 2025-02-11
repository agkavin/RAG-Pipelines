# 📚 Chat with Gemini - RAG-powered PDF Chatbot  

A Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and chat with them using Google's Gemini AI.  

---

## Overview  
This project allows users to upload PDF documents and interact with them by asking questions. It extracts relevant text, performs vector-based search to retrieve the most pertinent information, and generates AI-powered responses using Gemini, leveraging computational resources via the API.

---

## Features  
- **Upload & Chat with PDFs**: Users can upload PDFs and interact with them conversationally.  
- **Google Gemini AI**: Uses Gemini-1.5-Flash for fast and intelligent responses.  
- **Retrieval-Augmented Generation (RAG)**: Ensures responses are based on document content.  
- **Vector Search with ChromaDB**: Efficient document retrieval using embeddings.  
- **Gradio UI**: Simple web interface for seamless interaction.  

---

## Tech Stack  
- **Python** - Core programming language  
- **Gradio** - Web-based UI  
- **Google Gemini AI** - LLM for generating responses  
- **ChromaDB** - Vector storage for efficient retrieval  
- **LangChain** - Framework for building AI-driven applications  
- **PyMuPDF** - PDF document processing  
- **dotenv** - For environment variable management  

---

## Installation & Setup  

Prerequisites :
- Python 3.10+  
- A **Google Gemini API Key** (store it in `.env`)  
 

1. Clone the Repository 
   ```sh
   git clone https://github.com/agkavin/RAG-Workshop.git

2. Install the dependencies: 
   ```sh
   pip install -r requirements.txt
   cd RAG-with-Gemini
3. Run the application:
   ```sh
   python3 app.py
