# **Chat with Github Repo**  
This project implements a **Retrieval-Augmented Generation (RAG) system** to analyze GitHub repositories using **LangChain, FAISS, and Ollama**. It enables users to query repositories using natural language and retrieve relevant code snippets and insights.

## **Table of Contents**  
1. [Overview](#overview)  
2. [Features](#features)  
3. [Tech Stack](#tech-stack)  
4. [Installation](#installation)  
  
---

## **Overview**  
The GitHub Repository Analysis System is a Retrieval-Augmented Generation (RAG) system that allows users to analyze GitHub repositories using an AI-powered chatbot. This system processes code repositories, indexes relevant code snippets, and enables interactive question-answering based on the repository's contents.  

### **How It Works**  
- **Clone Repository**→ Uses gitpython to download the repo.
- **Extract & Split Text** → Extracts relevant code and documentation 
- **Vector Storage** → Uses FAISS to store embeddings.
- **AI Chat Response** → Queries the database and generates answers.
## **Features**   
✅ **Automatic Repository Cloning**  
✅ **Vector-based Search** Uses FAISS for efficient document retrieval.  

✅ **Language Model Integration**  Utilizes Ollama's Llama 3.1 for answering queries.  

✅ **Code Understanding** Parses and indexes .py, .md, .js, .ts, and related files.  


✅ **Gradio Interface** Provides an interactive web-based UI for querying repositories.

## **Tech Stack**  
- **Programming Language**: Python  
- **Frameworks & Libraries**:  
  - [LangChain](https://github.com/hwchase17/langchain)  
  - [Ollama Embeddings](https://ollama.ai)  
  - [Gradio](https://www.gradio.app/)  
  - [FAISS](https://ai.meta.com/tools/faiss/)  

## **Installation**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/agkavin/RAG-Workshop.git
2. Install dependancies:
   ```bash
   pip install -r requirements.txt
   cd Chat_with_Github_Repo
3. Run the application:
   ```bash
   python3 app.py
