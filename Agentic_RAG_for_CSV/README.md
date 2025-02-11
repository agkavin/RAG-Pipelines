# **Agentic RAG for CSV**  
🔍 A Retrieval-Augmented Generation (RAG)-powered chatbot that allows users to interact with CSV data using natural language queries and an intelligent agent.  

## **Table of Contents**  
1. [Overview](#overview)  
2. [Features](#features)  
3. [Tech Stack](#tech-stack)  
4. [Installation](#installation)  

---

## **Overview**  
This project integrates **Retrieval-Augmented Generation (RAG)** with an **Agentic approach** to enable users to analyze CSV files effortlessly. It utilizes **LangChain**, **Ollama embeddings**, and **Gradio** to create an interactive interface for CSV-based querying.  

### **How It Works**  
- Users upload a CSV file.  
- The system processes the data, creating a **vector store** for retrieval-based answers.  
- Users can interact in two modes:  
  - **General Queries**: RAG-powered responses.  
  - **Agentic Analysis (@agent command)**: Direct CSV analysis using an agent.  

## **Features**  
✅ **Uploads and Processes CSV Files**  
✅ **Supports Retrieval-Augmented Generation (RAG)** for intelligent responses  
✅ **Agent-Based CSV Analysis** using the `@agent` command  
✅ **Vector Store with ChromaDB for Efficient Retrieval**  
✅ **User-Friendly Gradio Interface**  

## **Tech Stack**  
- **Programming Language**: Python  
- **Frameworks & Libraries**:  
  - [LangChain](https://github.com/hwchase17/langchain)  
  - [Ollama Embeddings](https://ollama.ai)  
  - [Gradio](https://www.gradio.app/)  
  - [ChromaDB](https://www.trychroma.com/)  

## **Installation**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/agentic-rag-csv.git
   cd agentic-rag-csv
2. Installing the dependancies:
   ```bash
   pip install -r requirements.txt
3. Run the application:
   ```bash
   python app.py
