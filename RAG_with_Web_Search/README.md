# Web Search RAG

A **Retrieval-Augmented Generation (RAG)** system that performs real-time **web searches**, extracts and indexes web content into a **vector store (FAISS)**, and generates AI-driven responses using **Ollama's LLM**.


## How It Works
1. **User Query:** A user submits a question.
2. **Web Search Execution:** The system fetches relevant results using DuckDuckGo.
3. **Web Content Extraction:** Extracts and cleans webpage content.
4. **Vector Store Update:** The extracted content is chunked and stored in FAISS.
5. **AI Response Generation:** The LLM generates an answer based on retrieved knowledge.
6. **Conversational Memory:** Previous exchanges are considered for better responses.
   
## Features

- **Real-Time Web Search:** Fetches search results using DuckDuckGo.
- **Web Scraping & Cleaning:** Extracts webpage content with BeautifulSoup and `html2text`.
- **FAISS Vector Storage:** Indexes processed web content for efficient retrieval.
- **RAG-Based Answer Generation:** Uses `LangChain` to synthesize information from retrieved sources.
- **Conversational Memory:** Maintains chat history for contextual responses.
- **Gradio Chat UI:** Interactive interface for querying and response visualization.

## Technology Stack
This project leverages the following technologies:

- **LangChain** – Framework for building applications with LLMs and retrieval-augmented generation.
- **DuckDuckGo Search API** – Fetches real-time web search results.
- **BeautifulSoup & html2text** – Extracts and cleans webpage content.
- **FAISS (Facebook AI Similarity Search)** – Efficient vector-based search and retrieval.
- **Ollama** – Local LLM for embedding generation and response synthesis.
- **Gradio** – Provides an interactive web UI for querying.
- **Python** – Core programming language for the implementation.


## Installation

1. Clone the Repository
```bash
git clone https://github.com/agkavin/RAG-Workshop.git
```

2. Install Dependencies
```bash
pip install -r requirements.txt
cd RAG_with_Web_Search
```


## Example Usage

You can use the interactive Gradio UI to ask questions. Example queries:

- "What are the latest developments in AI?"
- "What is DeepSeek?"
- "What are the main features of Python 3.12?"

