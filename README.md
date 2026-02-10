# 🚀 RAG Pipelines - Production-Ready Retrieval-Augmented Generation Systems

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive collection of **5 production-ready Retrieval-Augmented Generation (RAG)** implementations, each designed for specific use cases. From analyzing CSV data to chatting with GitHub repositories, these pipelines demonstrate the versatility and power of RAG systems in real-world applications.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Implementations](#-implementations)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Use Cases](#-use-cases)

---

## 🎯 Overview

This repository showcases **five distinct RAG implementations**, each optimized for different data sources and use cases. All implementations leverage **LangChain** for orchestration, various **vector stores** (FAISS, ChromaDB) for efficient retrieval, and powerful **LLMs** (Ollama, Google Gemini) for generation.

### Why RAG?

Retrieval-Augmented Generation combines the best of both worlds:
- **Retrieval**: Fetches relevant context from your data sources
- **Generation**: Uses LLMs to generate accurate, contextual responses
- **Result**: Reduces hallucinations and provides grounded, factual answers

---

## 🔧 Implementations

### 1. 🤖 Agentic RAG for CSV
**Dual-mode intelligent CSV analysis system**

- **Unique Feature**: Hybrid approach with RAG + Agent capabilities
- **Use Case**: Data analysis, business intelligence, CSV exploration
- **Key Capabilities**:
  - General queries via RAG for conceptual understanding
  - Precise data analysis via `@agent` command for calculations
  - Vector-based semantic search over CSV content
  - Interactive Gradio interface

**Tech**: LangChain • Ollama (llama3.1) • ChromaDB • Gradio

[📖 Full Documentation](./Agentic_RAG_for_CSV/README.md)

---

### 2. 💻 Chat with GitHub Repo
**AI-powered codebase analysis and exploration**

- **Unique Feature**: Clone and analyze entire GitHub repositories
- **Use Case**: Code review, documentation generation, onboarding
- **Key Capabilities**:
  - Automatic repository cloning and processing
  - Multi-language support (.py, .js, .ts, .md, .ipynb)
  - MMR (Maximum Marginal Relevance) search for diverse results
  - Code snippet extraction with context

**Tech**: LangChain • Ollama (llama3.1 + nomic-embed-text) • FAISS • GitPython • Gradio

[📖 Full Documentation](./Chat_with_Github_Repo/README.md)

---

### 3. 📄 RAG for PDFs
**Document intelligence and Q&A system**

- **Unique Feature**: Local-first PDF processing with Ollama
- **Use Case**: Research, document analysis, knowledge extraction
- **Key Capabilities**:
  - Multi-PDF support with persistent storage
  - Chunking with overlap for context preservation
  - MMR-based retrieval for comprehensive answers
  - Privacy-focused (all processing local)

**Tech**: LangChain • Ollama (llama3.1 + nomic-embed-text) • FAISS • PyMuPDF • Gradio

[📖 Full Documentation](./RAG_for_pdfs/README.md)

---

### 4. ✨ RAG with Gemini
**Cloud-powered PDF chat with Google's Gemini AI**

- **Unique Feature**: Leverages Google's Gemini-1.5-Flash for fast responses
- **Use Case**: Enterprise document processing, scalable Q&A
- **Key Capabilities**:
  - Google Gemini AI integration
  - ChromaDB for persistent vector storage
  - Optimized for speed with Gemini-1.5-Flash
  - API-based, no local GPU required

**Tech**: LangChain • Google Gemini AI • ChromaDB • PyPDF • Gradio

[📖 Full Documentation](./RAG_with_Gemini/README.md)

---

### 5. 🌐 RAG with Web Search
**Real-time web search and synthesis**

- **Unique Feature**: Live web scraping and knowledge synthesis
- **Use Case**: Current events, research, fact-checking
- **Key Capabilities**:
  - Real-time DuckDuckGo web search
  - Automatic web content extraction and cleaning
  - Conversational memory for follow-up questions
  - Source citation in responses

**Tech**: LangChain • Ollama (llama3.1) • FAISS • DuckDuckGo Search • BeautifulSoup • Gradio

[📖 Full Documentation](./RAG_with_Web_Search/README.md)

---

## 🏗️ Architecture

All implementations follow a consistent RAG architecture:

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Ingestion │ ◄── PDF / CSV / GitHub / Web
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Splitting  │ (RecursiveCharacterTextSplitter)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │ (Ollama / Gemini)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │ (FAISS / ChromaDB)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Retrieval    │ (Similarity / MMR Search)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM + Prompt   │ (Context + Query)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Response     │
└─────────────────┘
```

---

## 🛠️ Tech Stack

### Core Frameworks
- **LangChain** - RAG orchestration and chain management
- **LangChain Community** - Document loaders and integrations
- **LangChain Experimental** - Advanced agent capabilities

### LLMs & Embeddings
- **Ollama** - Local LLM inference (llama3.1, nomic-embed-text)
- **Google Gemini AI** - Cloud-based LLM (gemini-1.5-flash)

### Vector Stores
- **FAISS** - Facebook AI Similarity Search (fast, in-memory)
- **ChromaDB** - Persistent vector database

### Document Processing
- **PyMuPDF** - PDF text extraction
- **PyPDF** - Alternative PDF processing
- **GitPython** - GitHub repository cloning
- **BeautifulSoup4** - Web scraping
- **html2text** - HTML to markdown conversion

### Search & Retrieval
- **DuckDuckGo Search** - Web search API

### UI Framework
- **Gradio** - Interactive web interfaces

---

## 📦 Installation

### Prerequisites

- **Python 3.10+**
- **Ollama** (for local implementations) - [Install Ollama](https://ollama.ai)
- **Google Gemini API Key** (for Gemini implementation) - [Get API Key](https://makersuite.google.com/app/apikey)

### Step 1: Clone the Repository

```bash
git clone https://github.com/agkavin/RAG-Pipelines.git
cd RAG-Pipelines
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Setup Ollama (for local implementations)

```bash
# Install Ollama from https://ollama.ai

# Pull required models
ollama pull llama3.1
ollama pull nomic-embed-text
```

### Step 4: Configure Environment Variables (for Gemini)

For the RAG with Gemini implementation, create a `.env` file:

```bash
cd RAG_with_Gemini
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

---

## 🚀 Quick Start

### Running Individual Implementations

Each implementation can be run independently:

```bash
# Agentic RAG for CSV
cd Agentic_RAG_for_CSV
python app.py

# Chat with GitHub Repo
cd Chat_with_Github_Repo
python app.py

# RAG for PDFs
cd RAG_for_pdfs
python app.py

# RAG with Gemini
cd RAG_with_Gemini
python app.py

# RAG with Web Search
cd RAG_with_Web_Search
python app.py
```

All applications will launch a **Gradio interface** accessible at `http://localhost:7860`

---

## 📁 Project Structure

```
RAG-Pipelines/
│
├── Agentic_RAG_for_CSV/
│   ├── app.py                 # Dual-mode CSV analysis
│   └── README.md
│
├── Chat_with_Github_Repo/
│   ├── app.py                 # GitHub repository analyzer
│   └── README.md
│
├── RAG_for_pdfs/
│   ├── app.py                 # Local PDF chat system
│   └── README.md
│
├── RAG_with_Gemini/
│   ├── app.py                 # Gemini-powered PDF chat
│   ├── .env                   # API key configuration
│   └── README.md
│
├── RAG_with_Web_Search/
│   ├── app.py                 # Real-time web search RAG
│   └── README.md
│
├── sample_files/
│   ├── attention-is-all-you-need-Paper.pdf
│   └── extracted_jobs.csv
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 💡 Use Cases

### Business & Analytics
- **CSV Analysis**: Query sales data, customer records, financial reports
- **Document Intelligence**: Extract insights from contracts, reports, research papers

### Software Development
- **Code Review**: Analyze codebases for patterns, dependencies, documentation
- **Onboarding**: Help new developers understand large repositories

### Research & Education
- **Literature Review**: Chat with academic papers and research documents
- **Current Events**: Get synthesized answers from real-time web searches

### Enterprise
- **Knowledge Management**: Build internal Q&A systems over company documents
- **Customer Support**: Create intelligent chatbots with domain-specific knowledge

---

## 🎓 Key Concepts Demonstrated

### 1. **Text Splitting Strategies**
- Recursive character splitting with overlap
- Chunk size optimization (1000 chars)
- Context preservation across chunks

### 2. **Retrieval Methods**
- **Similarity Search**: Find most similar documents
- **MMR (Maximum Marginal Relevance)**: Balance relevance and diversity
- **Configurable k**: Control number of retrieved documents

### 3. **Prompt Engineering**
- System prompts for role definition
- Context injection patterns
- Few-shot examples

### 4. **Agent Patterns**
- Agentic CSV analysis with tool use
- Conversational retrieval chains
- Memory management

### 5. **Vector Store Selection**
- **FAISS**: Fast, in-memory, great for prototyping
- **ChromaDB**: Persistent, production-ready, easy to use

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Ideas for Contributions
- Add new RAG implementations (e.g., SQL databases, Excel files)
- Improve retrieval strategies (hybrid search, reranking)
- Add evaluation metrics and benchmarks
- Enhance UI/UX with better Gradio components
- Add Docker support for easy deployment
- Implement streaming responses

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
