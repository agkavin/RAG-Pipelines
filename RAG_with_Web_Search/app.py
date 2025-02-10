from typing import List, Dict
import gradio as gr
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
import html2text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
import os
import re

class WebSearchRAG:
    def __init__(self, vector_store_dir="./web_search_vectors"):
        self.vector_store_dir = vector_store_dir
        os.makedirs(vector_store_dir, exist_ok=True)
        
        # Initialize Ollama components
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        self.llm = ChatOllama(
            model="llama3.1",
            base_url="http://localhost:11434",
            temperature=0.7
        )
        
        # Initialize vector store
        self.vector_store = FAISS.from_texts(
            ["Initial document"], 
            self.embeddings
        )
        
        # Create enhanced prompt for web search synthesis
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant that helps users by synthesizing information from multiple web sources.
            Analyze the provided context from various web pages and provide a comprehensive, well-organized answer.
            Always cite your sources using the URLs from the metadata when making specific claims.
            
            Context: {context}
            """),
            ("human", "{question}")
        ])

    def perform_web_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Perform web search using DuckDuckGo"""
        search_results = []
        with DDGS() as ddgs:
            results = ddgs.text(query, backend="lite", max_results=num_results)
            for r in results:
                search_results.append({
                    "title": r["title"],
                    "link": r["href"],
                    "snippet": r["body"]
                })
        return search_results

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text

    def extract_web_content(self, url: str) -> tuple:
        """Extract and process web content"""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None, None

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.extract()

            # Convert to markdown
            h2t = html2text.HTML2Text()
            h2t.ignore_links = False
            h2t.ignore_images = True
            text = h2t.handle(str(soup))
            
            # Get metadata
            title = soup.title.string if soup.title else url
            metadata = {
                'title': title,
                'url': url
            }
            
            return self.clean_text(text), metadata
            
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return None, None

    def process_web_results(self, query: str, status_callback=None) -> str:
        """Process web search results and store in vector database"""
        try:
            # Perform web search
            if status_callback:
                status_callback("Performing web search...")
            results = self.perform_web_search(query)
            
            documents = []
            for result in results:
                url = result['link']
                if status_callback:
                    status_callback(f"Processing: {url}")
                
                content, metadata = self.extract_web_content(url)
                if content:
                    # Split content into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_text(content)
                    
                    # Create documents with metadata
                    for chunk in chunks:
                        doc = Document(page_content=chunk, metadata=metadata)
                        documents.append(doc)
            
            # Update vector store
            if documents:
                if status_callback:
                    status_callback("Updating vector store...")
                texts = [doc.page_content for doc in documents]
                metadatas = [doc.metadata for doc in documents]
                self.vector_store.add_texts(texts, metadatas=metadatas)
                
            return f"Processed {len(results)} web pages successfully."
            
        except Exception as e:
            return f"Error processing web results: {str(e)}"

    def format_chat_history(self, chat_history: List[tuple]) -> List[tuple]:
        """Convert Gradio chat history to the format expected by ConversationalRetrievalChain"""
        formatted_history = []
        for human, ai in chat_history:
            # Only add completed exchanges to the history
            if human and ai:
                formatted_history.append((human, ai))
        return formatted_history

    def get_response(self, query: str, chat_history: List[tuple]) -> str:
        """Generate response using RAG"""
        try:
            # Setup retrieval chain
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 5, 'fetch_k': 20}
            )
            
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                chain_type="stuff",
                combine_docs_chain_kwargs={"prompt": self.prompt},
                get_chat_history=lambda h: "\n".join([f"Human: {q}\nAssistant: {a}" for q, a in self.format_chat_history(h)])
            )
            
            # Get response
            response = chain.invoke({
                "question": query,
                "chat_history": self.format_chat_history(chat_history)
            })
            
            return response['answer']
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Create Gradio interface
def create_gradio_interface():
    # Initialize RAG system
    rag_system = WebSearchRAG()
    
    # Define chat function
    def chat(message, history):
        # First, process web search results
        status = rag_system.process_web_results(
            message, 
            lambda x: gr.Info(x)
        )
        
        # Then get response
        response = rag_system.get_response(message, history)
        
        history.append((message, response))
        return history, ""

    # Create interface
    with gr.Blocks(title="Web Search RAG Chat") as demo:
        gr.Markdown("# Web Search RAG Chat System")
        gr.Markdown("Ask questions and get answers synthesized from real-time web search results!")

        chatbot = gr.Chatbot(
            label="Chat History",
            height=400,
            show_label=True
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask anything...",
                scale=4
            )
            submit = gr.Button("Send", scale=1)

        with gr.Row():
            clear = gr.Button("Clear Chat")
            
        with gr.Row():
            gr.Examples(
                examples=[
                    "What are the latest developments in AI?",
                    "what is Deepseek ?",
                    "What are the main features of Python 3.12?",
                ],
                inputs=msg,
                label="Example Questions"
            )

        # Event handlers
        submit.click(chat, [msg, chatbot], [chatbot, msg])
        msg.submit(chat, [msg, chatbot], [chatbot, msg])
        clear.click(lambda: [], None, chatbot)

        gr.Markdown("""
        ### How it works:
        1. Your question triggers a web search
        2. Relevant web pages are processed and stored
        3. An AI assistant synthesizes information from these sources
        4. You get a comprehensive answer with citations!
        """)

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()