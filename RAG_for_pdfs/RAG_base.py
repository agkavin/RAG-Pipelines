# RAG_base.py

import os
import shutil
import faiss
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGSystem:
    def __init__(self, data_dir="./data", vector_store_dir="./vector_store"):
        self.data_dir = data_dir
        self.vector_store_dir = vector_store_dir
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        self.llm = ChatOllama(
            model="llama3.1",  # Changed to a more standard model name
            base_url="http://localhost:11434"
        )
        
        self.prompt = ChatPromptTemplate.from_template("""
            You are an assistant for question-answering tasks. Use the following retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. Answer in bullet points.

            Context: {context} 

            Question: {question} 

            Answer:
        """)

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(vector_store_dir, exist_ok=True)
        

        if os.path.exists(os.path.join(vector_store_dir, "index.faiss")):
            self.vector_store = FAISS.load_local(
                vector_store_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            single_vector = self.embeddings.embed_query("dummy text")
            index = faiss.IndexFlatL2(len(single_vector))
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )

        self.setup_rag_chain()

    def setup_rag_chain(self):
        """Setup the RAG chain with retriever and LLM"""
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 3, 'fetch_k': 50, 'lambda_mult': 0.8}
        )
        
        self.chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def process_uploaded_file(self, file_path: str) -> bool:
        """Process a single uploaded file and add it to the vector store"""
        try:
            filename = os.path.basename(file_path)
            destination = os.path.join(self.data_dir, filename)
            shutil.copy2(file_path, destination)

            loader = PyMuPDFLoader(destination)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(docs)

            self.vector_store.add_documents(documents=chunks)
            self.vector_store.save_local(self.vector_store_dir)

            return True
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return False

    def get_chat_response(self, query: str, chat_history: List[tuple]) -> str:
        """Generate a response using the RAG chain"""
        try:
            # Pass the query string directly
            response = self.chain.invoke(query)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
