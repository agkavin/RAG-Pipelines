from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import gradio as gr
import git
import os
import shutil
import faiss

class GitHubRAGSystem:
    def __init__(self, vector_store_dir="./github_vectors"):
        self.vector_store_dir = vector_store_dir
        self.temp_clone_dir = "./temp_repos"

        # remove vector_store if it exists
        if os.path.exists(vector_store_dir):
            print("Clearing existing vector store")
            shutil.rmtree(vector_store_dir)
        
        # remove temp_clone_dir if it exists
        if os.path.exists(self.temp_clone_dir):
            print("Clearing existing temp clone dir")
            shutil.rmtree(self.temp_clone_dir)

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
        
        # Enhanced prompt for GitHub repo analysis
        self.prompt = ChatPromptTemplate.from_template("""
            You are a GitHub repository analysis assistant. Use the following retrieved code context 
            to answer questions about the repository. Include relevant code snippets in your answers 
            when appropriate. Format your response in clear, organized bullet points.

            Code Context: {context}

            Question: {question}

            Please analyze and respond:
        """)

        # Create necessary directories
        os.makedirs(self.temp_clone_dir, exist_ok=True)
        os.makedirs(vector_store_dir, exist_ok=True)

        # Initialize vector store
        self.initialize_vector_store()
        
        # Setup the RAG chain
        self.setup_rag_chain()

    def initialize_vector_store(self):
        """Initialize or load the FAISS vector store"""
        if os.path.exists(os.path.join(self.vector_store_dir, "index.faiss")):
            self.vector_store = FAISS.load_local(
                self.vector_store_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # Initialize empty vector store
            single_vector = self.embeddings.embed_query("dummy text")
            index = faiss.IndexFlatL2(len(single_vector))
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )

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

    def process_github_repo(self, repo_url: str) -> str:
        """Clone and process a GitHub repository"""
        try:
            # Extract repo name from URL
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            clone_path = os.path.join(self.temp_clone_dir, repo_name)

            # Clear existing clone if present
            if os.path.exists(clone_path):
                shutil.rmtree(clone_path)

            # Clone repository
            git.Repo.clone_from(repo_url, clone_path)

            # Process files
            documents = []
            allowed_extensions = ['.py', '.ipynb', '.md', '.js', '.jsx', '.ts', '.tsx']

            for root, _, files in os.walk(clone_path):
                for file in files:
                    if any(file.endswith(ext) for ext in allowed_extensions):
                        try:
                            file_path = os.path.join(root, file)
                            loader = TextLoader(file_path, encoding='utf-8')
                            documents.extend(loader.load())
                        except Exception as e:
                            print(f"Error loading {file}: {str(e)}")

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)

            # Add to vector store
            self.vector_store.add_documents(documents=chunks)
            self.vector_store.save_local(self.vector_store_dir)

            # Cleanup
            shutil.rmtree(clone_path)

            return f"Successfully processed repository: {repo_name}"
        except Exception as e:
            return f"Error processing repository: {str(e)}"

    def get_response(self, query: str) -> str:
        """Generate a response using the RAG chain"""
        try:
            response = self.chain.invoke(query)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Initialize the system
rag_system = GitHubRAGSystem()

# Create Gradio interface
with gr.Blocks(title="GitHub Repository Analysis") as demo:
    gr.Markdown("# GitHub Repository Analysis System")

    with gr.Row():
        repo_url = gr.Textbox(
            label="GitHub Repository URL",
            placeholder="https://github.com/username/repository.git"
        )
        process_button = gr.Button("Process Repository")

    with gr.Row():
        status_output = gr.Textbox(label="Processing Status", interactive=False)

    with gr.Row():
        chatbot = gr.Chatbot(label="Chat History", height=400)

    with gr.Row():
        msg = gr.Textbox(
            label="Your Question",
            placeholder="Ask about the repository...",
            scale=2
        )
        submit = gr.Button("Send", scale=1)

    with gr.Row():
        clear = gr.Button("Clear Chat")
        examples = gr.Examples(
            examples=[
                "What are the main features of this repository?",
                "What dependencies does this project use?",
                "Explain the core functionality of this codebase.",
            ],
            inputs=msg,
            label="Example Questions"
        )

    def chat(message, history):
        """Handle chat interactions"""
        try:
            response = rag_system.get_response(message)
            history.append((message, response))
            return history, ""
        except Exception as e:
            return history, str(e)

    # Event handlers
    process_button.click(
        fn=rag_system.process_github_repo,
        inputs=[repo_url],
        outputs=[status_output]
    )
    
    submit.click(
        fn=chat,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    msg.submit(
        fn=chat,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()