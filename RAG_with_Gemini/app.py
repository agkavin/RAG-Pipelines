import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import dotenv
import shutil

# Load environment variables
dotenv.load_dotenv()

key = os.getenv("GEMINI_API_KEY")

# Initialize chat model
chat_model = ChatGoogleGenerativeAI(google_api_key=key, model="gemini-1.5-flash")

class RAGSystem:
    def __init__(self):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            google_api_key=key, 
            model="models/embedding-001"
        )
        self.db_connection = None
        self.rag_chain = None
        self.setup_chat_template()
        
        # remove chroma db if it exists
        if os.path.exists("./chroma_db_"):
            print("Clearing existing vector store")
            shutil.rmtree("./chroma_db_")

    def process_uploaded_file(self, file_path):
        try:
            # Load and split the document
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(pages)

            
            
            # Create and persist vector store
            self.db_connection = Chroma.from_documents(
                chunks,
                self.embedding_model,
                persist_directory="./chroma_db_"
            )
            self.db_connection.persist()
            
            # Set up retriever and RAG chain
            retriever = self.db_connection.as_retriever(search_kwargs={"k": 5})
            self.setup_rag_chain(retriever)
            
            return True
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return False

    def setup_chat_template(self):
        self.chat_template = ChatPromptTemplate.from_template("""
            You are an assistant for question-answering tasks. Use the following retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. Answer in bullet points.

            Context: {context} 

            Question: {question} 

            Answer:
        """)

    def setup_rag_chain(self, retriever):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.chat_template
            | chat_model
            | StrOutputParser()
        )

    def get_chat_response(self, message, history):
        if self.rag_chain is None:
            return "Please upload a document first."
        try:
            response = self.rag_chain.invoke(message)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Initialize RAG system
rag_system = RAGSystem()

def upload_file(file):
    if file is None:
        return "No file uploaded"
    try:
        success = rag_system.process_uploaded_file(file.name)
        return (
            f"Successfully processed {os.path.basename(file.name)}" 
            if success 
            else f"Failed to process {os.path.basename(file.name)}"
        )
    except Exception as e:
        return f"Error processing file: {str(e)}"

def chat(message, history):
    try:
        response = rag_system.get_chat_response(message, history)
        history.append((message, response))
        return history, ""
    except Exception as e:
        return history, str(e)

# Create Gradio interface
with gr.Blocks(title="RAG Chat System") as demo:
    gr.Markdown("# Document Chat System")
    gr.Markdown("Upload your PDF documents and chat with them using our RAG system.")

    with gr.Row():
        upload_button = gr.File(
            label="Upload PDF Document",
            file_types=[".pdf"],
            type="filepath"
        )
        file_output = gr.Textbox(label="Upload Status", interactive=False)

    with gr.Row():
        chatbot = gr.Chatbot(label="Chat History", height=400)

    with gr.Row():
        msg = gr.Textbox(
            label="Type your message",
            placeholder="Ask about your documents...",
            scale=4
        )
        submit = gr.Button("Send", scale=1)

    with gr.Row():
        clear = gr.Button("Clear Chat")
        examples = gr.Examples(
            examples=[
                "What topics are covered in the documents?",
                "Can you summarize the main points?",
                "What are the key findings?",
            ],
            inputs=msg,
            label="Example Questions"
        )

    # Event handlers
    upload_button.upload(
        fn=upload_file,
        inputs=[upload_button],
        outputs=[file_output]
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