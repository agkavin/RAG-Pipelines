import gradio as gr
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

class CSVSystem:
    def __init__(self):
        # Initialize components
        self.agent = None
        self.rag_chain = None
        self.embedding_model = None
        self.chat_model = None
        self.db_connection = None
        
        # Setup models
        self.setup_models()
        self.setup_chat_template()

    def setup_models(self):
        self.embedding_model = OllamaEmbeddings(model="llama3.1")
        self.chat_model = ChatOllama(model="llama3.1", temperature=0.7)

    def setup_chat_template(self):
        self.chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful AI assistant that answers questions about CSV data.
                         For general questions, use the provided context. For specific data analysis,
                         use the CSV agent capabilities."""),
            HumanMessagePromptTemplate.from_template(
                """Answer the question based on the given context.
                Context: {context}
                Question: {question}
                Answer: """
            )
        ])

    def process_csv(self, file_path):
        try:
            # Load and split the document for RAG
            loader = TextLoader(file_path)
            documents = loader.load()
            
            # Split into chunks using RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
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
            
            # Create CSV agent
            self.agent = create_csv_agent(
                self.chat_model,
                file_path,
                verbose=True,
                allow_dangerous_code=True
            )
            
            return True, "CSV file processed successfully for both RAG and agent use!"
        except Exception as e:
            return False, f"Error processing CSV: {str(e)}"

    def setup_rag_chain(self, retriever):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.chat_template
            | self.chat_model
            | StrOutputParser()
        )

    def get_response(self, query, history):
        if self.rag_chain is None or self.agent is None:
            return "Please upload a CSV file first."
            
        try:
            # Check if query starts with @agent
            if query.startswith("@agent"):
                # Remove @agent flag and use the agent
                actual_query = query.replace("@agent", "").strip()
                response = self.agent.invoke(actual_query)
                return str(response)
            else:
                # Use RAG for general queries
                response = self.rag_chain.invoke(query)
                return response
        except Exception as e:
            return f"Error: {str(e)}"

def upload_file(file):
    """Handle file upload and system initialization"""
    if file is None:
        return None, "No file uploaded"
    
    try:
        system = CSVSystem()
        success, message = system.process_csv(file.name)
        
        if success:
            return system, f"Successfully processed {os.path.basename(file.name)}"
        else:
            return None, message
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def chat(message, history, system):
    """Handle chat interactions"""
    try:
        if system is None:
            response = "Please upload a CSV file first."
        else:
            response = system.get_response(message, history)
        
        history.append((message, response))
        return history, ""
    except Exception as e:
        history.append((message, f"Error: {str(e)}"))
        return history, ""

# Create Gradio interface
with gr.Blocks(title="CSV Analysis System") as demo:
    system_state = gr.State(None)
    
    gr.Markdown("# CSV Analysis System")
    gr.Markdown("""Upload your CSV file and interact with it in two ways:
    - Ask general questions about the data
    - Use @agent prefix for specific data analysis queries""")
    
    with gr.Row():
        upload_button = gr.File(
            label="Upload CSV File",
            file_types=[".csv"],
            type="filepath"
        )
        file_output = gr.Textbox(label="Upload Status", interactive=False)
    
    with gr.Row():
        chatbot = gr.Chatbot(
            label="Chat History",
            height=400,
            show_copy_button=True
        )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Type your message",
            placeholder="Ask about the data or use @agent for specific analysis...",
            scale=4
        )
        submit = gr.Button("Send", scale=1)
    
    with gr.Row():
        clear = gr.Button("Clear Chat")
        examples = gr.Examples(
            examples=[
                "What kind of data is in this CSV?",
                "@agent How many rows are in the dataset?",
                "Tell me about the numerical columns",
                "@agent Calculate the mean of column X",
                "What patterns do you see in the data?",
                "@agent Show me the correlation between columns Y and Z"
            ],
            inputs=msg,
            label="Example Questions"
        )
    
    # Event handlers
    upload_button.upload(
        fn=upload_file,
        inputs=[upload_button],
        outputs=[system_state, file_output]
    )
    
    submit.click(
        fn=chat,
        inputs=[msg, chatbot, system_state],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        fn=chat,
        inputs=[msg, chatbot, system_state],
        outputs=[chatbot, msg]
    )
    
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()