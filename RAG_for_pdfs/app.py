# app.py

import gradio as gr
import tempfile
import os
from RAG_base import RAGSystem

# Initialize RAG system
rag_system = RAGSystem()

def upload_file(file):
    """Handle file upload"""
    if file is None:
        return "No file uploaded"

    try:
        temp_path = file.name
        success = rag_system.process_uploaded_file(temp_path)
        return (
            f"Successfully processed {os.path.basename(temp_path)}" 
            if success 
            else f"Failed to process {os.path.basename(temp_path)}"
        )
    except Exception as e:
        return f"Error processing file: {str(e)}"

def chat(message, history):
    """Handle chat interactions"""
    try:
        response = rag_system.get_chat_response(message, history)
        history.append((message, response))
        return history, ""  # Clear the message input after sending
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