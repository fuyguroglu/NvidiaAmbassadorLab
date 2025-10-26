"""
Simple Gradio Web Interface for RAG System
==========================================

This provides an easy-to-use web interface for the RAG system,
making it accessible for students without command-line experience.
"""

import gradio as gr
from rag_flexible import FlexibleRAG
from pathlib import Path

# Global RAG instance
rag_system = None


def initialize_system(backend_choice):
    """Initialize the RAG system with the selected backend."""
    global rag_system

    try:
        # Map friendly names to backend keys
        backend_map = {
            "Retrieval Only (No LLM - Works Everywhere!)": "retrieval_only",
            "TinyLlama (Lightweight - 3-4GB RAM)": "tinyllama",
            "Phi-2 (Standard - 5-6GB RAM)": "phi2",
            "Phi-3 Mini (Better - 5-6GB RAM)": "phi3_mini",
            "Groq API (Cloud - Fast & Free!)": "groq",
            "OpenAI API (Cloud - Paid)": "openai",
            "Anthropic API (Cloud - Paid)": "anthropic",
        }

        backend = backend_map.get(backend_choice, "retrieval_only")

        # Create RAG instance
        rag_system = FlexibleRAG(
            backend=backend,
            data_dir="./data",
            chunk_size=500,
            chunk_overlap=50,
            k_retrieve=3
        )

        # Setup the system
        success = rag_system.setup()

        if success:
            return f"✅ System initialized successfully with **{backend_choice}**!\n\nYou can now ask questions about your documents."
        else:
            return "❌ Setup failed! Please make sure you have documents in the `data/` folder."

    except Exception as e:
        return f"❌ Error initializing system: {str(e)}\n\nTip: Try 'Retrieval Only' mode first!"


def ask_question(question, history):
    """Process a question and return the answer."""
    global rag_system

    if rag_system is None:
        return "⚠️ Please initialize the system first using the 'Initialize System' button!"

    if not question.strip():
        return "⚠️ Please enter a question!"

    try:
        # Query the RAG system
        result = rag_system.query(question)

        if result['mode'] == 'retrieval_only':
            # Format retrieval-only response
            response = "### 📄 Retrieved Context\n\n"
            response += "*(Retrieval-only mode - showing relevant document chunks)*\n\n"

            for i, doc in enumerate(result['sources'], 1):
                source = Path(doc.metadata.get('source', 'Unknown')).name
                page = doc.metadata.get('page', 'N/A')
                content = doc.page_content

                response += f"**Chunk {i}**\n"
                response += f"- Source: `{source}`\n"
                response += f"- Page: {page}\n\n"
                response += f"{content}\n\n"
                response += "---\n\n"

        else:
            # Format full response with LLM answer
            response = f"### 💬 Answer\n\n{result['answer']}\n\n"

            if result['sources']:
                response += f"### 📚 Sources\n\n"
                response += f"Based on {len(result['sources'])} document chunk(s):\n\n"

                for i, doc in enumerate(result['sources'], 1):
                    source = Path(doc.metadata.get('source', 'Unknown')).name
                    response += f"{i}. `{source}`\n"

        return response

    except Exception as e:
        return f"❌ Error processing question: {str(e)}"


def get_system_info():
    """Get information about the current system state."""
    global rag_system

    if rag_system is None:
        return "System not initialized yet."

    info = f"""
### System Information

- **Backend**: {rag_system.backend_config.name}
- **Device**: {rag_system.device.upper()}
- **RAM Required**: {rag_system.backend_config.ram_required}
- **Embedding Model**: {rag_system.embedding_model_name}
- **Chunk Size**: {rag_system.chunk_size} characters
- **Chunks Retrieved**: {rag_system.k_retrieve} per query
"""

    if rag_system.vectorstore:
        info += f"- **Documents Indexed**: ✅ Ready\n"
    else:
        info += f"- **Documents Indexed**: ❌ Not ready\n"

    return info


# Create Gradio interface
with gr.Blocks(title="RAG System - Student Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🎓 RAG System - Interactive Demo

    Welcome! This is a **Retrieval-Augmented Generation (RAG)** system that can answer questions about your documents.

    ## How to Use:
    1. **Initialize** the system with your preferred backend
    2. **Ask questions** about the documents in the `data/` folder
    3. **View answers** with source citations

    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ System Setup")

            backend_dropdown = gr.Dropdown(
                choices=[
                    "Retrieval Only (No LLM - Works Everywhere!)",
                    "TinyLlama (Lightweight - 3-4GB RAM)",
                    "Phi-2 (Standard - 5-6GB RAM)",
                    "Phi-3 Mini (Better - 5-6GB RAM)",
                    "Groq API (Cloud - Fast & Free!)",
                    "OpenAI API (Cloud - Paid)",
                    "Anthropic API (Cloud - Paid)",
                ],
                value="Retrieval Only (No LLM - Works Everywhere!)",
                label="Select Backend",
                info="Choose based on your hardware capabilities"
            )

            init_button = gr.Button("🚀 Initialize System", variant="primary")
            init_output = gr.Markdown()

            gr.Markdown("---")

            info_button = gr.Button("ℹ️ Show System Info")
            info_output = gr.Markdown()

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")

            chatbot = gr.Chatbot(
                height=400,
                label="Conversation",
                show_label=True
            )

            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    label="Your Question",
                    lines=2,
                    scale=4
                )
                submit_button = gr.Button("Ask", variant="primary", scale=1)

            gr.Markdown("""
            ### 💡 Example Questions:
            - What is this document about?
            - What are the main responsibilities described?
            - Can you summarize the key points?
            """)

    gr.Markdown("""
    ---

    ### 📚 Tips:
    - **Retrieval Only** mode shows you the relevant document chunks without generating an answer (great for learning!)
    - **Local models** (TinyLlama, Phi) download automatically on first use
    - **API models** require API keys set as environment variables
    - Add your documents to the `data/` folder (supports .pdf and .txt files)

    ### 🔧 Need Help?
    Check the `SETUP_GUIDE.md` file for detailed installation and troubleshooting instructions!
    """)

    # Event handlers
    init_button.click(
        fn=initialize_system,
        inputs=[backend_dropdown],
        outputs=[init_output]
    )

    info_button.click(
        fn=get_system_info,
        inputs=[],
        outputs=[info_output]
    )

    def respond(message, chat_history):
        """Handle question submission."""
        answer = ask_question(message, chat_history)
        chat_history.append((message, answer))
        return "", chat_history

    submit_button.click(
        fn=respond,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot]
    )

    question_input.submit(
        fn=respond,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot]
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🌐 STARTING WEB INTERFACE")
    print("=" * 70)
    print("\nThe interface will open in your web browser automatically.")
    print("If it doesn't, copy the URL shown below and paste it in your browser.")
    print("\nPress Ctrl+C to stop the server.")
    print("=" * 70 + "\n")

    demo.launch(
        share=False,
        server_name="0.0.0.0",  # Allow access from network
        server_port=7860,
        show_error=True
    )
