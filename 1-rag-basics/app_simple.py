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
                response += f"\n### 📚 Sources\n\n"
                response += f"Based on {len(result['sources'])} document chunk(s):\n\n"

                for i, doc in enumerate(result['sources'], 1):
                    source = Path(doc.metadata.get('source', 'Unknown')).name
                    # Check if pre-chunked and show metadata
                    if doc.metadata.get('pre_chunked', False):
                        topic = doc.metadata.get('topic', '')
                        if topic:
                            response += f"{i}. 📦 `{source}` (Pre-chunked: {topic})\n"
                        else:
                            response += f"{i}. 📦 `{source}` (Pre-chunked)\n"
                    else:
                        response += f"{i}. ✂️ `{source}` (Auto-chunked)\n"

        return response

    except Exception as e:
        return f"❌ Error processing question: {str(e)}"


def get_system_info():
    """Get information about the current system state."""
    global rag_system

    if rag_system is None:
        return "System not initialized yet."

    info = f"""
### 🖥️ System Information

**Backend Configuration:**
- **Backend**: {rag_system.backend_config.name}
- **Device**: {rag_system.device.upper()}
- **RAM Required**: {rag_system.backend_config.ram_required}

**Document Processing:**
- **Embedding Model**: {rag_system.embedding_model_name}
- **Chunk Size**: {rag_system.chunk_size} characters (for auto-chunking)
- **Chunks Retrieved**: {rag_system.k_retrieve} per query
"""

    # Get chunk statistics if available
    if hasattr(rag_system, '_chunk_stats'):
        stats = rag_system._chunk_stats
        info += f"\n**Chunk Statistics:**\n"
        if stats.get('prechunked', 0) > 0:
            info += f"- 📦 **Pre-chunked**: {stats['prechunked']} (manual control)\n"
        if stats.get('autochunked', 0) > 0:
            info += f"- ✂️ **Auto-chunked**: {stats['autochunked']} (automatic)\n"
        info += f"- 📑 **Total Chunks**: {stats.get('total', 0)}\n"

    if rag_system.vectorstore:
        info += f"\n**Status**: ✅ Documents indexed and ready\n"
    else:
        info += f"\n**Status**: ❌ Not ready - initialize system first\n"

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
                show_label=True,
                type="tuples"  # Explicitly set to avoid deprecation warning
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

    ### 📚 Document Formats Supported:

    **Automatic Chunking:**
    - 📄 `.pdf` files - Automatically split into chunks
    - 📝 `.txt` files - Automatically split into chunks

    **Pre-Chunked (Advanced):**
    - 📦 `*_chunks.json` - JSON format with metadata
    - 📦 `*_chunks.txt` - Simple text format with separators

    You can **mix both formats** in the `data/` folder! See `PRECHUNKED_FORMAT.md` for details.

    ### 💡 Tips:
    - **Retrieval Only** mode shows you the relevant document chunks without generating an answer (great for learning!)
    - **Local models** (TinyLlama, Phi) download automatically on first use
    - **API models** require API keys set as environment variables
    - **Pre-chunked files** give you precise control over chunk boundaries

    ### 🔧 Need Help?
    - Setup: Check `SETUP_GUIDE.md`
    - Pre-chunking: Check `PRECHUNKED_FORMAT.md`
    - Examples: See `data/example_chunks.json` and `data/example_chunks.txt`
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
    print("=" * 70)
    print("🌐 Starting web interface...")
    print("=" * 70)
    print()
    print("📍 Access the interface at:")
    print("   • Local: http://localhost:7860")
    print("   • Network: http://127.0.0.1:7860")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70 + "\n")

    demo.launch(
        share=False,
        server_name="0.0.0.0",  # Allow access from network
        server_port=7860,
        show_error=True
    )
