"""
RAG Application with Gradio Interface
======================================

A user-friendly web interface for the RAG system.
Upload documents, ask questions, and get AI-powered answers!
"""

import gradio as gr
from pathlib import Path
import sys
import traceback
from typing import Tuple, List

from rag_simple import SimpleRAG


class RAGApp:
    """
    Gradio web application for RAG system.
    """

    def __init__(self):
        self.rag = None
        self.is_initialized = False
        self.data_dir = Path("./data")

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)

    def initialize_rag(self, status_callback=None) -> str:
        """
        Initialize the RAG system.

        Returns:
            Status message
        """
        try:
            if status_callback:
                status_callback("Initializing RAG system...")

            self.rag = SimpleRAG(
                data_dir=str(self.data_dir),
                model_name="microsoft/phi-2",
                chunk_size=500,
                chunk_overlap=50,
                k_retrieve=3
            )

            if status_callback:
                status_callback("Loading documents and creating embeddings...")

            success = self.rag.setup()

            if success:
                self.is_initialized = True
                return "✅ RAG system initialized successfully! You can now ask questions."
            else:
                return "⚠️ No documents found in data/ directory. Please add some documents first."

        except Exception as e:
            error_msg = f"❌ Error initializing RAG: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg

    def answer_question(self, question: str, chat_history: List) -> Tuple[List, str]:
        """
        Process a question and return the answer.

        Args:
            question: User's question
            chat_history: Previous conversation history

        Returns:
            Updated chat history and empty string (to clear input)
        """
        if not self.is_initialized or self.rag is None:
            chat_history.append((question, "❌ Please initialize the RAG system first by clicking 'Initialize RAG System'."))
            return chat_history, ""

        if not question.strip():
            return chat_history, ""

        try:
            # Get answer from RAG
            result = self.rag.query(question)
            answer = result['answer']
            sources = result['sources']

            # Format answer with sources
            formatted_answer = answer

            if sources:
                formatted_answer += "\n\n**Sources:**\n"
                for i, doc in enumerate(sources, 1):
                    source = Path(doc.metadata.get('source', 'Unknown')).name
                    page = doc.metadata.get('page', 'N/A')
                    formatted_answer += f"{i}. {source}"
                    if page != 'N/A':
                        formatted_answer += f" (Page {page})"
                    formatted_answer += "\n"

            chat_history.append((question, formatted_answer))

        except Exception as e:
            error_msg = f"❌ Error processing question: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            chat_history.append((question, error_msg))

        return chat_history, ""

    def get_relevant_context(self, question: str) -> str:
        """
        Show relevant document chunks for a question (debugging/exploration).

        Args:
            question: The query

        Returns:
            Formatted string showing relevant chunks
        """
        if not self.is_initialized or self.rag is None:
            return "❌ Please initialize the RAG system first."

        if not question.strip():
            return "Please enter a question."

        try:
            chunks = self.rag.get_relevant_chunks(question, k=5)

            result = f"**Top {len(chunks)} relevant chunks for:** \"{question}\"\n\n"
            result += "=" * 60 + "\n\n"

            for i, chunk in enumerate(chunks, 1):
                source = Path(chunk['source']).name
                result += f"**Chunk {i}** (from {source}):\n"
                result += f"{chunk['content']}\n\n"
                result += "-" * 60 + "\n\n"

            return result

        except Exception as e:
            return f"❌ Error retrieving chunks: {str(e)}"

    def list_documents(self) -> str:
        """
        List all documents in the data directory.

        Returns:
            Formatted string listing documents
        """
        txt_files = list(self.data_dir.glob("*.txt"))
        pdf_files = list(self.data_dir.glob("*.pdf"))

        if not txt_files and not pdf_files:
            return "📁 No documents found in data/ directory.\n\nPlease add .txt or .pdf files to get started!"

        result = "📁 **Documents in data/ directory:**\n\n"

        if txt_files:
            result += "**Text Files:**\n"
            for f in txt_files:
                size_kb = f.stat().st_size / 1024
                result += f"  • {f.name} ({size_kb:.1f} KB)\n"
            result += "\n"

        if pdf_files:
            result += "**PDF Files:**\n"
            for f in pdf_files:
                size_kb = f.stat().st_size / 1024
                result += f"  • {f.name} ({size_kb:.1f} KB)\n"

        return result

    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="RAG Q&A System", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🤖 RAG (Retrieval-Augmented Generation) Q&A System

            Upload your documents to the `data/` directory, initialize the system, and start asking questions!
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=400,
                        show_label=True
                    )

                    with gr.Row():
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything about your documents...",
                            lines=2,
                            scale=4
                        )
                        submit_btn = gr.Button("Ask", variant="primary", scale=1)

                    clear_btn = gr.Button("Clear Chat")

                with gr.Column(scale=1):
                    # Control panel
                    gr.Markdown("### 🎛️ Control Panel")

                    init_btn = gr.Button("🚀 Initialize RAG System", variant="primary")
                    init_status = gr.Textbox(
                        label="Status",
                        lines=3,
                        interactive=False
                    )

                    gr.Markdown("---")

                    docs_btn = gr.Button("📄 List Documents")
                    docs_output = gr.Textbox(
                        label="Documents",
                        lines=8,
                        interactive=False
                    )

            with gr.Accordion("🔍 Advanced: View Retrieved Context", open=False):
                gr.Markdown("See which document chunks are being used to answer your question.")

                with gr.Row():
                    context_input = gr.Textbox(
                        label="Question",
                        placeholder="Enter a question to see relevant chunks...",
                        scale=3
                    )
                    context_btn = gr.Button("Retrieve", scale=1)

                context_output = gr.Textbox(
                    label="Relevant Document Chunks",
                    lines=15,
                    interactive=False
                )

            gr.Markdown("""
            ---
            ### 📝 Instructions:

            1. **Add Documents**: Place your .txt or .pdf files in the `1-rag-basics/data/` directory
            2. **Initialize**: Click "Initialize RAG System" (this may take a few minutes first time)
            3. **Ask Questions**: Type your questions and get AI-powered answers based on your documents!
            4. **Explore Context**: Use the Advanced section to see which document chunks are being retrieved

            ### 💡 Tips:
            - Be specific in your questions for better answers
            - The system retrieves relevant context from your documents automatically
            - You can see the source documents for each answer
            - Try the "View Retrieved Context" feature to understand how RAG works!
            """)

            # Event handlers
            init_btn.click(
                fn=self.initialize_rag,
                outputs=init_status
            )

            submit_btn.click(
                fn=self.answer_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )

            question_input.submit(
                fn=self.answer_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )

            clear_btn.click(
                fn=lambda: [],
                outputs=chatbot
            )

            docs_btn.click(
                fn=self.list_documents,
                outputs=docs_output
            )

            context_btn.click(
                fn=self.get_relevant_context,
                inputs=context_input,
                outputs=context_output
            )

            # Auto-load documents list on startup
            interface.load(
                fn=self.list_documents,
                outputs=docs_output
            )

        return interface


def main():
    """
    Launch the RAG application.
    """
    print("=" * 60)
    print("Starting RAG Q&A Application...")
    print("=" * 60)

    app = RAGApp()
    interface = app.create_interface()

    # Launch with public link option
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public sharing
        show_error=True
    )


if __name__ == "__main__":
    main()
