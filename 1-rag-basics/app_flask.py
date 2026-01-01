"""
Simple Flask Web Interface for RAG System
==========================================

A minimal, reliable web interface that works everywhere.
No complex dependencies, just Flask and basic HTML/CSS/JS.
"""

from flask import Flask, render_template_string, request, jsonify, session
from pathlib import Path
import secrets
from rag_flexible import FlexibleRAG

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # For session management

# Global RAG instance (per session would be better for production, but this is simpler for students)
rag_system = None

# HTML Template (inline to keep everything in one file)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System - Student Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }

        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }

        .main-content {
            display: grid;
            grid-template-columns: 350px 1fr;
            min-height: 600px;
        }

        .sidebar {
            background: #f8f9fa;
            padding: 25px;
            border-right: 1px solid #dee2e6;
        }

        .chat-area {
            display: flex;
            flex-direction: column;
            padding: 25px;
        }

        .section-title {
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #555;
            font-size: 0.9em;
        }

        select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 0.95em;
            background: white;
            cursor: pointer;
            transition: border-color 0.2s;
        }

        select:focus {
            outline: none;
            border-color: #667eea;
        }

        button {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 6px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: #6c757d;
            color: white;
            margin-top: 10px;
        }

        .btn-secondary:hover {
            background: #5a6268;
        }

        .status-box {
            margin-top: 15px;
            padding: 15px;
            border-radius: 6px;
            font-size: 0.9em;
            line-height: 1.6;
        }

        .status-box.success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }

        .status-box.error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }

        .status-box.info {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
        }

        .info-section {
            margin-top: 25px;
            padding-top: 25px;
            border-top: 2px solid #dee2e6;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            max-height: 500px;
        }

        .message {
            margin-bottom: 20px;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message-label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }

        .message-content {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            white-space: pre-wrap;
            line-height: 1.6;
        }

        .message.assistant .message-content {
            border-left-color: #764ba2;
        }

        .input-area {
            display: flex;
            gap: 10px;
        }

        #questionInput {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
            font-family: inherit;
            resize: vertical;
        }

        #questionInput:focus {
            outline: none;
            border-color: #667eea;
        }

        #askButton {
            width: 120px;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #667eea;
        }

        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .examples {
            margin-top: 15px;
            padding: 15px;
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            font-size: 0.85em;
        }

        .examples h4 {
            margin-bottom: 8px;
            color: #856404;
        }

        .examples ul {
            margin-left: 20px;
            color: #856404;
        }

        .examples li {
            margin: 5px 0;
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }

            .sidebar {
                border-right: none;
                border-bottom: 1px solid #dee2e6;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 RAG System - Interactive Demo</h1>
            <p>Retrieval-Augmented Generation for Document Q&A</p>
        </div>

        <div class="main-content">
            <div class="sidebar">
                <div class="section-title">⚙️ System Setup</div>

                <div class="form-group">
                    <label for="backendSelect">Select Backend:</label>
                    <select id="backendSelect">
                        <option value="retrieval_only">Retrieval Only (No LLM - Works Everywhere!)</option>
                        <option value="tinyllama">TinyLlama (Lightweight - 3-4GB RAM)</option>
                        <option value="phi2">Phi-2 (Standard - 5-6GB RAM)</option>
                        <option value="phi3_mini">Phi-3 Mini (Better - 5-6GB RAM)</option>
                        <option value="groq">Groq API (Cloud - Fast & Free!)</option>
                        <option value="openai">OpenAI API (Cloud - Paid)</option>
                        <option value="anthropic">Anthropic API (Cloud - Paid)</option>
                    </select>
                </div>

                <button class="btn-primary" onclick="initializeSystem()">🚀 Initialize System</button>

                <div id="initStatus"></div>

                <div class="info-section">
                    <button class="btn-secondary" onclick="getSystemInfo()">ℹ️ Show System Info</button>
                    <div id="systemInfo"></div>
                </div>

                <div class="examples">
                    <h4>💡 Example Questions:</h4>
                    <ul>
                        <li>What is this document about?</li>
                        <li>What are the main responsibilities?</li>
                        <li>Can you summarize the key points?</li>
                    </ul>
                </div>
            </div>

            <div class="chat-area">
                <div class="section-title">💬 Ask Questions</div>

                <div class="chat-messages" id="chatMessages">
                    <div style="text-align: center; color: #999; padding: 50px 20px;">
                        <p>👋 Welcome! Initialize the system to get started.</p>
                        <p style="margin-top: 10px;">Your questions and answers will appear here.</p>
                    </div>
                </div>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <div>Processing your question...</div>
                </div>

                <div class="input-area">
                    <textarea id="questionInput" rows="2" placeholder="Ask a question about your documents..."></textarea>
                    <button id="askButton" class="btn-primary" onclick="askQuestion()">Ask</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let chatHistory = [];

        async function initializeSystem() {
            const backend = document.getElementById('backendSelect').value;
            const statusDiv = document.getElementById('initStatus');

            statusDiv.innerHTML = '<div class="status-box info">Initializing system... This may take a minute on first run.</div>';

            try {
                const response = await fetch('/initialize', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({backend: backend})
                });

                const data = await response.json();

                if (data.success) {
                    statusDiv.innerHTML = `<div class="status-box success">✅ ${data.message}</div>`;
                } else {
                    statusDiv.innerHTML = `<div class="status-box error">❌ ${data.message}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="status-box error">❌ Error: ${error.message}</div>`;
            }
        }

        async function getSystemInfo() {
            const infoDiv = document.getElementById('systemInfo');

            try {
                const response = await fetch('/system-info');
                const data = await response.json();

                if (data.success) {
                    infoDiv.innerHTML = `<div class="status-box info" style="margin-top: 15px;">${data.info.replace(/\\n/g, '<br>')}</div>`;
                } else {
                    infoDiv.innerHTML = `<div class="status-box error" style="margin-top: 15px;">${data.message}</div>`;
                }
            } catch (error) {
                infoDiv.innerHTML = `<div class="status-box error" style="margin-top: 15px;">Error: ${error.message}</div>`;
            }
        }

        async function askQuestion() {
            const questionInput = document.getElementById('questionInput');
            const question = questionInput.value.trim();

            if (!question) {
                alert('Please enter a question!');
                return;
            }

            // Add user message to chat
            addMessage('user', question);
            questionInput.value = '';

            // Show loading
            document.getElementById('loading').style.display = 'block';

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                });

                const data = await response.json();

                if (data.success) {
                    addMessage('assistant', data.answer);
                } else {
                    addMessage('assistant', `❌ Error: ${data.message}`);
                }
            } catch (error) {
                addMessage('assistant', `❌ Error: ${error.message}`);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function addMessage(role, content) {
            const messagesDiv = document.getElementById('chatMessages');

            // Clear welcome message if present
            if (chatHistory.length === 0) {
                messagesDiv.innerHTML = '';
            }

            chatHistory.push({role, content});

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.innerHTML = `
                <div class="message-label">${role === 'user' ? '👤 You' : '🤖 Assistant'}</div>
                <div class="message-content">${content.replace(/\\n/g, '<br>')}</div>
            `;

            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // Allow Enter to submit (Shift+Enter for new line)
        document.getElementById('questionInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the RAG system with selected backend."""
    global rag_system

    try:
        data = request.json
        backend = data.get('backend', 'retrieval_only')

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
            return jsonify({
                'success': True,
                'message': f'System initialized successfully with {backend}! You can now ask questions.'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Setup failed! Please make sure you have documents in the data/ folder.'
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error initializing system: {str(e)}'
        })


@app.route('/query', methods=['POST'])
def query():
    """Process a question and return the answer."""
    global rag_system

    if rag_system is None:
        return jsonify({
            'success': False,
            'message': 'Please initialize the system first!'
        })

    try:
        data = request.json
        question = data.get('question', '').strip()

        if not question:
            return jsonify({
                'success': False,
                'message': 'Please provide a question!'
            })

        # Query the RAG system
        result = rag_system.query(question)

        # Format the response
        if result['mode'] == 'retrieval_only':
            # Format retrieval-only response
            answer = "### 📄 Retrieved Context\n\n"
            answer += "*(Retrieval-only mode - showing relevant document chunks)*\n\n"

            for i, doc in enumerate(result['sources'], 1):
                source = Path(doc.metadata.get('source', 'Unknown')).name
                page = doc.metadata.get('page', 'N/A')
                content = doc.page_content

                answer += f"**Chunk {i}**\n"
                answer += f"- Source: {source}\n"
                answer += f"- Page: {page}\n\n"
                answer += f"{content}\n\n"
                answer += "---\n\n"
        else:
            # Format full response with LLM answer
            answer = f"### 💬 Answer\n\n{result['answer']}\n\n"

            if result['sources']:
                answer += f"\n### 📚 Sources\n\n"
                answer += f"Based on {len(result['sources'])} document chunk(s):\n\n"

                for i, doc in enumerate(result['sources'], 1):
                    source = Path(doc.metadata.get('source', 'Unknown')).name
                    # Check if pre-chunked and show metadata
                    if doc.metadata.get('pre_chunked', False):
                        topic = doc.metadata.get('topic', '')
                        if topic:
                            answer += f"{i}. 📦 {source} (Pre-chunked: {topic})\n"
                        else:
                            answer += f"{i}. 📦 {source} (Pre-chunked)\n"
                    else:
                        answer += f"{i}. ✂️ {source} (Auto-chunked)\n"

        return jsonify({
            'success': True,
            'answer': answer
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing question: {str(e)}'
        })


@app.route('/system-info')
def system_info():
    """Get information about the current system state."""
    global rag_system

    if rag_system is None:
        return jsonify({
            'success': False,
            'message': 'System not initialized yet.'
        })

    try:
        info = f"🖥️ System Information\n\n"
        info += f"Backend: {rag_system.backend_config.name}\n"
        info += f"Device: {rag_system.device.upper()}\n"
        info += f"RAM Required: {rag_system.backend_config.ram_required}\n\n"

        info += f"Document Processing:\n"
        info += f"- Embedding Model: {rag_system.embedding_model_name}\n"
        info += f"- Chunk Size: {rag_system.chunk_size} characters\n"
        info += f"- Chunks Retrieved: {rag_system.k_retrieve} per query\n"

        # Get chunk statistics if available
        if hasattr(rag_system, '_chunk_stats'):
            stats = rag_system._chunk_stats
            info += f"\nChunk Statistics:\n"
            if stats.get('prechunked', 0) > 0:
                info += f"- 📦 Pre-chunked: {stats['prechunked']} (manual control)\n"
            if stats.get('autochunked', 0) > 0:
                info += f"- ✂️ Auto-chunked: {stats['autochunked']} (automatic)\n"
            info += f"- 📑 Total Chunks: {stats.get('total', 0)}\n"

        if rag_system.vectorstore:
            info += f"\nStatus: ✅ Documents indexed and ready"
        else:
            info += f"\nStatus: ❌ Not ready"

        return jsonify({
            'success': True,
            'info': info
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting system info: {str(e)}'
        })


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🌐 STARTING FLASK WEB INTERFACE")
    print("=" * 70)
    print("\nFlask is a simple, reliable web framework.")
    print("This interface will work on any system with Python installed.")
    print("\nPress Ctrl+C to stop the server.")
    print("=" * 70)
    print()
    print("📍 Access the interface at:")
    print("   • http://localhost:7860")
    print("   • http://127.0.0.1:7860")
    print()
    print("Note: If port 7860 is busy, Flask will suggest an alternative.")
    print("=" * 70 + "\n")

    app.run(
        host='0.0.0.0',
        port=7860,
        debug=False,  # Set to True for development
        threaded=True
    )
