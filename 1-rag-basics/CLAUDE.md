# RAG Basics - Development Session Notes

## Session Date: 2025-10-24

### What We Accomplished

✅ **Complete RAG System Implementation**
- Created `rag_simple.py` - Core RAG engine with full functionality
- Created `app.py` - Gradio web interface for easy interaction
- Created `test_rag.py` - Test script for validation
- Created `README.md` - Comprehensive documentation
- Created `requirements.txt` - All dependencies

✅ **Successful Components**
1. **Document Loading**: Successfully loaded PDF (CIU_REGULATIONS_ON_STUDENT_ADISORY_SERVICES.pdf)
2. **Text Chunking**: Created 23 chunks from 3-page document (500 chars per chunk, 50 char overlap)
3. **Embeddings**: Sentence-BERT embeddings working perfectly
4. **Vector Database**: ChromaDB successfully created with all chunks
5. **Retrieval**: Document retrieval system fully functional

### Current Status

**Working:**
- ✅ PDF document loading (PyPDF)
- ✅ Text splitting and chunking
- ✅ Embedding generation (sentence-transformers/all-MiniLM-L6-v2)
- ✅ Vector database creation (ChromaDB)
- ✅ Similarity search and retrieval

**Issue Encountered:**
- ❌ Out of memory error loading microsoft/phi-2 model (2.7B parameters, ~5GB)
- The model was successfully downloaded but failed to load into RAM
- Error: `RuntimeError: unable to mmap 4995584424 bytes from file`

### Files Created

```
1-rag-basics/
├── rag_simple.py          # Core RAG implementation (297 lines)
├── app.py                 # Gradio web interface (213 lines)
├── test_rag.py            # Test script (68 lines)
├── README.md              # Comprehensive documentation
├── requirements.txt       # Python dependencies
├── CLAUDE.md             # This file
└── data/
    └── CIU_REGULATIONS_ON_STUDENT_ADISORY_SERVICES.pdf (42KB)
```

### Technical Details

**Architecture:**
```
Question → Embedding → Vector Search → Top-K Chunks → LLM → Answer
```

**Key Components:**
1. **Document Loaders**: LangChain (PyPDFLoader, TextLoader)
2. **Text Splitter**: RecursiveCharacterTextSplitter
3. **Embeddings**: HuggingFace Sentence-BERT (all-MiniLM-L6-v2)
4. **Vector Store**: ChromaDB
5. **LLM**: Microsoft Phi-2 (intended, but memory constrained)

**Configuration:**
- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Top-K retrieval: 3 chunks
- Device: CPU (no GPU available in WSL2)

### Dependencies Fixed

During installation, we encountered LangChain import changes and fixed:
```python
# Old imports (deprecated)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# New imports (working)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
```

### Test Results

```
Total documents loaded: 3
Created 23 chunks
Embeddings initialized: ✓
Vector database created: ✓ (23 chunks indexed)
Model download: ✓ (microsoft/phi-2 downloaded)
Model loading: ✗ (out of memory)
```

### Next Steps (TODO)

1. **Immediate Solutions:**
   - [ ] Use smaller model (TinyLlama, DistilGPT2)
   - [ ] Use API-based LLM (OpenAI, Anthropic)
   - [ ] Test retrieval-only mode (without LLM)
   - [ ] Try quantized models (4-bit, 8-bit)

2. **Alternative Approaches:**
   - [ ] Use cloud-based inference (Hugging Face Inference API)
   - [ ] Implement retrieval-only demo showing chunks
   - [ ] Use lightweight local models
   - [ ] Consider GPU instance for proper testing

3. **Code Improvements:**
   - [ ] Add error handling for OOM scenarios
   - [ ] Add model size detection and warnings
   - [ ] Create lightweight testing mode
   - [ ] Add configuration for different model backends

4. **Documentation:**
   - [ ] Add troubleshooting section for memory issues
   - [ ] Document alternative model options
   - [ ] Create comparison table of model requirements
   - [ ] Add resource requirements guide

### Lessons Learned

1. **LangChain Evolution**: LangChain has significantly reorganized imports in recent versions. Need to use `langchain-classic`, `langchain-core`, `langchain-community`, and `langchain-text-splitters` separately.

2. **Model Memory Requirements**:
   - Phi-2 (2.7B params) requires ~6GB RAM minimum
   - WSL2 environment may have additional memory constraints
   - Need to plan for smaller models or API-based solutions

3. **RAG Architecture**: The retrieval components work perfectly independently of the LLM, demonstrating that RAG can be modular.

4. **First-Time Setup**: Model downloads take significant time and space:
   - Phi-2: ~5GB download
   - Embeddings: ~80MB
   - Total setup time: ~10 minutes

### System Information

- **OS**: Linux (WSL2) - `Linux 6.6.87.2-microsoft-standard-WSL2`
- **Python**: 3.13
- **CUDA**: Available but not utilized (CPU-only run)
- **Working Directory**: `/mnt/d/Desktop/Nvidia Ambassador/1-rag-basics`

### Performance Notes

- PDF loading: Instant
- Chunking 3 pages: <1 second
- Embedding model download: ~30 seconds
- Embedding generation: ~5 seconds for 23 chunks
- Vector DB creation: ~2 seconds
- Model download: ~8 minutes (first time only)
- Model loading: Failed (OOM)

### References

- LangChain Documentation: https://python.langchain.com/
- ChromaDB: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- Microsoft Phi-2: https://huggingface.co/microsoft/phi-2

### Future Enhancements

- [ ] Multi-document support testing
- [ ] Different embedding models comparison
- [ ] Retrieval quality evaluation
- [ ] Response quality metrics
- [ ] Web UI deployment
- [ ] API endpoint creation
- [ ] Docker containerization
- [ ] Cloud deployment guide

---

## Session Date: 2025-10-26

### What We Accomplished

✅ **Flexible Backend System**
- Created `config.py` with 7 different backend options
- Supports retrieval-only, local LLMs (TinyLlama, Phi-2, Phi-3), and cloud APIs (Groq, OpenAI, Anthropic)
- Hardware-agnostic design: works from 4GB laptops to GPU workstations

✅ **Enhanced RAG Implementation**
- Replaced `rag_simple.py` with `rag_flexible.py`
- Modular backend initialization
- Automatic GPU detection and utilization
- Support for both local and API-based models

✅ **Web Interface**
- Created `app_simple.py` with Gradio
- Beautiful UI with backend selection dropdown
- Real-time system information display
- Chat-style Q&A interface with source citations

✅ **Testing & Validation**
- `test_system.py` - Retrieval-only mode testing
- `test_with_llm.py` - LLM backend testing
- Successfully tested with TinyLlama on GPU (cuda:0)
- Generated answers with source citations

✅ **Cross-Platform Support**
- Created `SETUP_GUIDE.md` with detailed Windows & Linux instructions
- Platform-specific launcher scripts (.sh for Linux, .bat for Windows)
- WSL2 setup instructions included

✅ **Documentation**
- Completely rewrote README.md for new flexible system
- Added troubleshooting sections
- Included learning objectives and experiments
- RAG vs Fine-tuning comparison

✅ **Project Cleanup**
- Removed old files (rag_simple.py, app.py, test_rag.py)
- Updated requirements.txt (added accelerate)
- Organized file structure
- Ready for git commit

### Technical Achievements

**Working Components:**
- ✅ Multi-backend RAG system (7 options)
- ✅ Gradio web interface (http://localhost:7860)
- ✅ GPU acceleration (CUDA detected and utilized)
- ✅ TinyLlama integration (2.2GB model, successfully loaded)
- ✅ Retrieval system (sentence-transformers embeddings + ChromaDB)
- ✅ Document processing (PDF + TXT support)

**Test Results:**
```
Backend: TinyLlama 1.1B
Device: cuda (GPU)
Documents: 2 PDFs (6 pages)
Chunks: 41 created (500 chars, 50 overlap)
Model: Successfully downloaded and loaded
Answer Generated: ✅ (with source citations)
```

### Interesting Discoveries

**RAG Retrieval Challenge:**
- Discovered semantic search limitation: query "What are the tasks of the Student Advisors?" retrieved chunks about administration tasks (appointing, supervising advisors) instead of advisor duties
- The correct information existed in the document but wasn't ranked in top-3
- Model hallucinated answer based on general knowledge despite prompt instructions
- Perfect teaching moment for students about retrieval quality vs LLM instruction-following

**Model Behavior:**
- TinyLlama includes full prompt template in output (small model limitation)
- Demonstrates difference between small local models and larger/cloud models
- Shows trade-offs between model size and instruction-following capability

### Files Created/Modified

**New Files:**
- config.py (250 lines)
- rag_flexible.py (480 lines)
- app_simple.py (238 lines)
- test_system.py (46 lines)
- test_with_llm.py (51 lines)
- SETUP_GUIDE.md (693 lines)
- start_web_interface.sh
- start_web_interface.bat

**Modified:**
- README.md (completely rewritten, 345 lines)
- requirements.txt (added accelerate)
- CLAUDE.md (this file)

**Removed:**
- rag_simple.py (old version)
- app.py (old version)
- test_rag.py (old version)

### Dependencies Added

- accelerate>=0.20.0 (required for device_map with transformers)

### System Information

- **OS**: Linux (WSL2)
- **Python**: 3.11
- **CUDA**: Available and utilized
- **GPU**: Detected and working
- **Working Directory**: `/home/monster/NvidiaAmbassadorLab/1-rag-basics`

### Performance Notes

**Retrieval-Only Mode:**
- Setup: ~5 seconds
- Query: < 1 second
- RAM: < 1GB

**TinyLlama (Local LLM):**
- First download: ~5-10 minutes (2.2GB)
- Model loading: ~30 seconds (GPU)
- Query: ~10-20 seconds (GPU)
- RAM: ~3-4GB

**Web Interface:**
- Startup: < 5 seconds
- Port: 7860
- Responsive and user-friendly

### Next Steps (TODO)

**Improvements to Consider:**
- [ ] Add post-processing to clean TinyLlama output (extract answer only)
- [ ] Improve prompt template for better instruction-following
- [ ] Implement hybrid search (semantic + keyword/BM25)
- [ ] Adjust chunking strategy for better retrieval
- [ ] Add retrieval scoring/ranking visualization
- [ ] Test with Phi-2/Phi-3 for comparison
- [ ] Add evaluation metrics for retrieval quality

**Student Features:**
- [ ] Add sample generic document (not course-specific)
- [ ] Create tutorial videos/screenshots
- [ ] Add more example questions
- [ ] Include retrieval quality debugging guide

**Future Enhancements:**
- [ ] Multi-document upload via web UI
- [ ] Chat history persistence
- [ ] Export conversation to file
- [ ] Comparison mode (test multiple backends simultaneously)
- [ ] Performance benchmarking dashboard

### Lessons Learned

1. **Backend Flexibility is Key**: Having multiple backend options makes the system accessible to all students regardless of hardware

2. **Retrieval Quality Matters**: Even with a perfect document, semantic search can miss relevant chunks - chunking strategy and k-value are critical

3. **Model Size Trade-offs**: TinyLlama works but has limitations (instruction-following, output format) - demonstrates why larger models or APIs are preferred for production

4. **Web UI Improves Accessibility**: Gradio interface makes the system much more approachable than command-line for students

5. **Documentation is Critical**: Comprehensive setup guide with platform-specific instructions prevents setup frustration

6. **First-Run Experience**: Model downloads can be slow - important to set student expectations

### References

- LangChain: https://python.langchain.com/
- ChromaDB: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- Gradio: https://www.gradio.app/
- TinyLlama: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Groq: https://console.groq.com/

---

## Session Date: 2025-10-30

### What We Accomplished

✅ **Smart GPU Detection & Installation System**
- Created `detect_gpu.py` - Automatic GPU detection utility
- Created `setup.sh` and `setup.bat` - Intelligent setup scripts
- Automatic conda TOS acceptance
- GPU-aware package installation (CPU-only vs GPU with CUDA)
- Saves ~2.5GB bandwidth for students without GPUs

**Key Innovation**: Smart recommendation system
```
🔍 Detecting GPU capabilities...
💻 No GPU detected (CPU-only mode)

💡 Recommendation: Install CPU-only version (smaller, no CUDA)
Follow this recommendation? (Y/n): [Enter]
🔄 Installing recommended version...
```

✅ **Requirements Files Split**
- `requirements-gpu.txt` - CUDA support (~2-3GB)
- `requirements-cpu.txt` - CPU-only (~200MB)
- `requirements-base.txt` - Base dependencies only
- Fixed: Changed `--index-url` to `--extra-index-url` for proper PyPI + PyTorch index usage

✅ **Pre-Chunked Document Support (Major Feature)**
- Two input formats:
  - `*_chunks.json` - JSON format with rich metadata
  - `*_chunks.txt` - Simple text format with `---` separators
- Automatic detection and handling
- Mix auto-chunked and pre-chunked in same folder
- Full metadata support

**Example JSON Format**:
```json
[
  {
    "content": "Machine learning is...",
    "metadata": {"source": "ML Intro", "topic": "Definition"}
  }
]
```

**Example Text Format**:
```
---
First chunk content...
META: source=Doc, topic=Introduction

---
Second chunk content...
META: source=Doc, topic=Details
```

✅ **Updated RAG Engine (`rag_flexible.py`)**
- `load_prechunked_json()` - Load JSON chunks
- `load_prechunked_txt()` - Load text chunks
- Enhanced `load_documents()` - Auto-detect file types
- Updated `split_documents()` - Skip pre-chunked, process others
- Track statistics for UI display

✅ **Enhanced Web Interface (`app_simple.py`)**
- Shows chunk statistics (pre-chunked vs auto-chunked)
- Source display indicates chunking method:
  - 📦 Pre-chunked chunks (with metadata)
  - ✂️ Auto-chunked chunks
- Updated tips and documentation links
- Fixed Gradio deprecation warning

✅ **Comprehensive Documentation**
- `PRECHUNKED_FORMAT.md` - Complete guide (700+ lines)
  - Format specifications
  - Best practices
  - Use cases and examples
  - Common mistakes
  - Conversion tips
- `data/README_CHUNKS.md` - Quick reference
- `data/example_chunks.json` - Working JSON example
- `data/example_chunks.txt` - Working text example
- `GPU_SETUP_INFO.md` - GPU detection system documentation
- Updated `README.md` with all new features
- Updated `SETUP_GUIDE.md` with GPU detection

✅ **URL Fix**
- Updated all scripts to show correct URLs (`localhost:7860` not `0.0.0.0:7860`)
- Clear warnings in startup scripts
- Better messaging in `app_simple.py`

### Technical Achievements

**Files Created/Modified:**
```
New Files:
- detect_gpu.py
- setup.sh, setup.bat
- requirements-gpu.txt, requirements-cpu.txt, requirements-base.txt
- PRECHUNKED_FORMAT.md
- GPU_SETUP_INFO.md
- data/example_chunks.json
- data/example_chunks.txt
- data/README_CHUNKS.md

Modified Files:
- rag_flexible.py (added pre-chunked support)
- app_simple.py (enhanced UI)
- start_web_interface.sh, start_web_interface.bat (URL fix + GPU status)
- README.md (comprehensive updates)
- SETUP_GUIDE.md (GPU detection info)
- CLAUDE.md (this file)
```

**Working Flow:**
```
1. Student runs: ./setup.sh
2. Script detects: No GPU → recommends CPU version
3. Student presses: Enter (accepts recommendation)
4. Installs: CPU-only PyTorch (~200MB vs ~2GB)
5. Student adds: PDFs + pre-chunked JSON files
6. System loads: Both formats automatically
7. Web UI shows: Statistics and chunk types
8. Student queries: See which chunks were used (pre vs auto)
```

### Test Results

```
Testing pre-chunked document loading...
📂 Loading documents from data...
  ✓ Loaded pre-chunked: example_chunks.json (5 chunks)
  ✓ Loaded pre-chunked: example_chunks.txt (5 chunks)
  ✓ Loaded: CIU_REGULATIONS_ON_STUDENT_ADISORY_SERVICES.pdf

📚 Total documents loaded: 13
   ✓ 10 pre-chunked
   ✓ 3 will be auto-chunked

✂️  Splitting documents...
   Created 23 auto-chunks
   Kept 10 pre-chunks as-is

📑 Total chunks: 33

✅ Test successful!
```

### System Impact

**Bandwidth Savings:**
- Students without GPU: ~2.5GB saved per installation
- 100 students (50% no GPU): ~125GB total saved
- Setup time: Reduced from 20min → 10min for CPU-only

**Pre-Chunked Use Cases:**
1. **FAQ Systems** - Each Q&A pair precisely chunked
2. **Definitions** - Term-by-term control
3. **Code Documentation** - Function-level chunks
4. **Course Material** - Manually curated with metadata
5. **Quality Control** - Review and refine each chunk

**Student Benefits:**
- ✅ Full control over chunking when needed
- ✅ Automatic chunking for quick experiments
- ✅ Mix both approaches in same project
- ✅ Rich metadata support
- ✅ Clear visibility in UI

### Lessons Learned

1. **Smart Defaults Matter**: Auto-detecting GPU and defaulting to "Yes" makes setup painless

2. **Package Indices**: `--extra-index-url` (additive) vs `--index-url` (exclusive) - critical for PyTorch + PyPI packages

3. **Flexibility is Key**: Giving students both auto-chunking AND pre-chunking opens up advanced use cases while keeping it simple for beginners

4. **UI Transparency**: Showing chunk statistics and types helps students understand what's happening

5. **Documentation is Critical**: Comprehensive guides (`PRECHUNKED_FORMAT.md`) with examples prevent confusion

6. **Import Changes**: langchain updates require `langchain_core.documents.Document` instead of `langchain.schema.Document`

### Next Steps (Potential Future Enhancements)

**Could Be Added:**
- [ ] Chunk quality metrics/validation
- [ ] Automatic chunk generation from documents (AI-assisted)
- [ ] Chunk editing interface in web UI
- [ ] Export retrieved chunks to pre-chunked format
- [ ] Hybrid search (semantic + keyword)
- [ ] Chunk size recommendations based on content type
- [ ] Batch conversion tools for existing documents

### Performance Notes

**Setup Time:**
- GPU detection: < 1 second
- Environment creation: ~2-3 minutes
- CPU-only installation: ~5-10 minutes
- GPU installation: ~10-20 minutes

**Runtime:**
- Pre-chunked file loading: Instant
- System shows clear progress for all operations
- Web interface responsive and informative

### References

- LangChain: https://python.langchain.com/
- ChromaDB: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- Gradio: https://www.gradio.app/
- PyTorch: https://pytorch.org/

---

**Status**: Complete flexible RAG system with multiple backends, web interface, full documentation, GPU detection, and pre-chunked document support.

**Last Updated**: 2025-10-30 10:30 UTC
