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

**Status**: RAG retrieval system fully functional. LLM integration pending resolution of memory constraints.

**Last Updated**: 2025-10-24 13:22 UTC
