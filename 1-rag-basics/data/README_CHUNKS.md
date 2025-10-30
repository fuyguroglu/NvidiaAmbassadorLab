# How to Use Pre-Chunked Documents

## Quick Start

### Option 1: Use Example Files
The `example_chunks.json` and `example_chunks.txt` files show working examples. Try them first!

### Option 2: Create Your Own

**JSON Format:**
```json
[
  {
    "content": "Your text here...",
    "metadata": {"source": "My Doc", "topic": "Introduction"}
  }
]
```
Save as `my_document_chunks.json`

**Text Format:**
```
---
Your first chunk here...
META: source=My Doc, topic=Introduction

---
Your second chunk here...
META: source=My Doc, topic=Details
```
Save as `my_document_chunks.txt`

## File Naming Rules

✅ **Will be recognized as pre-chunked:**
- `document_chunks.json`
- `faq_chunks.json`
- `notes_chunks.txt`
- `chunks.json`
- `chunks.txt`

❌ **Will be auto-chunked:**
- `document.pdf`
- `notes.txt`
- `data.json` (doesn't end with chunks)

## Testing

After adding your chunks:
```bash
conda activate nvidia_rag
python rag_flexible.py
```

Look for:
```
✓ Loaded pre-chunked: your_file_chunks.json (10 chunks)
```

## Full Documentation

See [PRECHUNKED_FORMAT.md](../PRECHUNKED_FORMAT.md) for complete guide with examples and best practices.
