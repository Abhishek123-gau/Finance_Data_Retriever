# Finance Data Retriever

A multi-stage document retrieval pipeline for searching and ranking financial documents (e.g., employment agreements, financial reports) using hybrid retrieval methods.

## Overview

This project implements a **three-stage retrieval system** that combines:
1. **BM25 (Sparse Retrieval):** Fast keyword-based ranking
2. **Dense Retrieval:** Semantic similarity using sentence transformers
3. **Cross-Encoder Re-ranking:** Fine-grained relevance scoring

This hybrid approach balances speed, semantic understanding, and precision for finding the most relevant document chunks.

## Features

- **Text Chunking:** Splits large documents into 200-word chunks for manageable search units
- **Preprocessing:** Standardizes text (lowercase, removes special chars, normalizes whitespace)
- **BM25 Ranking:** Quick sparse retrieval to identify top 30 candidate chunks
- **Semantic Embeddings:** Encodes chunks and queries using `all-MiniLM-L6-v2` model
- **Cross-Encoder Re-ranking:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for fine-grained scoring
- **Top Results:** Displays top 10 most relevant chunks by final score

## Installation

### Prerequisites
- Python 3.8+
- pip

### Step 1: Install Core Dependencies
```bash
pip install rank_bm25
pip install sentence-transformers
pip install torch
pip install nltk
```

### Step 2: Clone or Set Up the Repository
```bash
git clone <repository-url>
cd Retrieval
```

### Step 3: Prepare Input Data
Place your document or text file at:
```
./input.txt
```

## Usage

### Running the Notebook

1. **Open** `Finance.ipynb` in Jupyter or VS Code with Jupyter extension.

2. **Run cells in order:**
   - Cell 1: Install `rank_bm25`
   - Cells 2-4: Import libraries and download NLTK data
   - Cell 5: Load your document from `input.txt`
   - Cell 6: Check document length
   - Cell 7: Split into 200-word chunks
   - Cell 8: Display first chunk
   - Cell 9: BM25 preprocessing, indexing, and query (returns top 5 by BM25 score)
   - Cells 10-11: Collect top 30 chunks
   - Cell 12: Load sentence transformer model and compute dense embeddings
   - Cell 13: Extract top 20 chunks by cosine similarity
   - Cell 14: Load cross-encoder and score/rank all pairs

3. **View Results:** Each stage prints top results with scores.

### Example Query
```python
query = "executive employment agreement compensation"
```

Modify this in Cell 9 to search for different topics.

## Architecture

### Stage 1: BM25 Sparse Retrieval
- **Input:** Preprocessed document chunks, tokenized query
- **Output:** Top 30 chunks ranked by BM25 score
- **Speed:** Fast (milliseconds)
- **Limitation:** Keyword-only, misses semantic similarity

### Stage 2: Dense Retrieval (Sentence Transformers)
- **Input:** Top 30 chunks, query
- **Model:** `all-MiniLM-L6-v2` (sentence embeddings)
- **Output:** Cosine similarity scores, top 20 chunks
- **Speed:** Moderate (seconds for embedding)
- **Benefit:** Captures semantic meaning

### Stage 3: Cross-Encoder Re-ranking
- **Input:** Top 20 (query, chunk) pairs
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Output:** Fine-grained relevance scores, top 10 chunks
- **Speed:** Slower (pair-wise comparison), but most accurate
- **Benefit:** Best for final ranking

## Data Flow

```
input.txt
    ↓
[Split into 200-word chunks]
    ↓
[Preprocess (lowercase, remove special chars)]
    ↓
[BM25 Index & Query] → Top 30
    ↓
[Sentence Transformer Embeddings] → Top 20 by cosine sim
    ↓
[Cross-Encoder Pairs & Scoring] → Top 10 final results
```

## Output

- **BM25 Top 5:** Keyword relevance scores
- **Dense Top 5:** Cosine similarity scores (0–1, higher is better)
- **Cross-Encoder Top 10:** Relevance scores (typically 0–1, higher is better)

Example output format:
```
Top 10 chunks by cross-encoder score:
Chunk 4, Score: 0.8234
<chunk text preview>...

Chunk 0, Score: 0.7956
<chunk text preview>...
```

## Models Used

| Model | Purpose | Size | Speed |
|-------|---------|------|-------|
| `rank_bm25` | Sparse retrieval | N/A | Fast |
| `all-MiniLM-L6-v2` | Dense embeddings | ~80 MB | Moderate |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranking | ~150 MB | Slower |

All models are auto-downloaded from Hugging Face Hub on first run.

## Customization

### Change Chunk Size
In Cell 7, modify `chunk_size`:
```python
chunk_size = 300  # Instead of 200
```

### Change Query
In Cell 9, modify `query`:
```python
query = "your custom query here"
```

### Change Top-K Results
In Cell 9, modify the slice:
```python
top_indices = sorted(..., reverse=True)[:10]  # Instead of 5
```

### Use GPU
In Cell 14, the code auto-detects CUDA. Force it:
```python
device = 'cuda'  # or 'cpu'
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
```

## Requirements

- **rank_bm25** — BM25 sparse retrieval
- **sentence-transformers** — Dense embeddings & cross-encoders
- **torch** — PyTorch backend for transformers
- **nltk** — Text tokenization
- **pandas** — (optional) data manipulation
- **huggingface_hub** — (optional) manual model downloads

## Troubleshooting

### Model Not Found (404 Error)
Ensure you use a valid model ID from [Hugging Face Hub](https://huggingface.co/models):
- Cross-encoders: Search "cross-encoder"
- Sentence transformers: Search "sentence-transformers"

### Out of Memory
- Reduce `chunk_size` for fewer embeddings
- Reduce `batch_size` in `cross_encoder.predict()`
- Use CPU if GPU memory is limited

### Slow Performance
- Use GPU: Install `torch` with CUDA support
- Reduce number of chunks to process
- Pre-download models to avoid runtime downloads

## Future Enhancements

- [ ] Support multiple document formats (PDF, DOCX)
- [ ] Batch query processing
- [ ] Result export to JSON/CSV
- [ ] Web API wrapper
- [ ] Fine-tuning on domain-specific documents
- [ ] Hybrid loss function for end-to-end training

## License

[Specify your license, e.g., MIT, Apache 2.0]

## Contributing

Pull requests welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

## Contact

For questions or issues, please open a GitHub Issue.

---

**Built with:** [rank_bm25](https://github.com/dorianbrown/rank_bm25) | [Sentence Transformers](https://www.sbert.net/) | [Hugging Face](https://huggingface.co/)
