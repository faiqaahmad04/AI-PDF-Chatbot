# RAG (Retrieval-Augmented Generation) System

A Python-based Retrieval-Augmented Generation system that combines document retrieval with large language models to provide accurate, context-aware answers from your document collection.

## Features

- **Document Processing**: Support for PDF and text file ingestion
- **Vector Storage**: Dual storage support with FAISS and ChromaDB
- **Semantic Search**: Uses sentence transformers for intelligent document retrieval
- **LLM Integration**: Powered by Groq LLM for fast, accurate summarization
- **Jupyter Notebooks**: Includes example notebooks for data loading and PDF processing

## Requirements

- Python 3.13
- Dependencies listed in `requirements.txt`

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (create a `.env` file):
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Project Structure

```
RAG/
├── app.py                          # Main application entry point
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration
├── data/
│   ├── files/
│   │   ├── pdf/                   # PDF documents for processing
│   │   └── text_files/            # Text documents for processing
│   └── vector_store/              # ChromaDB vector storage
├── faiss_store/                    # FAISS vector index storage
├── notebook/
│   ├── document.ipynb              # Document loading notebook
│   └── pdf_loader.ipynb            # PDF processing notebook
└── src/
    └── rag/
        ├── __init__.py
        ├── data_loader.py           # Document loading utilities
        ├── embedding.py             # Embedding generation
        ├── search.py                # RAG search and retrieval logic
        └── vectorstore.py           # Vector store management
```

## Usage

### Basic Example

Run the main application:
```bash
python app.py
```

This will execute a sample query ("What is predictive maintenance?") and return an answer based on your document collection.

### Custom Query

Modify `app.py` to change the query:
```python
from src.rag.search import RAGSearch

rag_search = RAGSearch()

query = "Your custom question here"

answer = rag_search.search_and_summarize(
    query,
    top_k=3  # Number of documents to retrieve
)

print(answer)
```

### Jupyter Notebooks

- **pdf_loader.ipynb**: Load and process PDF files
- **document.ipynb**: Process and index documents

## Key Components

### Data Loader (`data_loader.py`)
Handles loading and processing of documents from PDF and text files.

### Embedding (`embedding.py`)
Generates embeddings using sentence transformers for semantic similarity.

### Search (`search.py`)
Core RAG logic that retrieves relevant documents and generates answers.

### Vector Store (`vectorstore.py`)
Manages vector storage and retrieval using FAISS and ChromaDB.

## Dependencies

- **langchain**: LLM orchestration framework
- **langchain-groq**: Groq LLM integration
- **sentence-transformers**: Semantic embedding generation
- **faiss-cpu**: Vector similarity search
- **chromadb**: Vector database
- **pypdf**: PDF reading
- **pymupdf**: PDF processing
- **python-dotenv**: Environment variable management

## Configuration

Hugging Face models are cached locally in the `.hf_cache` directory within the project folder to avoid repeated downloads.

## License

This project is available for use. Modify as needed for your use case.

