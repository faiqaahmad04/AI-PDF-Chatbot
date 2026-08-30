# AI PDF Chatbot

A simple Retrieval-Augmented Generation (RAG) app that lets you upload a PDF, ask questions about it, and get answers based on the document content.

## Features

- Upload a PDF from the browser
- Extract text from the PDF
- Split the content into chunks
- Generate embeddings and store them in FAISS
- Search the most relevant pages/chunks
- Ask questions using a Groq-powered LLM
- View the answer with source page numbers

## Tech Stack

- Python
- FastAPI
- FAISS
- LangChain
- Groq
- Hugging Face embeddings

## Setup

1. Clone the project.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_api_key_here
```

## Run the app

Start the server:

```bash
python -m uvicorn app:app --reload
```

Then open:

```text
http://localhost:8000
```

## Usage

1. Open the website in the browser.
2. Upload a PDF file.
3. Type a question in the input box.
4. The app will search the uploaded document and return an answer with relevant page references.

## Project Structure

```text
AI-PDF-Chatbot/
├── app.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── data/
│   └── uploads/
├── faiss_store/
├── static/
│   └── index.html
├── src/
│   └── rag/
│       ├── data_loader.py
│       ├── embedding.py
│       ├── search.py
│       ├── vectorstore.py
│       └── __init__.py

```

## Notes

- Hugging Face model files are cached locally in `.hf_cache` to avoid repeated downloads.
- The app currently supports one uploaded PDF at a time.

## License

This project is for learning and personal use.

