from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str) -> List:
    """
    Load a PDF file and return LangChain documents.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    print(f"[INFO] Loading PDF: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    print(f"[INFO] Loaded {len(documents)} pages.")

    return documents