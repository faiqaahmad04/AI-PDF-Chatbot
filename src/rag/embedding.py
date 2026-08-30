from typing import List, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingPipeline:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        print(
            f"[INFO] Loaded embedding model: {model_name}"
        )

    def chunk_documents(
        self,
        documents: List[Any]
    ) -> List[Any]:

        # Split documents into smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

        chunks = splitter.split_documents(documents)

        print(
            f"[INFO] Split {len(documents)} documents "
            f"into {len(chunks)} chunks."
        )

        return chunks

    def embed_chunks(
        self,
        chunks: List[Any]
    ) -> np.ndarray:

        # Extract text from each chunk
        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        if not texts:
            raise ValueError(
                "No text chunks available for embedding."
            )

        print(
            f"[INFO] Generating embeddings "
            f"for {len(texts)} chunks..."
        )

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        embeddings = np.array(
            embeddings
        ).astype("float32")

        print(
            f"[INFO] Embeddings shape: {embeddings.shape}"
        )

        return embeddings