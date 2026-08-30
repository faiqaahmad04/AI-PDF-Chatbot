import os
import faiss
import numpy as np
import pickle
from typing import List, Any

from sentence_transformers import SentenceTransformer

from src.rag.embedding import EmbeddingPipeline


class FaissVectorStore:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):

        self.persist_dir = persist_dir

        os.makedirs(
            self.persist_dir,
            exist_ok=True
        )

        self.index = None
        self.metadata = []

        self.embedding_model = embedding_model

        # Load the same embedding model used during indexing
        self.model = SentenceTransformer(
            embedding_model
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(
            f"[INFO] Loaded embedding model: "
            f"{embedding_model}"
        )

    # --------------------------------------------------
    # Build FAISS index from documents
    # --------------------------------------------------

    def build_from_documents(
        self,
        documents: List[Any]
    ):

        print(
            f"[INFO] Building vector store from "
            f"{len(documents)} pages..."
        )

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Split documents into chunks
        chunks = emb_pipe.chunk_documents(
            documents
        )

        # Generate embeddings
        embeddings = emb_pipe.embed_chunks(
            chunks
        )

        # --------------------------------------------------
        # Store text + PDF page information
        # --------------------------------------------------

        metadatas = []

        for chunk in chunks:

            page = chunk.metadata.get(
                "page"
            )

            source = chunk.metadata.get(
                "source",
                ""
            )

            metadatas.append(
                {
                    "text": chunk.page_content,
                    "source": source,
                    "page": page
                }
            )

        # Add embeddings to FAISS
        self.add_embeddings(
            np.array(
                embeddings
            ).astype("float32"),
            metadatas
        )

        # Save index and metadata
        self.save()

        print(
            f"[INFO] Vector store built with "
            f"{len(chunks)} chunks."
        )

    # --------------------------------------------------
    # Add embeddings
    # --------------------------------------------------

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Any] = None
    ):

        if embeddings is None or len(embeddings) == 0:

            raise ValueError(
                "No embeddings were provided."
            )

        # Create FAISS index
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        # Add vectors
        self.index.add(
            embeddings
        )

        # Store metadata
        if metadatas is not None:

            self.metadata = metadatas

        print(
            f"[INFO] Added "
            f"{embeddings.shape[0]} vectors."
        )

    # --------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------

    def save(self):

        if self.index is None:

            raise ValueError(
                "Cannot save an empty FAISS index."
            )

        faiss_path = os.path.join(
            self.persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            self.persist_dir,
            "metadata.pkl"
        )

        # Save FAISS index
        faiss.write_index(
            self.index,
            faiss_path
        )

        # Save metadata
        with open(
            meta_path,
            "wb"
        ) as f:

            pickle.dump(
                self.metadata,
                f
            )

        print(
            f"[INFO] Saved vector store to "
            f"{self.persist_dir}"
        )

    # --------------------------------------------------
    # Load existing FAISS index
    # --------------------------------------------------

    def load(self):

        faiss_path = os.path.join(
            self.persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            self.persist_dir,
            "metadata.pkl"
        )

        if not os.path.exists(
            faiss_path
        ):

            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{faiss_path}"
            )

        if not os.path.exists(
            meta_path
        ):

            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{meta_path}"
            )

        # Load FAISS index
        self.index = faiss.read_index(
            faiss_path
        )

        # Load metadata
        with open(
            meta_path,
            "rb"
        ) as f:

            self.metadata = pickle.load(
                f
            )

        print(
            f"[INFO] Loaded vector store from "
            f"{self.persist_dir}"
        )

        print(
            f"[INFO] Loaded "
            f"{len(self.metadata)} metadata entries."
        )

    # --------------------------------------------------
    # Search FAISS
    # --------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5
    ):

        if self.index is None:

            raise ValueError(
                "FAISS index has not been loaded."
            )

        if not query_text.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        # Make sure top_k does not exceed
        # the number of vectors in the index
        top_k = min(
            top_k,
            self.index.ntotal
        )

        if top_k <= 0:

            return []

        print(
            f"[INFO] Querying vector store for: "
            f"'{query_text}'"
        )

        # Generate embedding for the question
        query_embedding = self.model.encode(
            [query_text]
        ).astype("float32")

        # Search FAISS
        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx, distance in zip(
            indices[0],
            distances[0]
        ):

            idx = int(idx)

            # Ignore invalid FAISS indexes
            if (
                idx < 0
                or idx >= len(self.metadata)
            ):

                continue

            results.append(
                {
                    "index": idx,
                    "distance": float(distance),
                    "metadata": self.metadata[idx]
                }
            )

        print(
            f"[INFO] Retrieved "
            f"{len(results)} relevant chunks."
        )

        return results


# ------------------------------------------------------
# Optional direct test
# ------------------------------------------------------

if __name__ == "__main__":

    store = FaissVectorStore(
        persist_dir="faiss_store"
    )

    store.load()

    results = store.query(
        "What is predictive maintenance?",
        top_k=3
    )

    for result in results:

        print(
            "\nPage:",
            result["metadata"].get("page")
        )

        print(
            "Distance:",
            result["distance"]
        )

        print(
            "Text:",
            result["metadata"].get("text", "")[:500]
        )