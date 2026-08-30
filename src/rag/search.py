import os
from dotenv import load_dotenv

from src.rag.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-120b"
    ):

        # Load vector store
        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model
        )

        faiss_path = os.path.join(
            persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            persist_dir,
            "metadata.pkl"
        )

        # Build vector store if it doesn't exist
        if not (
            os.path.exists(faiss_path)
            and os.path.exists(meta_path)
        ):

            from src.rag.data_loader import load_all_documents

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)

        else:

            self.vectorstore.load()

        # Get Groq API key
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env"
            )

        # Initialize Groq
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=llm_model
        )

        print(
            f"[INFO] Groq LLM initialized: {llm_model}"
        )

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:

        # Retrieve relevant documents
        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        # Extract text from metadata
        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        # RAG prompt
        prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the context provided below.

If the answer cannot be found in the context,
say that the information is not available
in the provided documents.

Question:
{query}

Context:
{context}

Answer:
"""

        # Generate answer
        response = self.llm.invoke(prompt)

        return response.content


if __name__ == "__main__":

    rag_search = RAGSearch()

    query = "What is predictive maintenance?"

    answer = rag_search.search_and_summarize(
        query,
        top_k=3
    )

    print("\nAnswer:")
    print(answer)