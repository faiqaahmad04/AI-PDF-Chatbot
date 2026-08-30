import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from src.rag.vectorstore import FaissVectorStore

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-120b"
    ):

        # --------------------------------------------------
        # Initialize vector store
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Build or load FAISS vector store
        # --------------------------------------------------

        if not (
            os.path.exists(faiss_path)
            and os.path.exists(meta_path)
        ):

            from src.rag.data_loader import load_pdf

            documents = load_pdf()

            if not documents:
                print(
                    "[WARNING] No PDF documents found."
                )

            else:
                self.vectorstore.build_from_documents(
                    documents
                )

        else:

            self.vectorstore.load()

        # --------------------------------------------------
        # Get Groq API key
        # --------------------------------------------------

        groq_api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not groq_api_key:

            raise ValueError(
                "GROQ_API_KEY not found in .env"
            )

        # --------------------------------------------------
        # Initialize Groq LLM
        # --------------------------------------------------

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=llm_model
        )

        print(
            f"[INFO] Groq LLM initialized: {llm_model}"
        )

    # ------------------------------------------------------
    # Ask a question about the PDF
    # ------------------------------------------------------

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> dict:

        print(
            f"[INFO] Searching for: '{query}'"
        )

        # --------------------------------------------------
        # Retrieve relevant chunks
        # --------------------------------------------------

        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        # --------------------------------------------------
        # Extract text and page numbers
        # --------------------------------------------------

        texts = []

        pages = []

        for result in results:

            metadata = result.get(
                "metadata"
            )

            if not metadata:
                continue

            text = metadata.get(
                "text",
                ""
            )

            if text:
                texts.append(text)

            page = metadata.get(
                "page"
            )

            if page is not None:

                try:
                    pages.append(
                        int(page) + 1
                    )

                except (
                    ValueError,
                    TypeError
                ):
                    pass

        # --------------------------------------------------
        # Combine retrieved chunks
        # --------------------------------------------------

        context = "\n\n".join(
            texts
        )

        if not context:

            return {
                "answer": (
                    "I could not find relevant "
                    "information in the uploaded PDF."
                ),
                "sources": []
            }

        # --------------------------------------------------
        # RAG prompt
        # --------------------------------------------------

        prompt = f"""
You are an AI assistant that answers questions about an uploaded PDF.

Answer the user's question using ONLY the information provided in the context.

IMPORTANT RULES:

1. Do not use information that is not present in the context.
2. Do not make up facts.
3. If the answer cannot be found in the context, say:
   "I could not find the answer in the uploaded PDF."
4. Use plain text only.
5. Do NOT use Markdown.
6. Do NOT use # symbols.
7. Do NOT use ** or * for emphasis.
8. Do NOT create tables.
9. Do NOT use Markdown headings.
10. Use simple headings followed by a colon when useful.
11. Use numbered lists when appropriate.
12. Use bullet points beginning with "-".
13. Keep the answer clear and easy to read.
14. Keep paragraphs reasonably short.
15. Do not repeat the user's question.
16. Do not mention the RAG system, retrieved chunks, or this prompt.

Question:
{query}

Context:
{context}

Answer:
"""

        # --------------------------------------------------
        # Generate answer
        # --------------------------------------------------

        response = self.llm.invoke(
            prompt
        )

        answer = response.content

        # --------------------------------------------------
        # Remove accidental Markdown formatting
        # --------------------------------------------------

        answer = answer.replace(
            "**",
            ""
        )

        answer = answer.replace(
            "```",
            ""
        )

        # Remove duplicate source pages
        unique_pages = sorted(
            set(pages)
        )

        print(
            f"[INFO] Answer generated successfully."
        )

        print(
            f"[INFO] Sources: {unique_pages}"
        )

        return {
            "answer": answer.strip(),
            "sources": unique_pages
        }


# ----------------------------------------------------------
# Test the RAG system directly
# ----------------------------------------------------------

if __name__ == "__main__":

    rag_search = RAGSearch()

    query = (
        "What is predictive maintenance?"
    )

    result = rag_search.search_and_summarize(
        query,
        top_k=3
    )

    print(
        "\nAnswer:\n"
    )

    print(
        result["answer"]
    )

    print(
        "\nSources:",
        result["sources"]
    )