import os

# Store Hugging Face models inside the project folder
os.environ["HF_HOME"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".hf_cache"
)

from src.rag.search import RAGSearch


if __name__ == "__main__":

    rag_search = RAGSearch()

    query = "What is predictive maintenance?"

    answer = rag_search.search_and_summarize(
        query,
        top_k=3
    )

    print("\nAnswer:")
    print(answer)