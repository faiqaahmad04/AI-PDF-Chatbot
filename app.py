import os

# --------------------------------------------------
# Hugging Face cache inside the project
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

HF_CACHE = os.path.join(
    PROJECT_DIR,
    ".hf_cache"
)

os.makedirs(
    HF_CACHE,
    exist_ok=True
)

os.environ["HF_HOME"] = HF_CACHE

os.environ["HF_HUB_CACHE"] = os.path.join(
    HF_CACHE,
    "hub"
)

os.environ["TRANSFORMERS_CACHE"] = os.path.join(
    HF_CACHE,
    "transformers"
)


# --------------------------------------------------
# Imports
# --------------------------------------------------

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from pydantic import BaseModel

from src.rag.data_loader import load_pdf
from src.rag.embedding import EmbeddingPipeline
from src.rag.vectorstore import FaissVectorStore

from langchain_groq import ChatGroq

from dotenv import load_dotenv


load_dotenv()


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="AI PDF Chatbot",
    description="Chat with your PDF using RAG",
    version="1.0.0"
)


# --------------------------------------------------
# Directories
# --------------------------------------------------

UPLOAD_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "uploads"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Global variables
# --------------------------------------------------

rag_system = None


# --------------------------------------------------
# Question model
# --------------------------------------------------

class Question(BaseModel):

    question: str


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.get("/")
def home():

    from fastapi.responses import FileResponse

    return FileResponse(
        os.path.join(
            PROJECT_DIR,
            "static",
            "index.html"
        )
    )


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    global rag_system

    # --------------------------------------------------
    # Check file type
    # --------------------------------------------------

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    try:

        # --------------------------------------------------
        # Save uploaded PDF
        # --------------------------------------------------

        pdf_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        content = await file.read()

        with open(
            pdf_path,
            "wb"
        ) as buffer:

            buffer.write(
                content
            )

        print(
            f"[INFO] PDF saved: {pdf_path}"
        )

        # --------------------------------------------------
        # Load PDF
        # --------------------------------------------------

        documents = load_pdf(
            pdf_path
        )

        if not documents:

            raise ValueError(
                "No text could be extracted from the PDF."
            )

        print(
            f"[INFO] Loaded {len(documents)} PDF pages."
        )

        # --------------------------------------------------
        # Create embeddings
        # --------------------------------------------------

        embedding_pipeline = EmbeddingPipeline()

        chunks = embedding_pipeline.chunk_documents(
            documents
        )

        embeddings = embedding_pipeline.embed_chunks(
            chunks
        )

        # --------------------------------------------------
        # Create fresh FAISS store
        # --------------------------------------------------

        vectorstore = FaissVectorStore(
            persist_dir="faiss_store"
        )

        metadatas = []

        for chunk in chunks:

            page = chunk.metadata.get(
                "page"
            )

            metadatas.append(
                {
                    "text": chunk.page_content,
                    "page": page
                }
            )

        vectorstore.add_embeddings(
            embeddings,
            metadatas
        )

        vectorstore.save()

        # --------------------------------------------------
        # Create Groq LLM
        # --------------------------------------------------

        groq_api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not groq_api_key:

            raise ValueError(
                "GROQ_API_KEY not found in .env"
            )

        llm = ChatGroq(
            api_key=groq_api_key,
            model="openai/gpt-oss-120b"
        )

        # --------------------------------------------------
        # Create RAG object
        # --------------------------------------------------

        class PDFRAG:

            def __init__(
                self,
                vectorstore,
                llm
            ):

                self.vectorstore = vectorstore
                self.llm = llm

            def ask(
                self,
                query,
                top_k=5
            ):

                results = self.vectorstore.query(
                    query,
                    top_k=top_k
                )

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
                        texts.append(
                            text
                        )

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

                context = "\n\n".join(
                    texts
                )

                if not context:

                    return {
                        "answer": (
                            "I could not find the "
                            "answer in the uploaded PDF."
                        ),
                        "sources": []
                    }

                prompt = f"""
You are an AI assistant that answers questions about an uploaded PDF.

Answer the user's question using ONLY the information in the context.

Rules:

- Do not make up information.
- If the answer is not in the context, say:
  "I could not find the answer in the uploaded PDF."
- Use plain text only.
- Do not use Markdown.
- Do not use # symbols.
- Do not use ** or *.
- Do not create tables.
- Use short paragraphs.
- Use numbered lists or simple "-" bullet points when useful.
- Do not repeat the question.
- Do not mention the RAG system.

Question:
{query}

Context:
{context}

Answer:
"""

                response = self.llm.invoke(
                    prompt
                )

                answer = response.content

                answer = answer.replace(
                    "**",
                    ""
                )

                answer = answer.replace(
                    "```",
                    ""
                )

                unique_pages = sorted(
                    set(pages)
                )

                return {
                    "answer": answer.strip(),
                    "sources": unique_pages
                }

        rag_system = PDFRAG(
            vectorstore,
            llm
        )

        print(
            "[INFO] PDF RAG system ready."
        )

        return {
            "message": (
                "PDF uploaded and processed successfully."
            ),
            "filename": file.filename,
            "pages": len(documents),
            "chunks": len(chunks)
        }

    except Exception as e:

        print(
            f"[ERROR] PDF processing failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )


# --------------------------------------------------
# Ask question
# --------------------------------------------------

@app.post("/ask")
async def ask_question(
    request: Question
):

    global rag_system

    if rag_system is None:

        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first."
        )

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        result = rag_system.ask(
            request.question,
            top_k=5
        )

        return result

    except Exception as e:

        print(
            f"[ERROR] Question answering failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )