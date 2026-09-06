import io
import os
import uuid

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from privacy import mask_sensitive_data


# -------------------------------------------------
# APPLICATION CONFIGURATION
# -------------------------------------------------

load_dotenv()

app = FastAPI(
    title="Privacy-Aware Internal Document Assistant",
    description=(
        "Upload PDF documents, search them semantically, "
        "and ask document-grounded questions."
    ),
    version="1.0.0",
)


# -------------------------------------------------
# GEMINI CONFIGURATION
# -------------------------------------------------

api_key = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("Gemini_api_key")
)

gemini_client = (
    genai.Client(api_key=api_key)
    if api_key
    else None
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash",
)


# -------------------------------------------------
# EMBEDDING MODEL
# -------------------------------------------------

# This model runs locally on your computer.
# Each text chunk becomes an embedding containing
# 384 numerical values.
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------------------------------
# CHROMADB CONFIGURATION
# -------------------------------------------------

# PersistentClient saves the vector database
# inside the chroma_db folder.
chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

document_collection = (
    chroma_client.get_or_create_collection(
        name="internal_documents"
    )
)


# -------------------------------------------------
# PYDANTIC REQUEST MODELS
# -------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=500,
    )


class DocumentQuestionRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=500,
    )

    document_id: str = Field(
        min_length=1,
    )


# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------

def split_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[str]:
    """
    Split a long document into smaller overlapping chunks.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []
    start = 0

    while start < len(text):
        end = min(
            start + chunk_size,
            len(text),
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Stop when we reach the end of the document.
        if end == len(text):
            break

        # Move backwards slightly so the next chunk
        # shares some text with the previous chunk.
        start = end - overlap

    return chunks


def create_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Convert text into numerical embedding vectors.
    """

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
    )

    return embeddings.tolist()


def store_document(
    filename: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> str:
    """
    Store document chunks and embeddings in ChromaDB.
    """

    document_id = str(uuid.uuid4())

    chunk_ids = []
    metadata = []

    for index in range(len(chunks)):
        chunk_ids.append(
            f"{document_id}-chunk-{index}"
        )

        metadata.append(
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
            }
        )

    document_collection.add(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
    )

    return document_id


def retrieve_relevant_chunks(
    question: str,
    document_id: str,
    number_of_results: int = 3,
) -> list[dict]:
    """
    Search only inside the selected document.
    """

    if document_collection.count() == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents have been uploaded.",
        )

    question_embedding = create_embeddings(
        [question]
    )[0]

    results = document_collection.query(
        query_embeddings=[question_embedding],
        n_results=min(
            number_of_results,
            document_collection.count(),
        ),
        where={
            "document_id": document_id,
        },
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    if not documents:
        raise HTTPException(
            status_code=404,
            detail=(
                "The selected document was not found. "
                "Please upload it again."
            ),
        )

    matches = []

    for text, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        matches.append(
            {
                "text": text,
                "filename": metadata.get(
                    "filename",
                    "Unknown document",
                ),
                "document_id": metadata.get(
                    "document_id"
                ),
                "chunk_index": metadata.get(
                    "chunk_index"
                ),
                "distance": float(distance),
            }
        )

    return matches


def combine_redaction_counts(
    total_counts: dict,
    new_counts: dict,
) -> None:
    """
    Add new masking counts to the total counts.
    """

    for category, count in new_counts.items():
        total_counts[category] = (
            total_counts.get(category, 0) + count
        )


def generate_gemini_answer(
    prompt: str,
) -> str:
    """
    Send a prompt to Gemini and return its answer.
    """

    if gemini_client is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini API key is not configured. "
                "Add GEMINI_API_KEY to your .env file."
            ),
        )

    try:
        response = (
            gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response.",
            )

        return response.text

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini could not process the request: "
                f"{str(error)}"
            ),
        ) from error


# -------------------------------------------------
# BASIC ENDPOINTS
# -------------------------------------------------

@app.get("/")
def root():
    return {
        "message": (
            "Privacy-Aware Internal Document "
            "Assistant API"
        )
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "embedding_model": (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        "stored_chunks": (
            document_collection.count()
        ),
    }


@app.post("/ask")
def ask_general_ai(
    request: QuestionRequest,
):
    """
    Ask Gemini a general question without document RAG.
    """

    masked_question, redaction_counts = (
        mask_sensitive_data(request.question)
    )

    answer = generate_gemini_answer(
        masked_question
    )

    return {
        "question": request.question,
        "answer": answer,
        "redaction_counts": redaction_counts,
    }


# -------------------------------------------------
# PDF UPLOAD ENDPOINT
# -------------------------------------------------

@app.post("/documents/extract")
async def extract_document(
    file: UploadFile = File(...),
):
    """
    Extract, chunk, embed, and store an uploaded PDF.
    """

    filename = file.filename or "uploaded_document.pdf"

    is_pdf = (
        file.content_type == "application/pdf"
        or filename.lower().endswith(".pdf")
    )

    if not is_pdf:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    try:
        file_contents = await file.read()

        if not file_contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty.",
            )

        # Limit the uploaded file to approximately 10 MB.
        if len(file_contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=(
                    "The PDF is too large. "
                    "Maximum size is 10 MB."
                ),
            )

        pdf_reader = PdfReader(
            io.BytesIO(file_contents)
        )

        extracted_pages = []

        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""

            if page_text.strip():
                extracted_pages.append(
                    page_text.strip()
                )

        full_text = "\n\n".join(
            extracted_pages
        )

        if not full_text.strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    "No readable text was found. "
                    "The PDF may contain scanned images."
                ),
            )

        chunks = split_text(full_text)

        embeddings = create_embeddings(chunks)

        document_id = store_document(
            filename=filename,
            chunks=chunks,
            embeddings=embeddings,
        )

        return {
            "filename": filename,
            "page_count": len(pdf_reader.pages),
            "character_count": len(full_text),
            "chunk_count": len(chunks),
            "chunk_preview": chunks[:2],
            "embedding_count": len(embeddings),
            "embedding_dimensions": (
                len(embeddings[0])
                if embeddings
                else 0
            ),
            "embedding_preview": (
                embeddings[0][:5]
                if embeddings
                else []
            ),
            "document_id": document_id,
            "stored_chunks": len(chunks),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "The PDF could not be processed: "
                f"{str(error)}"
            ),
        ) from error

    finally:
        await file.close()


# -------------------------------------------------
# SEMANTIC SEARCH ENDPOINT
# -------------------------------------------------

@app.post("/documents/search")
def search_documents(
    request: DocumentQuestionRequest,
):
    """
    Retrieve chunks from one selected document.
    """

    matches = retrieve_relevant_chunks(
        question=request.question,
        document_id=request.document_id,
    )

    return {
        "question": request.question,
        "document_id": request.document_id,
        "match_count": len(matches),
        "matches": matches,
    }


# -------------------------------------------------
# DOCUMENT RAG ENDPOINT
# -------------------------------------------------

@app.post("/documents/ask")
def ask_document(
    request: DocumentQuestionRequest,
):
    """
    Retrieve relevant chunks and ask Gemini to answer
    using only those chunks.
    """

    matches = retrieve_relevant_chunks(
        question=request.question,
        document_id=request.document_id,
    )

    # Mask sensitive information in the question.
    masked_question, redaction_counts = (
        mask_sensitive_data(request.question)
    )

    safe_matches = []

    # Mask sensitive information inside every retrieved chunk.
    for match in matches:
        masked_text, chunk_redactions = (
            mask_sensitive_data(match["text"])
        )

        combine_redaction_counts(
            redaction_counts,
            chunk_redactions,
        )

        safe_match = match.copy()
        safe_match["text"] = masked_text

        safe_matches.append(safe_match)

    context_sections = []

    for index, match in enumerate(
        safe_matches,
        start=1,
    ):
        source = (
            f"Source {index} "
            f"(chunk {match['chunk_index']})"
        )

        context_sections.append(
            f"{source}:\n{match['text']}"
        )

    context = "\n\n".join(
        context_sections
    )

    prompt = f"""
You are an internal document assistant.

Answer the question using only the document context
provided below.

If the answer is not present in the context, say:
"I could not find that information in the document."

Do not invent information.
Keep the answer clear and concise.

DOCUMENT CONTEXT:
{context}

QUESTION:
{masked_question}

ANSWER:
"""

    answer = generate_gemini_answer(prompt)

    return {
        "question": request.question,
        "document_id": request.document_id,
        "answer": answer,
        "sources": safe_matches,
        "redaction_counts": redaction_counts,
    }
