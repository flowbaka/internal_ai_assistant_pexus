import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from google import genai
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from uuid import uuid4

from privacy import mask_sensitive_data



import chromadb




# Load variables from the .env file
load_dotenv()

api_key = os.getenv("Gemini_api_key")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in the .env file"
    )


# Create Gemini client
client = genai.Client(api_key=api_key)


# Load the local Hugging Face embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# database and collection setup
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

document_collection = chroma_client.get_or_create_collection(
    name="internal_documents"
)

# Create FastAPI application
app = FastAPI(
    title="Privacy-Aware Internal Document Assistant",
    description="An AI assistant for internal document question answering",
)


# Request model for POST /ask
class QuestionRequest(BaseModel):
    question: str


def split_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 30,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero"
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be zero or more and smaller than chunk size"
        )

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks

def create_embeddings(
    chunks: list[str],
) -> list[list[float]]:
    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
    )

    return embeddings.tolist()

def store_document(
    filename: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> str:
    document_id = str(uuid4())

    chunk_ids = [
        f"{document_id}_chunk_{index}"
        for index in range(len(chunks))
    ]

    metadata = [
        {
            "document_id": document_id,
            "filename": filename,
            "chunk_number": index,
        }
        for index in range(len(chunks))
    ]

    document_collection.add(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
    )

    return document_id


# create a reusable retriever function for document search
def retrieve_relevant_chunks(
    question: str,
    number_of_results: int = 3,
) -> list[dict]:
    stored_chunk_count = document_collection.count()

    if stored_chunk_count == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents have been uploaded",
        )

    question_embedding = create_embeddings([question])[0]

    results = document_collection.query(
        query_embeddings=[question_embedding],
        n_results=min(
            number_of_results,
            stored_chunk_count,
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "text": document,
            "filename": metadata["filename"],
            "chunk_number": metadata["chunk_number"],
            "distance": round(float(distance), 4),
        })

    return matches

def retrieve_relevant_chunks(
    question: str,
    number_of_results: int = 3,
) -> list[dict]:
    stored_chunk_count = document_collection.count()

    if stored_chunk_count == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents have been uploaded",
        )

    question_embedding = create_embeddings([question])[0]

    results = document_collection.query(
        query_embeddings=[question_embedding],
        n_results=min(
            number_of_results,
            stored_chunk_count,
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "text": document,
            "filename": metadata["filename"],
            "chunk_number": metadata["chunk_number"],
            "distance": round(float(distance), 4),
        })

    return matches

@app.get("/")
def home():
    return {
        "message": "Internal AI Assistant is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        interaction = client.interactions.create(
            model="gemini-3.7-flash",
            input=request.question,
        )

        return {
            "question": request.question,
            "answer": interaction.output_text,
        }

    except Exception as error:
        print(f"Gemini API error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Gemini could not generate an answer",
        ) from error


@app.post("/documents/extract")
def extract_document(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed",
        )

    try:
        reader = PdfReader(file.file)

        extracted_pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            page_text = page.extract_text() or ""

            extracted_pages.append({
                "page": page_number,
                "text": page_text,
            })

        full_text = "\n".join(
            page["text"] for page in extracted_pages
        )

        if not full_text.strip():
            raise HTTPException(
                status_code=422,
                detail="The PDF contains no readable text",
            )

        chunks = split_text(full_text)
        embeddings = create_embeddings(chunks)

        document_id = store_document(
            filename=file.filename or "unnamed.pdf",
            chunks=chunks,
            embeddings=embeddings,
        )

        return {
            "filename": file.filename,
            "page_count": len(extracted_pages),
            "character_count": len(full_text),
            "chunk_count": len(chunks),
            "chunk_preview": chunks[:2],
            "embedding_count": len(embeddings),
            "embedding_dimensions": len(embeddings[0]),
            "embedding_preview": embeddings[0][:5],
            "document_id": document_id,
            "stored_chunks": len(chunks),
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Document processing error: {error}")

        raise HTTPException(
            status_code=400,
            detail="The PDF could not be processed",
        ) from error


@app.post("/documents/search")
def search_documents(request: QuestionRequest):
    matches = retrieve_relevant_chunks(
        request.question
    )

    return {
        "question": request.question,
        "matches": matches,
    }


@app.post("/documents/ask")
def ask_document(request: QuestionRequest):
    matches = retrieve_relevant_chunks(
        request.question
    )

    masked_question, redaction_counts = (
        mask_sensitive_data(request.question)
    )

    safe_matches = []

    for match in matches:
        masked_text, chunk_redactions = (
            mask_sensitive_data(match["text"])
        )

        for category, count in chunk_redactions.items():
            redaction_counts[category] += count

        safe_match = match.copy()
        safe_match["text"] = masked_text

        safe_matches.append(safe_match)

    context_sections = []

    for index, match in enumerate(safe_matches, start=1):
        source = (
            f"Source {index}\n"
            f"Filename: {match['filename']}\n"
            f"Chunk: {match['chunk_number']}\n"
            f"Content: {match['text']}"
        )

        context_sections.append(source)

    context = "\n\n".join(context_sections)

    prompt = f"""
You are an internal document assistant.

Answer using only the supplied document context.

If the answer is unavailable, respond:
"I could not find that information in the uploaded documents."

Do not invent information.

Question:
{masked_question}

Document context:
{context}
"""

    try:
        interaction = client.interactions.create(
            model="gemini-3.7-flash",
            input=prompt,
        )

        return {
            "question": masked_question,
            "answer": interaction.output_text,
            "sources": safe_matches,
            "privacy": {
                "redactions": redaction_counts,
                "total_redactions": sum(
                    redaction_counts.values()
                ),
            },
        }

    except Exception as error:
        print(f"Gemini RAG error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Gemini could not answer the question",
        ) from error