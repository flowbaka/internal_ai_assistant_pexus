import os

from dotenv import load_dotenv
from fastapi import  FastAPI, HTTPException, UploadFile         
from google import genai
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()


api_key = os.getenv("Gemini_api_key")

if not api_key:
    raise ValueError("Gemini_api_key environment variable is not set or found in .env.")

client = genai.Client(api_key=api_key)



app = FastAPI(
    title = "Internal AI Assistant", 
    description = "A privacy-aware AI assistant for internal use within an organization.",
)


class QuestionRequest(BaseModel):
    question: str


# RAG cannot properly search an entire document as one huge piece, so we divide it into smaller chunks. This is a simple implementation and can be improved with more advanced chunking strategies.

# function to split text into chunks of a specified size
def split_text(
        text:str, 
        chunk_size:int = 200,
        overlap:int = 30, 
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunck_words = words[start:end]
        chunk = " ".join(chunck_words)

        chunck.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks




@app.get("/")
def root():
    return {
        "message": "Welcome to the Internal AI Assistant API!"
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
            model = "gemini-3.7-flash",
            input = request.question
        )

        return {
            "question": request.question,
            "answer": interaction.output_text,
        }

    except Exception as e:
        print(f"Gemini API error: {e}")

        raise HTTPException(status_code=500, detail="Error processing the request with Gemini API.")    


    

@app.post("/document/extract")
async def extract_document(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are supported.") 


    try: 
        reader = PdfReader(file.file)

        extracted_text = []

        for page_number, page in enumerate(reader.pages, start = 1):
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
                status_code=400, 
                detail="No text could be extracted from the PDF file."
            )

        return {
            "filename": file.filename,
            "page_count": len(extracted_pages),
            "character_count": len(full_text),
            "text": full_text,
        }


    except HTTPException:
        raise

    except Exception as e:
        print(f"PDF extraction error: {e}")

        raise HTTPException(
            status_code = 400, 
            detail = "The PDF could not be processed. It may be encrypted, corrupted, or in an unsupported format."
        )