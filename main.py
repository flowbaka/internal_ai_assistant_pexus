import os

from dotenv import load_dotenv
from fastapi import  FastAPI, HTTPException
from google import genai
from pydantic import BaseModel

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


    


