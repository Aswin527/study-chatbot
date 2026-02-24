from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class Question(BaseModel):
    message: str

@app.post("/chat")
async def chat(question: Question):
    try:
        response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "You are a smart homework helper chatbot for school students from grade 6 to 10. Keep answers simple and structured."}
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"text": question.message}
                ],
            },
        ],
        config={
            "temperature": 0.2,
            "top_p": 0.95,
        },
        )

        return {"reply": response.text}

    except Exception as e:
        return {"error": str(e)}