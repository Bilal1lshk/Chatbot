import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


# Allow Next.js frontend to call FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
         "https://leadmanagmentsystem-nu.vercel.app",

    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Gemini client
client = genai.Client(
    api_key=os.getenv("gemini_api_key")
)


# Request body from Next.js
class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "message": "LeadWise AI API is running"
    }


@app.post("/ai/chat")
def chat(request: ChatRequest):

    chat_session = client.chats.create(
        model="gemini-3.6-flash",
        config={
            "system_instruction": """
You are the AI assistant for LeadWise.

About LeadWise:
LeadWise is a modern CRM platform designed to help businesses manage
their leads, sales pipeline, follow-ups, tasks, and analytics.

LeadWise features include:
- Lead management
- Sales pipeline management
- Follow-up management
- Task management
- Sales analytics
- Dashboard for tracking business activity
- AI-powered assistance

Your role:
- Talk to users about LeadWise.
- Explain LeadWise features.
- Help users understand how to use the platform.
- Answer questions about the website and its features.
- Be friendly, helpful, and professional.
- Keep answers clear and reasonably short.
- Do not claim that LeadWise has a feature if it is not mentioned above.
"""
        }
    )

    response = chat_session.send_message(
        request.message
    )

    return {
        "success": True,
        "response": response.text
    }