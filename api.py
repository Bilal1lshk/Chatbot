import os
import json

from fastapi import FastAPI, HTTPException
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


class ChatRequest(BaseModel):
    message: str


class EmailRequest(BaseModel):
    lead_name: str
    lead_email: str
    company: str | None = None
    status: str | None = None
    source: str | None = None
    estimated_value: float | None = None
    notes: str | None = None
    goal: str
    tone: str = "professional but warm"
    sender_name: str | None = None
    extra_instructions: str | None = None


@app.get("/")
def home():
    return {
        "message": "LeadWise AI API is running"
    }


@app.post("/ai/email")
def generate_email(request: EmailRequest):
    lead_facts = "\n".join(filter(None, [
        f"Name: {request.lead_name}",
        f"Company: {request.company}" if request.company else None,
        f"Pipeline status: {request.status}" if request.status else None,
        f"Lead source: {request.source}" if request.source else None,
        f"Estimated deal value: ${request.estimated_value}" if request.estimated_value else None,
        f"Notes: {request.notes}" if request.notes else None,
    ]))

    chat_session = client.chats.create(
        model="gemini-3.6-flash",
        config={
            "system_instruction": """
You are an email-writing assistant for LeadWise, a CRM platform.

Write short, personalized sales outreach emails based only on the lead
details and goal given to you. Never invent facts about the lead that
weren't provided.

Rules:
- Keep the body under 150 words.
- No generic filler like "I hope this email finds you well."
- Reference something specific from the lead details to make it feel
  personalized, not templated.
- Do not include a signature or sign-off name — that gets appended
  separately by the app.
- Respond with ONLY valid JSON, no markdown fences, no commentary,
  in exactly this shape:
{"subject": "...", "body": "..."}
The "body" field should use \\n for line breaks.
"""
        }
    )

    prompt = f"""Lead details:
{lead_facts}

Goal of this email: {request.goal}
Tone: {request.tone}
{f"Sender name: {request.sender_name}" if request.sender_name else ""}
{f"Additional instructions: {request.extra_instructions}" if request.extra_instructions else ""}
"""

    try:
        response = chat_session.send_message(prompt)
        raw = response.text.strip()

        # Strip accidental markdown fences
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        parsed = json.loads(cleaned)

        if "subject" not in parsed or "body" not in parsed:
            raise ValueError("Missing subject or body in AI response")

        return {
            "success": True,
            "subject": parsed["subject"],
            "body": parsed["body"],
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="AI response wasn't valid JSON.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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