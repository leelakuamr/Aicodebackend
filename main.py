"""
AI Code Review & Rewrite Agent - Backend
Chat, Review, Rewrite APIs using Groq
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx


# =============================================================================
# LOAD ENV
# =============================================================================

env_path = Path(__file__).parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).parent.parent / ".env"

load_dotenv(env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="AI Code Review API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://aicodere.web.app",
    ],
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# DEBUG + HEALTH
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "groq_configured": bool(GROQ_API_KEY),
    }


@app.get("/debug-env")
async def debug_env():
    return {
        "groq_key_exists": bool(GROQ_API_KEY),
        "groq_key_length": len(GROQ_API_KEY or ""),
    }


# =============================================================================
# SHARED GROQ CALLER
# =============================================================================

async def call_groq(messages: List[dict], temperature=0.4, max_tokens=2048) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid GROQ API Key")

        response.raise_for_status()

        data = response.json()

    choices = data.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="No response from Groq")

    return choices[0]["message"]["content"]


# =============================================================================
# CHAT
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    code: Optional[str] = ""


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message required")

    user_prompt = request.message
    if request.code:
        user_prompt = f"Code:\n{request.code}\n\nQuestion:\n{request.message}"

    system_prompt = "You are a helpful programming assistant."

    content = await call_groq(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=1024,
    )

    return {"reply": content}


# =============================================================================
# REVIEW
# =============================================================================

class ReviewRequest(BaseModel):
    code: str
    language: str = "python"


REVIEW_PROMPT = """
You are an expert code reviewer.

Return ONLY valid JSON:

{
 "summary": "...",
 "critical_issues": [],
 "security_issues": [],
 "performance_improvements": [],
 "best_practices": [],
 "score": 85
}
"""


@app.post("/api/review")
async def review(request: ReviewRequest):
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code required")

    content = await call_groq(
        [
            {"role": "system", "content": REVIEW_PROMPT},
            {
                "role": "user",
                "content": f"Review this {request.language} code:\n{request.code}",
            },
        ],
        temperature=0.3,
    )

    try:
        # Remove markdown if model adds it
        if content.startswith("```"):
            content = re.sub(r"```(?:json)?|```", "", content).strip()

        data = json.loads(content)
    except Exception:
        raise HTTPException(status_code=502, detail="Failed to parse AI response")

    return data


# =============================================================================
# REWRITE
# =============================================================================

class RewriteRequest(BaseModel):
    code: str
    language: str = "python"


REWRITE_PROMPT = """
Rewrite the code to improve readability and best practices.

Return ONLY JSON:
{
 "optimized_code": "...",
 "explanation": "..."
}
"""


@app.post("/api/rewrite")
async def rewrite(request: RewriteRequest):
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code required")

    content = await call_groq(
        [
            {"role": "system", "content": REWRITE_PROMPT},
            {
                "role": "user",
                "content": f"Rewrite this {request.language} code:\n{request.code}",
            },
        ],
        temperature=0.3,
    )

    try:
        if content.startswith("```"):
            content = re.sub(r"```(?:json)?|```", "", content).strip()

        data = json.loads(content)
        return data
    except Exception:
        return {
            "optimized_code": content,
            "explanation": "Model returned raw code instead of JSON",
        }
