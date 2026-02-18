"""
AI Code Review & Rewrite Agent - Backend
Chat, Review, Rewrite APIs using Groq (Llama 3.3 70B)
"""
import json
import os
import re       
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import secrets

# Load .env from backend folder or project root
env_path = Path(__file__).parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# =============================================================================
# CONFIGURATION (shared across all endpoints)
# =============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
DEV_MODE = os.getenv("DEV_MODE", "false").lower() in {"1", "true", "yes"}

app = FastAPI(title="AI Code Review API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080", "http://127.0.0.1:8080",
        "http://localhost:5500", "http://127.0.0.1:5500",
        "https://aicodere.web.app",
    ],
    allow_origin_regex=r"https://.*\.(netlify\.app|web\.app)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# SHARED GROQ CLIENT (no duplication)
# =============================================================================
async def _call_groq(
    messages: list[dict],
    temperature: float = 0.4,
    max_tokens: int = 2048,
) -> str:
    """Shared Groq API caller. Used by chat, review, and rewrite endpoints."""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not set. Add it to .env file.",
        )
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
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
        resp.raise_for_status()
        data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="No response from Groq")
    return choices[0].get("message", {}).get("content", "") or ""


# =============================================================================
# CHATBOT (existing - do not modify)
# =============================================================================
class ChatRequest(BaseModel):
    message: str
    code: str = ""


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
async def health():
    """Health check - verify backend is running."""
    return {"status": "ok", "groq_configured": bool(GROQ_API_KEY)}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with AI about code - uses Groq Llama 3.3 70B"""
    message = (request.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    system_prompt = """You are a helpful AI assistant for a code review and rewrite tool. 
Answer questions about programming, code quality, debugging, best practices, and refactoring.
Be concise and practical. If the user shares code, analyze it and give actionable advice."""

    user_content = message
    if request.code:
        user_content = f"User's code:\n```\n{request.code}\n```\n\nUser's question: {message}"

    try:
        content = await _call_groq(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return ChatResponse(reply=content or "No response.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=e.response.text or str(e),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# =============================================================================
# AUTH ENDPOINTS (minimal stubs for frontend compatibility)
# =============================================================================
class SignupRequest(BaseModel):
    email: str
    password: str
    name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class SendCodeRequest(BaseModel):
    email: str


class VerifyRequest(BaseModel):
    email: str
    code: str


VERIFICATION_STORE: dict[str, dict] = {}

def _gen_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


@app.post("/api/signup")
@app.post("/signup")
async def signup(request: SignupRequest):
    email = (request.email or "").strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email is required")
    name = (request.name or email.split("@")[0]).strip()
    code = _gen_code()
    VERIFICATION_STORE[email.lower()] = {
        "code": code,
        "expires": datetime.utcnow() + timedelta(minutes=10),
        "attempts": 0,
        "verified": False,
    }
    return JSONResponse(
        {
            "ok": True,
            "message": "Account created",
            "user": {"email": email, "name": name, "id": f"user_{abs(hash(email))%10_000_000}"},
            "token": "demo-token",
            "verification": {"sent": True, "method": "code", **({"dev_code": code} if DEV_MODE else {})},
        }
    )


@app.post("/api/login")
@app.post("/login")
async def login(request: LoginRequest):
    email = (request.email or "").strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email is required")
    return {
        "ok": True,
        "message": "Signed in",
        "user": {"email": email, "name": email.split("@")[0], "id": f"user_{abs(hash(email))%10_000_000}"},
        "token": "demo-token",
    }


@app.post("/api/send-code")
@app.post("/api/resend-code")
async def send_code(request: SendCodeRequest):
    email = (request.email or "").strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email is required")
    code = _gen_code()
    VERIFICATION_STORE[email.lower()] = {
        "code": code,
        "expires": datetime.utcnow() + timedelta(minutes=10),
        "attempts": 0,
        "verified": False,
    }
    return {
        "ok": True,
        "message": "Verification code sent",
        **({"dev_code": code} if DEV_MODE else {}),
    }


@app.post("/api/verify-email")
@app.post("/api/verify")
async def verify_email(request: VerifyRequest):
    email = (request.email or "").strip().lower()
    code = (request.code or "").strip()
    if not email or "@" not in email or not code:
        raise HTTPException(status_code=400, detail="Email and code are required")
    entry = VERIFICATION_STORE.get(email)
    if not entry:
        raise HTTPException(status_code=404, detail="No code sent")
    if entry.get("expires") and datetime.utcnow() > entry["expires"]:
        raise HTTPException(status_code=400, detail="Code expired")
    entry["attempts"] = entry.get("attempts", 0) + 1
    if entry["code"] != code:
        raise HTTPException(status_code=400, detail="Invalid code")
    entry["verified"] = True
    return {"ok": True, "message": "Email verified"}


# =============================================================================
# REVIEW ENDPOINT (new)
# =============================================================================
class ReviewRequest(BaseModel):
    code: str
    language: str = "python"
    focus_areas: list[str] | None = None  # optional, for backward compat


REVIEW_SYSTEM_PROMPT = """You are an expert code reviewer and security analyst. Analyze the given code thoroughly.

You MUST respond with ONLY a valid JSON object, no markdown, no code blocks, no extra text. Use this exact structure:

{
  "summary": "2-4 sentence overall assessment of the code quality",
  "critical_issues": ["issue1", "issue2"],
  "security_issues": ["security1", "security2"],
  "performance_improvements": ["improvement1", "improvement2"],
  "best_practices": ["practice1", "practice2"],
  "score": 85
}

Rules:
- score: integer 0-100 (code quality)
- Each array: list of strings, 0-5 items each. Be specific and actionable.
- Check for: bugs, security vulnerabilities (injection, XSS, hardcoded secrets, weak crypto), performance bottlenecks, memory leaks, error handling gaps, input validation, code style, maintainability.
- If no issues in a category, use empty array [].
- Output ONLY the JSON object, nothing else."""


def _parse_review_json(raw: str) -> dict:
    """Parse LLM response to structured review. Handles markdown fences."""
    txt = raw.strip()
    if txt.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", txt)
        if match:
            txt = match.group(1).strip()
        else:
            txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(txt)


def _review_to_backward_compat(data: dict) -> dict:
    """Build review, keyImprovements, explanation for existing frontend."""
    parts = []
    if data.get("critical_issues"):
        parts.append("## Critical Issues\n" + "\n".join(f"- {x}" for x in data["critical_issues"]))
    if data.get("security_issues"):
        parts.append("## Security Issues\n" + "\n".join(f"- {x}" for x in data["security_issues"]))
    if data.get("performance_improvements"):
        parts.append("## Performance\n" + "\n".join(f"- {x}" for x in data["performance_improvements"]))
    if data.get("best_practices"):
        parts.append("## Best Practices\n" + "\n".join(f"- {x}" for x in data["best_practices"]))
    parts.append("## Summary\n" + (data.get("summary") or ""))
    review_md = "\n\n".join(parts)
    improvements = [
        {"title": "Critical", "description": "; ".join(data.get("critical_issues", [])[:3]) or "None"}
    ]
    if data.get("security_issues"):
        improvements.append({"title": "Security", "description": "; ".join(data["security_issues"][:3])})
    if data.get("performance_improvements"):
        improvements.append({"title": "Performance", "description": "; ".join(data["performance_improvements"][:3])})
    return {
        "review": review_md,
        "keyImprovements": improvements,
        "explanation": data.get("summary", ""),
    }


@app.post("/api/review")
@app.post("/review")
async def review(request: ReviewRequest):
    """Review code for quality, security, performance, best practices."""
    code = (request.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="Code is required")

    user_content = f"Review this {request.language} code:\n```{request.language}\n{code}\n```"
    try:
        content = await _call_groq(
            [
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        data = _parse_review_json(content)
        result = {
            "summary": data.get("summary", ""),
            "critical_issues": data.get("critical_issues", []),
            "security_issues": data.get("security_issues", []),
            "performance_improvements": data.get("performance_improvements", []),
            "best_practices": data.get("best_practices", []),
            "score": int(data.get("score", 0)) if data.get("score") is not None else 0,
        }
        compat = _review_to_backward_compat(result)
        result.update(compat)
        return result
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Failed to parse review response: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# =============================================================================
# REWRITE ENDPOINT (new)
# =============================================================================
class RewriteRequest(BaseModel):
    code: str
    language: str = "python"
    focus_areas: list[str] | None = None  # optional, for backward compat


REWRITE_SYSTEM_PROMPT = """You are an expert programmer. Rewrite the given code to improve it.

Output format - use EXACTLY this structure (valid JSON, no markdown):
{"optimized_code": "YOUR_REWRITTEN_CODE_HERE", "explanation": "Brief explanation of changes"}

Rules:
- optimized_code: the complete rewritten code. Use \\n for newlines. Escape quotes as \\".
- Preserve exact functionality. Fix typos, improve style, add error handling if missing.
- Follow best practices for the given language.
- Output ONLY the JSON object, nothing before or after."""


def _strip_code_fence(text: str) -> str:
    """Remove ```lang ... ``` wrapper if present."""
    txt = (text or "").strip()
    if not txt:
        return ""
    if txt.startswith("```"):
        match = re.search(r"```(?:\w*)\s*([\s\S]*?)```", txt)
        if match:
            return match.group(1).strip()
        lines = txt.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return txt


def _extract_rewrite_response(content: str, original_code: str) -> tuple[str, str]:
    """Parse LLM response - may be JSON or raw code. Returns (code, explanation)."""
    content = (content or "").strip()
    if not content:
        return original_code, ""
    # Try JSON first
    try:
        txt = content
        if txt.startswith("```"):
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", txt)
            if match:
                txt = match.group(1).strip()
            else:
                txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(txt)
        opt = data.get("optimized_code", data.get("rewritten_code", ""))
        if opt:
            opt = opt.replace("\\n", "\n").replace('\\"', '"')
            return opt, data.get("explanation", "")
    except json.JSONDecodeError:
        pass
    # Fallback: treat as raw code
    return _strip_code_fence(content), "Code optimized for readability and best practices."


@app.post("/api/rewrite")
@app.post("/rewrite")
async def rewrite(request: RewriteRequest):
    """Rewrite/optimize code with explanation."""
    code = (request.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="Code is required")

    user_content = f"Rewrite this {request.language} code:\n```{request.language}\n{code}\n```"
    try:
        content = await _call_groq(
            [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        opt, explanation = _extract_rewrite_response(content, code)
        return {
            "optimized_code": opt,
            "explanation": explanation,
            "rewritten_code": opt,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
