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
# REVIEW ENDPOINT (new)
# =============================================================================
class ReviewRequest(BaseModel):
    code: str
    language: str = "python"
    focus_areas: list[str] | None = None  # optional, for backward compat


REVIEW_SYSTEM_PROMPT = """You are an expert code reviewer and security analyst.

Analyze the given code thoroughly.

You MUST respond with ONLY a valid JSON object using this exact structure:

{
  "summary": "2-4 sentence overall assessment",
  "critical_issues": [
    {"line": 4, "message": "Null pointer risk"},
    {"line": 10, "message": "Division by zero possible"}
  ],
  "security_issues": [
    {"line": 15, "message": "SQL Injection vulnerability"}
  ],
  "performance_improvements": [
    {"line": 22, "message": "Inefficient loop"}
  ],
  "best_practices": [
    {"line": 3, "message": "Missing input validation"}
  ],
  "score": 85
}

Rules:
- line must be actual line number from the code.
- message must be short and actionable.
- If no issues in category use [].
- Output ONLY JSON."""

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
        parts.append(
            "## Critical Issues\n"
            + "\n".join(
                f"- Line {item.get('line')}: {item.get('message')}"
                if isinstance(item, dict)
                else f"- {item}"
                for item in data["critical_issues"]
            )
        )
    if data.get("security_issues"):
        parts.append(
            "## Security Issues\n"
            + "\n".join(
                f"- Line {item.get('line')}: {item.get('message')}"
                if isinstance(item, dict)
                else f"- {item}"
                for item in data["security_issues"]
            )
        )
    if data.get("performance_improvements"):
        parts.append(
            "## Performance\n"
            + "\n".join(
                f"- Line {item.get('line')}: {item.get('message')}"
                if isinstance(item, dict)
                else f"- {item}"
                for item in data["performance_improvements"]
            )
        )
    if data.get("best_practices"):
        parts.append(
            "## Best Practices\n"
            + "\n".join(
                f"- Line {item.get('line')}: {item.get('message')}"
                if isinstance(item, dict)
                else f"- {item}"
                for item in data["best_practices"]
            )
        )
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
    code = (request.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="Code is required")

    language = _normalize_language(request.language)
    user_content = f"Review this {language} code with line numbers:\n```{language}\n{code}\n```"

    try:
        content = await _call_groq(
            [
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=2048,
        )

        data = _parse_review_json(content)

        # 🔥 Collect all errors into a single list and also group by line
        all_errors: list[dict] = []
        line_errors: dict[int, list[dict]] = {}

        for category in [
            "critical_issues",
            "security_issues",
            "performance_improvements",
            "best_practices",
        ]:
            for item in data.get(category, []):
                if isinstance(item, dict) and "line" in item and "message" in item:
                    err = {
                        "line": int(item["line"]),
                        "message": item["message"],
                        "category": category,
                    }
                    all_errors.append(err)
                    line_errors.setdefault(int(item["line"]), []).append(err)

        # Shape that is easy for frontend to highlight lines in red
        line_errors_compact = {
            str(line): [e["message"] for e in errs] for line, errs in line_errors.items()
        }

        return {
            "summary": data.get("summary", ""),
            "critical_issues": data.get("critical_issues", []),
            "security_issues": data.get("security_issues", []),
            "performance_improvements": data.get("performance_improvements", []),
            "best_practices": data.get("best_practices", []),
            "score": int(data.get("score", 0)),
            "errors": all_errors,  # full objects with line/message/category
            "line_errors": line_errors_compact,  # { "3": ["msg1", "msg2"], ... }
        }

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


def _normalize_language(language: str) -> str:
    lang = (language or "").strip().lower()
    aliases = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "c++": "cpp",
        "c#": "csharp",
        "golang": "go",
    }
    return aliases.get(lang, lang or "plaintext")


def _looks_like_language(code: str, language: str) -> bool:
    """
    Very lightweight heuristic to detect obvious language mismatches.
    This is intentionally conservative: if unsure, return True.
    """
    lang = _normalize_language(language)
    txt = (code or "").strip()
    if not txt:
        return True

    # Markup/data languages
    if lang in {"html", "xml", "svg"}:
        return "<" in txt and ">" in txt
    if lang in {"css", "scss", "sass", "less"}:
        return "{" in txt and "}" in txt and ":" in txt
    if lang == "json":
        return txt[0] in "{[" and txt[-1] in "}]"
    if lang in {"yaml", "yml"}:
        return "\n" in txt and (":" in txt)

    # Common programming languages (spot-check)
    lowered = txt.lower()
    if lang == "python":
        return bool(re.search(r"^\s*(def |class |import |from )", txt, flags=re.M))
    if lang in {"javascript", "typescript"}:
        return any(k in lowered for k in ("function", "const ", "let ", "var ", "=>", "export ", "import "))
    if lang in {"java", "kotlin"}:
        return any(k in lowered for k in ("class ", "public ", "private ", "package ", "import "))
    if lang in {"c", "cpp", "csharp"}:
        return any(k in lowered for k in ("#include", "using ", "namespace", "class ", "int main", "public "))
    if lang == "go":
        return any(k in lowered for k in ("package ", "func ", "import "))
    if lang == "rust":
        return any(k in lowered for k in ("fn ", "use ", "mod ", "crate"))

    # Unknown language: don't block.
    return True



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

    language = _normalize_language(request.language)
    strict_lang_guard = (
        "CRITICAL: You MUST keep the code in the SAME LANGUAGE as the input. "
        "Do NOT translate it into another language. "
        "If the input is HTML/CSS/JSON/XML, return valid optimized HTML/CSS/JSON/XML only."
    )
    user_content = (
        f"{strict_lang_guard}\n\n"
        f"Rewrite this {language} code:\n```{language}\n{code}\n```"
    )
    try:
        content = await _call_groq(
            [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT + "\n\n" + strict_lang_guard},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        opt, explanation = _extract_rewrite_response(content, code)
        if not _looks_like_language(opt, language) and _looks_like_language(code, language):
            # Model returned code in the wrong language (e.g., Python for HTML).
            # Prefer returning the original code rather than misleading output.
            opt = code
            explanation = (
                explanation
                or "The model returned output in a different language than requested. Please retry."
            )
        return {
            "optimized_code": opt,
            "explanation": explanation,
            "rewritten_code": opt,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
