from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Annotated
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 30 * 1024 * 1024
MAX_FILES = 20
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

app = FastAPI(
    title="TalentLens 2040 — Phase 2 Contextual Matching API",
    description="Local resume extraction with optional AI-assisted contextual matching.",
    version="0.3.0",
)
app.mount("/assets", StaticFiles(directory=BASE_DIR), name="assets")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/health")
def health():
    configured = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "status": "ok",
        "mode": "ai-configured" if configured else "keyword-fallback",
        "ai_provider": "OpenAI API" if configured else None,
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini") if configured else None,
        "note": "AI mode sends extracted resume text to the configured API provider. Without a key, matching remains local.",
    }


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_text(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return normalize(content.decode(encoding))
            except UnicodeDecodeError:
                continue
        return ""
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content), strict=False)
            if reader.is_encrypted:
                try:
                    if not reader.decrypt(""):
                        raise HTTPException(422, f"{filename}: encrypted PDFs are not supported.")
                except Exception as exc:
                    if isinstance(exc, HTTPException):
                        raise
                    raise HTTPException(422, f"{filename}: encrypted PDFs are not supported.") from exc
            return normalize("\n".join((page.extract_text() or "") for page in reader.pages[:100]))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, f"{filename}: could not extract PDF text. Try a text-based PDF.") from exc
    if ext == ".docx":
        try:
            from docx import Document
            document = Document(io.BytesIO(content))
            chunks = [p.text for p in document.paragraphs]
            for table in document.tables:
                for row in table.rows:
                    chunks.append(" ".join(cell.text for cell in row.cells))
            return normalize("\n".join(chunks))
        except Exception as exc:
            raise HTTPException(422, f"{filename}: could not read DOCX. Confirm the file is a valid DOCX.") from exc
    raise HTTPException(415, f"{filename}: unsupported format. Use PDF, DOCX, or TXT. Legacy .doc is not supported.")


SKILL_ALIASES = {
    "python": ["python", "pandas", "numpy", "scikit-learn"],
    "sql": ["sql", "postgresql", "mysql", "sqlite", "t-sql"],
    "excel": ["excel", "spreadsheets", "pivot table", "vlookup", "xlookup"],
    "statistics": ["statistics", "statistical", "hypothesis testing", "regression"],
    "data visualization": ["data visualization", "visualization", "dashboard", "tableau", "power bi", "matplotlib", "plotly"],
    "power bi": ["power bi", "powerbi", "dax"],
    "tableau": ["tableau"],
    "data cleaning": ["data cleaning", "data cleansing", "preprocessing", "data wrangling"],
    "communication": ["communication", "presented findings", "stakeholder", "storytelling"],
    "responsible ai": ["responsible ai", "ai ethics", "model governance", "fairness in ai"],
    "machine learning": ["machine learning", "scikit-learn", "classification", "clustering", "neural network"],
    "critical thinking": ["critical thinking", "problem solving", "root cause analysis"],
    "research": ["research", "literature review", "survey"],
    "javascript": ["javascript", "typescript", "node.js"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "tailwind"],
    "java": ["java", "spring boot"],
    "cloud": ["aws", "azure", "gcp", "cloud computing"],
    "project management": ["project management", "agile", "scrum", "sprint planning"],
    "leadership": ["leadership", "mentored", "team lead", "managed a team"],
    "accessibility": ["accessibility", "wcag", "screen reader", "inclusive design"],
    "security": ["cybersecurity", "security", "owasp", "penetration testing"],
    "generative ai": ["generative ai", "large language model", "llm", "prompt engineering", "rag"],
    "data storytelling": ["data storytelling", "communicated insights", "business insights"],
}


def find_skills(text: str) -> list[str]:
    lower = f" {text.lower()} "
    return [
        skill for skill, aliases in SKILL_ALIASES.items()
        if any(re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", lower) for alias in aliases)
    ]


def make_evidence(text: str, skill: str) -> str:
    lower = text.lower()
    for alias in SKILL_ALIASES.get(skill, [skill]):
        match = re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", lower)
        if match:
            return "…" + text[max(0, match.start()-75):min(len(text), match.end()+100)].strip() + "…"
    return "No matching term found in extracted text."


def keyword_score(job_title: str, job_description: str, filename: str, text: str, selected_skills: list[str] | None = None) -> dict:
    requirements = selected_skills or find_skills(job_title + " " + job_description)
    resume_skills = find_skills(text)
    matched = [s for s in requirements if s in resume_skills]
    missing = [s for s in requirements if s not in resume_skills]
    score = round(100 * len(matched) / len(requirements)) if requirements else 0
    evidence = [{"criterion": s, "excerpt": make_evidence(text, s)} for s in matched[:8]]
    warning = "No extractable text found; scanned PDF OCR is not included." if not text else (
        "Very little text was extracted. Check the source document." if len(text) < 80 else None
    )
    return {
        "name": Path(filename).stem[:100] or "Uploaded candidate",
        "role": "Uploaded resume", "score": score, "skills": matched[:8] or ["No matched terms"],
        "status": "Review", "experience": "Not inferred", "education": "Not inferred",
        "evidence": f"Matched {len(matched)} of {len(requirements)} detected role-skill terms. This is keyword overlap, not semantic understanding.",
        "gaps": "Terms not found: " + (", ".join(missing) if missing else "none of the detected terms") + ". Missing keywords do not prove missing skills.",
        "matched_skills": matched, "missing_skills": missing, "evidence_items": evidence,
        "extracted_characters": len(text), "warning": warning, "filename": Path(filename).name,
        "scoring_method": "deterministic-keyword-overlap-v1",
    }


def ai_score(job_title: str, job_description: str, filename: str, text: str, selected_skills: list[str] | None = None) -> dict:
    """Optional OpenAI API scoring. Sends extracted resume text externally only when explicitly configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("AI key not configured")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    # Bound prompt size to keep the local prototype predictable.
    resume_excerpt = text[:18000]
    prompt = {
        "job_title": job_title,
        "job_requirements": job_description[:10000],
        "recruiter_selected_required_skills": (selected_skills or [])[:40],
        "resume_text": resume_excerpt,
        "instruction": (
            "Evaluate only job-related evidence. Prioritize recruiter_selected_required_skills as the explicit criteria when provided. Never infer or use protected characteristics. "
            "Return JSON with: score (integer 0-100 indicating coverage of explicit job criteria, not probability), "
            "matched_skills (array of strings), missing_or_unclear_skills (array of strings), "
            "evidence (array of objects with criterion and short exact excerpt from resume_text), "
            "summary (string), limitations (string). Do not recommend automatic rejection. "
            "If evidence is absent or ambiguous, say so. Do not invent facts."
        ),
    }
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You assist human recruiters with evidence-grounded job criteria comparison. You do not make hiring decisions. Output valid JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }).encode("utf-8")
    request = Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"AI provider returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach AI provider: {exc}") from exc
    except (KeyError, ValueError, IndexError) as exc:
        raise RuntimeError("AI provider returned an unexpected response.") from exc

    score = max(0, min(100, int(parsed.get("score", 0))))
    matched = [str(s)[:100] for s in parsed.get("matched_skills", [])[:20]]
    missing = [str(s)[:100] for s in parsed.get("missing_or_unclear_skills", [])[:20]]
    ev = []
    for item in parsed.get("evidence", [])[:10]:
        criterion = str(item.get("criterion", ""))[:100]
        excerpt = str(item.get("excerpt", ""))[:350]
        # Only display excerpts that really occur in the extracted source text.
        if excerpt and excerpt.strip("… ") in text:
            ev.append({"criterion": criterion, "excerpt": excerpt})
        else:
            ev.append({"criterion": criterion, "excerpt": "AI supplied excerpt could not be verified exactly; check original resume."})
    return {
        "name": Path(filename).stem[:100] or "Uploaded candidate",
        "role": "Uploaded resume", "score": score, "skills": matched[:8] or ["No matched skills returned"],
        "status": "Review", "experience": "Not inferred", "education": "Not inferred",
        "evidence": str(parsed.get("summary", "AI-assisted criteria comparison; human review required."))[:1200],
        "gaps": str(parsed.get("limitations", ""))[:800] + (" Missing/unclear: " + ", ".join(missing) if missing else ""),
        "matched_skills": matched, "missing_skills": missing, "evidence_items": ev,
        "extracted_characters": len(text), "warning": "AI-generated assessment may be wrong. Verify all evidence against the original resume.",
        "filename": Path(filename).name, "scoring_method": "openai-contextual-assessment",
    }


@app.post("/api/screen")
async def screen_resumes(
    job_title: Annotated[str, Form(min_length=1, max_length=120)],
    job_description: Annotated[str, Form(min_length=10, max_length=12000)],
    selected_skills: Annotated[str, Form()] = "[]",
    resumes: Annotated[list[UploadFile], File()] = [],
):
    try:
        parsed_skills = json.loads(selected_skills or "[]")
        if not isinstance(parsed_skills, list):
            raise ValueError("selected_skills must be a JSON list")
        parsed_skills = [str(s).strip()[:80] for s in parsed_skills if str(s).strip()][:40]
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(422, "Selected skills must be a valid JSON list.") from exc
    if not resumes:
        raise HTTPException(400, "Upload at least one resume.")
    if len(resumes) > MAX_FILES:
        raise HTTPException(413, f"Maximum {MAX_FILES} resumes per scan.")
    use_ai = bool(os.getenv("OPENAI_API_KEY"))
    results, total_bytes = [], 0
    for upload in resumes:
        filename = Path(upload.filename or "resume.txt").name
        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"{filename}: unsupported format. Use PDF, DOCX, or TXT.")
        content = await upload.read(MAX_FILE_BYTES + 1)
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(413, f"{filename}: file exceeds 10 MB.")
        total_bytes += len(content)
        if total_bytes > MAX_TOTAL_BYTES:
            raise HTTPException(413, "Combined upload exceeds 30 MB.")
        text = extract_text(filename, content)
        if use_ai:
            try:
                item = ai_score(job_title.strip(), job_description.strip(), filename, text, parsed_skills or None)
            except RuntimeError as exc:
                # Do not silently switch to local matching when the user configured AI;
                # return the issue so the user knows which mode ran.
                raise HTTPException(502, f"AI matching failed: {exc}") from exc
        else:
            item = keyword_score(job_title.strip(), job_description.strip(), filename, text, parsed_skills or None)
        results.append(item)
        await upload.close()
        del content, text
    results.sort(key=lambda item: item["score"], reverse=True)
    return {
        "job_title": job_title.strip(), "selected_skills": parsed_skills, "candidate_count": len(results), "candidates": results,
        "mode": "OpenAI contextual matching" if use_ai else "local keyword fallback",
        "notice": "Scores are advisory only, may be inaccurate, and must not be used for automated rejection. Human review is required.",
    }
