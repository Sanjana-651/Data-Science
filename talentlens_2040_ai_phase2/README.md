# TalentLens 2040 — Phase 2 AI Matching (local development)

## Start the app on port 8002

Open PowerShell or CMD in this folder:

```powershell
python -m pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8002
```

Open:
- Website: http://127.0.0.1:8002
- Health/mode: http://127.0.0.1:8002/api/health
- API docs: http://127.0.0.1:8002/docs

If port 8002 is also occupied, stop the existing process or select a different free port.

## Optional contextual AI mode

By default, resume matching runs locally using an editable keyword/alias list. To enable contextual matching with the OpenAI API, configure the key **on your own computer** before starting Uvicorn. Do not paste API keys into chat or commit them to source control.

PowerShell (current terminal session only):
```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY_HERE"
$env:OPENAI_MODEL="gpt-4.1-mini"
uvicorn app:app --host 127.0.0.1 --port 8002
```

CMD (current terminal session only):
```cmd
set OPENAI_API_KEY=YOUR_API_KEY_HERE
set OPENAI_MODEL=gpt-4.1-mini
uvicorn app:app --host 127.0.0.1 --port 8002
```

If the key is set, the app sends extracted resume text and job description to the configured API provider. Review the provider's current data-use terms and obtain applicant consent/appropriate authorization before processing real resumes. API use may incur charges. If no key is set, it uses local keyword fallback and does not send resume text externally.

## What this phase adds
- Optional OpenAI API contextual comparison
- Prompt asks for job-related criteria only, uncertainty, and evidence excerpts
- Basic verification that AI-provided evidence excerpts appear in extracted resume text
- Clear mode reporting: AI or local keyword fallback
- Local PDF, DOCX, TXT extraction
- File type, size, and count limits
- No file persistence by the screening endpoint

## Limitations / safeguards
This is a learning prototype, not a validated hiring system. Scores are not probabilities or validated measures of job performance. Models can hallucinate, miss evidence, or be biased; review the original resume and all evidence manually. Do not automate rejection. Scanned PDFs need OCR, legacy `.doc` is unsupported, and production authentication, malware scanning, retention/deletion, auditing, monitoring, and security hardening are not included. Do not expose this local server to the public internet.
