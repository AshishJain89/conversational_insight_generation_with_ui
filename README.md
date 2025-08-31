# NL → SQL (Simplest) — Groq + LangChain + SQLite + FastAPI + React

## Run
1) Backend
```bash
cd backend
python -m venv .venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit .env and paste your GROQ_API_KEY
uvicorn main:app --host 127.0.0.1 --port 8000
```
Check: http://127.0.0.1:8000/health

2) Frontend (no Node)
```bash
cd ..
python -m http.server 5173 --bind 127.0.0.1 --directory frontend
```
Open http://127.0.0.1:5173

Ask a question and see the **SQL printed in backend terminal** under `[GENERATED SQL]`.
