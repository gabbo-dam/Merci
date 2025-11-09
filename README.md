# Merci (Demo)
FastAPI backend that turns natural requests into Jira tickets.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill values
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
