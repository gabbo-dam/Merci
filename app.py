import os, re, hmac, hashlib, time, json
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from dotenv import load_dotenv
import httpx

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = (os.getenv("SLACK_SIGNING_SECRET") or "").encode()

JIRA_BASE = os.environ["JIRA_BASE"]
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
JIRA_PROJECT = os.getenv("JIRA_PROJECT", "MRCI")

app = FastAPI(title="Merci Backend", version="0.1")

ACTION_REGEX = re.compile(r"\b(please|can you|could you|we need to|let's|todo|fix|update|add|remove)\b", re.I)
BUG_REGEX = re.compile(r"\b(bug|fix|broken|prod|outage|incident)\b", re.I)
PRIORITY_REGEX = re.compile(r"\b(prod|outage|urgent|sev|p0)\b", re.I)
DUE_REGEX = re.compile(r"\bby (eod|monday|tomorrow|next week)\b", re.I)

def analyze(text: str, mentioned: bool=False):
    # tiny but effective heuristics for demo
    conf = 0.5
    if ACTION_REGEX.search(text): conf += 0.2
    if PRIORITY_REGEX.search(text): conf += 0.2
    if DUE_REGEX.search(text): conf += 0.1
    if mentioned: conf += 0.1
    conf = min(conf, 1.0)

    issue_type = "Bug" if BUG_REGEX.search(text) else "Task"
    priority = "Highest" if conf >= 0.9 else ("High" if conf >= 0.8 else "Medium")
    title = (text.strip().splitlines()[0])[:80] or "Auto-created by Merci"
    description = f"Source text:\n{text.strip()}\n\nLabels: merci, autocaptured"
    return dict(confidence=conf, title=title, description=description, issue_type=issue_type, priority=priority)

def verify_slack(request: Request, raw_body: bytes):
    if not SLACK_SIGNING_SECRET:
        return  # allow when Slack not configured
    ts = request.headers.get("X-Slack-Request-Timestamp", "0")
    if abs(time.time() - int(ts)) > 60 * 5:
        raise HTTPException(401, "stale request")
    sig_basestring = f"v0:{ts}:{raw_body.decode()}".encode()
    my_sig = "v0=" + hmac.new(SLACK_SIGNING_SECRET, sig_basestring, hashlib.sha256).hexdigest()
    req_sig = request.headers.get("X-Slack-Signature", "")
    if not hmac.compare_digest(my_sig, req_sig):
        raise HTTPException(401, "invalid signature")

async def jira_create(summary: str, description: str, issue_type: str = "Task") -> str:
    payload = {
        "fields": {
            "project": {"key": JIRA_PROJECT},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type},
            "labels": ["merci", "autocaptured"],
        }
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{JIRA_BASE}/rest/api/3/issue",
            json=payload,
            auth=(JIRA_EMAIL, JIRA_API_TOKEN),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"Jira error: {r.text}")
    return r.json()["key"]

async def slack_post(channel: str, thread_ts: str, text: str):
    if not SLACK_BOT_TOKEN: return
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            data={"channel": channel, "thread_ts": thread_ts, "text": text},
        )

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/try", response_class=HTMLResponse)
async def try_form():
    return """
    <html><head><title>Merci Demo</title></head>
      <body style="font-family:system-ui;max-width:720px;margin:40px auto">
        <h1>Merci — Create a Jira Ticket</h1>
        <p>Paste a natural request (e.g., <i>please update onboarding docs before Monday</i>)</p>
        <form method="POST" action="/simulate">
          <textarea name="text" rows="6" style="width:100%;"></textarea><br/>
          <button type="submit" style="margin-top:10px;padding:8px 14px">Create Ticket</button>
        </form>
      </body>
    </html>
    """

@app.post("/simulate")
async def simulate(text: str = Form(...)):
    a = analyze(text)
    # auto-create at >= 0.8, else create anyway (this is a demo form)
    key = await jira_create(a["title"], a["description"], a["issue_type"])
    jira_url = f"{JIRA_BASE}/browse/{key}"
    return JSONResponse({"created": key, "url": jira_url, "analysis": a})

@app.post("/slack/command")
async def slack_command(request: Request):
    raw = await request.body()
    verify_slack(request, raw)
    form = dict([tuple(kv.split("=")) for kv in raw.decode().split("&") if "=" in kv])
    user_text = re.sub(r"\+", " ", form.get("text", ""))
    a = analyze(user_text)
    if a["confidence"] >= 0.8:
        key = await jira_create(a["title"], a["description"], a["issue_type"])
        return {"response_type": "in_channel", "text": f"✅ Created *{key}*: {a['title']}"}
    else:
        return {"response_type": "ephemeral", "text": f"Suggestion: {a['issue_type']} (conf {a['confidence']:.2f}). Re-run `/merci {user_text}` to confirm."}

@app.post("/slack/events")
async def slack_events(request: Request):
    raw = await request.body()
    verify_slack(request, raw)
    data = await request.json()
    if data.get("type") == "url_verification":
        return PlainTextResponse(data["challenge"])
    event = data.get("event", {})
    if not event or event.get("subtype"):  # ignore bot edits, joins, etc.
        return {"ok": True}
    text = event.get("text", "")
    channel = event.get("channel")
    ts = event.get("ts")
    a = analyze(text, mentioned=False)
    if a["confidence"] >= 0.9:
        key = await jira_create(a["title"], a["description"], a["issue_type"])
        await slack_post(channel, ts, f"✅ Created *{key}* from this message.")
    elif a["confidence"] >= 0.7:
        await slack_post(channel, ts, f"Create a {a['issue_type']}? (conf {a['confidence']:.2f}) Use `/merci {text}` to confirm.")
    return {"ok": True}
