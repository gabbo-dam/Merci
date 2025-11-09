import os, re, hmac, hashlib, time, json, asyncio, random
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from dotenv import load_dotenv
import httpx

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = (os.getenv("SLACK_SIGNING_SECRET") or "").encode()

JIRA_BASE = os.environ["JIRA_BASE"]
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
JIRA_PROJECT = os.getenv("JIRA_PROJECT", "MRCI")

# ---- AI config ----
AI_ENABLE = os.getenv("AI_ENABLE", "true").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
AI_TIMEOUT = float(os.getenv("AI_TIMEOUT", "8.0"))
AI_MAX_RETRIES = int(os.getenv("AI_MAX_RETRIES", "2"))

app = FastAPI(title="Merci Backend", version="0.3-ai-guarded")

# ---- Thresholds & guards ----
MIN_CONF_AUTO = float(os.getenv("MIN_CONF_AUTO", "0.80"))
MIN_CONF_SUGGEST = float(os.getenv("MIN_CONF_SUGGEST", "0.70"))
REQUIRE_SIGNAL_FOR_CREATE = os.getenv("REQUIRE_SIGNAL_FOR_CREATE", "true").lower() == "true"

ACTION_REGEX = re.compile(r"\b(please|can you|could you|we need to|let's|todo|fix|update|add|remove)\b", re.I)
BUG_REGEX = re.compile(r"\b(bug|fix|broken|prod|outage|incident)\b", re.I)
PRIORITY_REGEX = re.compile(r"\b(prod|outage|urgent|sev|p0)\b", re.I)
DUE_REGEX = re.compile(r"\bby (eod|monday|tomorrow|next week)\b", re.I)
SALUTATION_REGEX = re.compile(
    r"^\s*(hi|hello|hey|good (morning|afternoon|evening|day)|thanks|thank you|ty|yo)\b[^\n]*$",
    re.I,
)

def is_smalltalk(text: str) -> bool:
    if SALUTATION_REGEX.match(text.strip()):
        return True
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) <= 3 and not (ACTION_REGEX.search(text) or BUG_REGEX.search(text)):
        return True
    return False

def has_action_signal(text: str) -> bool:
    return bool(ACTION_REGEX.search(text) or BUG_REGEX.search(text) or
                PRIORITY_REGEX.search(text) or DUE_REGEX.search(text))

# ---------- Heuristic fallback ----------
def heuristic_analyze(text: str, mentioned: bool=False) -> Dict[str, Any]:
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
    return dict(source="heuristic", confidence=conf, title=title,
                description=description, issue_type=issue_type,
                priority=priority, labels=["merci","autocaptured"])

# ---------- AI analysis ----------
AI_SYSTEM_PROMPT = """You classify workplace chat messages for whether they are actionable requests that should become Jira issues.

Return ONLY strict JSON with keys:
- is_request: boolean
- confidence: number in [0,1]
- issue_type: one of ["Task","Bug","Story","Spike"]
- priority: one of ["Highest","High","Medium","Low"]
- title: concise ≤80 chars
- description: short plain text paragraph
- labels: array (must include "merci" and "autocaptured")
"""

def _scrub_to_json(text: str) -> Optional[dict]:
    if not text: return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start:end+1]
    try:
        return json.loads(candidate)
    except Exception:
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            return None

async def _openai_call(prompt: str) -> Optional[dict]:
    if not OPENAI_API_KEY:
        return None
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": AI_MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code >= 400:
            return None
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _scrub_to_json(content)

async def ai_analyze(text: str, mentioned: bool=False) -> Dict[str, Any]:
    if not AI_ENABLE:
        return heuristic_analyze(text, mentioned)

    user_prompt = f"Message:\n{text.strip()}\n\nThe user {'mentioned the bot' if mentioned else 'did not mention the bot'}."
    last_err = None
    for attempt in range(1, AI_MAX_RETRIES + 1):
        try:
            result = await _openai_call(user_prompt)
            if not result:
                raise RuntimeError("AI returned no JSON")
            is_request = bool(result.get("is_request"))
            conf = float(result.get("confidence", 0))
            issue_type = str(result.get("issue_type") or "Task")
            priority = str(result.get("priority") or "Medium")
            title = (str(result.get("title") or "").strip() or text.strip().splitlines()[0])[:80] or "Auto-created by Merci"
            desc = str(result.get("description") or f"Source text:\n{text.strip()}")
            labels = result.get("labels") or ["merci","autocaptured"]
            if "merci" not in labels: labels.append("merci")
            if "autocaptured" not in labels: labels.append("autocaptured")
            if not is_request:
                conf = min(conf, 0.5)
            return dict(source="ai", confidence=max(0.0, min(1.0, conf)),
                        title=title, description=desc, issue_type=issue_type,
                        priority=priority, labels=labels[:10])
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.2 * attempt + random.random() * 0.2)

    fallback = heuristic_analyze(text, mentioned)
    fallback["source"] = f"fallback:{type(last_err).__name__}"
    return fallback

# ---------- Jira + Slack ----------
async def jira_create(summary: str, description: str, issue_type: str = "Task",
                      labels: Optional[list]=None, priority: Optional[str]=None) -> str:
    if labels is None: labels = ["merci","autocaptured"]
    fields = {
        "project": {"key": JIRA_PROJECT},
        "summary": summary,
        "description": {
            "type": "doc",
            "version": 1,
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": description}]}
            ],
        },
        "issuetype": {"name": issue_type},
        "labels": labels,
    }
    if priority:
        fields["priority"] = {"name": priority}
    payload = {"fields": fields}
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

def verify_slack(request: Request, raw_body: bytes):
    if not SLACK_SIGNING_SECRET:
        return
    ts = request.headers.get("X-Slack-Request-Timestamp", "0")
    if abs(time.time() - int(ts)) > 60 * 5:
        raise HTTPException(401, "stale request")
    sig_basestring = f"v0:{ts}:{raw_body.decode()}".encode()
    my_sig = "v0=" + hmac.new(SLACK_SIGNING_SECRET, sig_basestring, hashlib.sha256).hexdigest()
    req_sig = request.headers.get("X-Slack-Signature", "")
    if not hmac.compare_digest(my_sig, req_sig):
        raise HTTPException(401, "invalid signature")

# ---------- Routes ----------
@app.get("/health")
async def health():
    return {"ok": True, "ai": AI_ENABLE, "auto_threshold": MIN_CONF_AUTO}

@app.get("/try", response_class=HTMLResponse)
async def try_form():
    return """
    <html><head><title>Merci Demo</title></head>
      <body style="font-family:system-ui;max-width:720px;margin:40px auto">
        <h1>Merci — Create a Jira Ticket</h1>
        <p>Paste a natural request (e.g., <i>please update onboarding docs before Monday</i>)</p>
        <form method="POST" action="/simulate">
          <textarea name="text" rows="6" style="width:100%;"></textarea><br/>
          <button type="submit" style="margin-top:10px;padding:8px 14px">Analyze</button>
        </form>
      </body>
    </html>
    """

@app.post("/simulate")
async def simulate(text: str = Form(...)):
    a = await ai_analyze(text)
    if is_smalltalk(text):
        return JSONResponse({"created": None, "reason": "smalltalk_greeting", "analysis": a})
    if REQUIRE_SIGNAL_FOR_CREATE and not has_action_signal(text):
        return JSONResponse({"created": None, "reason": "no_action_signals", "analysis": a})
    if a["confidence"] >= MIN_CONF_AUTO:
        key = await jira_create(a["title"], a["description"], a["issue_type"],
                                labels=a.get("labels"), priority=a.get("priority"))
        jira_url = f"{JIRA_BASE}/browse/{key}"
        return JSONResponse({"created": key, "url": jira_url, "analysis": a})
    elif a["confidence"] >= MIN_CONF_SUGGEST:
        return JSONResponse({"created": None, "suggestion": f"Create a {a['issue_type']} (conf {a['confidence']:.2f})",
                             "analysis": a})
    else:
        return JSONResponse({"created": None, "reason": "low_confidence", "analysis": a})

@app.post("/slack/command")
async def slack_command(request: Request):
    raw = await request.body()
    verify_slack(request, raw)
    form = dict([tuple(kv.split("=", 1)) for kv in raw.decode().split("&") if "=" in kv])
    user_text = re.sub(r"\+", " ", form.get("text", ""))
    a = await ai_analyze(user_text)
    if is_smalltalk(user_text) or (REQUIRE_SIGNAL_FOR_CREATE and not has_action_signal(user_text)):
        return {"response_type": "ephemeral",
                "text": "Not a request. Try again with an action (e.g., “please update…”, “fix…”, “add…”)"} 
    if a["confidence"] >= MIN_CONF_AUTO:
        key = await jira_create(a["title"], a["description"], a["issue_type"],
                                labels=a.get("labels"), priority=a.get("priority"))
        return {"response_type": "in_channel", "text": f"✅ Created *{key}*: {a['title']}"}
    else:
        return {"response_type": "ephemeral",
                "text": f"Suggestion: {a['issue_type']} (conf {a['confidence']:.2f}). Re-run `/merci {user_text}` to confirm."}

@app.post("/slack/events")
async def slack_events(request: Request):
    raw = await request.body()
    verify_slack(request, raw)
    data = await request.json()
    if data.get("type") == "url_verification":
        return PlainTextResponse(data["challenge"])
    event = data.get("event", {})
    if not event or event.get("subtype"):
        return {"ok": True}
    text = event.get("text", "")
    channel = event.get("channel")
    ts = event.get("ts")
    mentioned = False
    try:
        bot_user_id = data.get("authorizations", [{}])[0].get("user_id")
        if bot_user_id and re.search(rf"<@{re.escape(bot_user_id)}>", text):
            mentioned = True
    except Exception:
        mentioned = False
    a = await ai_analyze(text, mentioned=mentioned)
    if is_smalltalk(text) or (REQUIRE_SIGNAL_FOR_CREATE and not has_action_signal(text)):
        return {"ok": True}
    if a["confidence"] >= max(MIN_CONF_AUTO, 0.90):
        key = await jira_create(a["title"], a["description"], a["issue_type"],
                                labels=a.get("labels"), priority=a.get("priority"))
        await slack_post(channel, ts, f"✅ Created *{key}* from this message.")
    elif a["confidence"] >= MIN_CONF_SUGGEST:
        await slack_post(channel, ts,
                         f"Create a {a['issue_type']}? (conf {a['confidence']:.2f}) Use `/merci {text}` to confirm.")
    return {"ok": True}
