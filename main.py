from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from pydantic import BaseModel
import threading
import requests
import json
import os
from time import time

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set")


request_log = {}

memory_lock = threading.Lock()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEMORY_FILE = "memory.json"

CONVERSATION_FILE = "conversations.json"

# Ensure memory file exists
@app.on_event("startup")
def setup_files():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w") as f:
            json.dump({"users": {}}, f)

    if not os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "w") as f:
            json.dump({"sessions": {}}, f)

class ChatRequest(BaseModel):
    message: str


def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"users": {}}


def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def get_user_memory(memory, session_id):
    if "users" not in memory:
        memory["users"] = {}

    if session_id not in memory["users"]:
        memory["users"][session_id] = {
            "preferences": [],
            "corrections": [],
            "facts": []
        }

    return memory["users"][session_id]


def load_conversations():
    try:
        with open(CONVERSATION_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"sessions": {}}


def save_conversations(data):
    with open(CONVERSATION_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_session_history(data, session_id):
    if "sessions" not in data:
        data["sessions"] = {}

    if session_id not in data["sessions"]:
        data["sessions"][session_id] = []

    return data["sessions"][session_id]



def is_feedback(message: str):
    keywords = ["vague", "wrong", "bad", "dont", "don't", "better", "improve", "too"]
    return any(k in message.lower() for k in keywords)

def extract_fact(message: str):
    triggers = ["my name is", "i am", "i like", "i prefer"]

    for t in triggers:
        if t in message.lower():
            return message.strip()

    return None

def update_memory(message: str, session_id: str):
    with memory_lock:
        memory_data = load_memory()
        user_memory = get_user_memory(memory_data, session_id)

        if "vague" in message.lower():
            user_memory["preferences"].append("be specific and structured")

        if "wrong" in message.lower():
            user_memory["corrections"].append(message)

        user_memory["corrections"] = user_memory["corrections"][-20:]
        user_memory["preferences"] = user_memory["preferences"][-20:]

        save_memory(memory_data)


def query_groq_messages(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": messages
    }

    response = requests.post(url, headers=headers, json=data, timeout=10)

    if response.status_code != 200:
        print("ERROR:", response.status_code, response.text)
        return "Error talking to AI"

    result = response.json()
    return result["choices"][0]["message"]["content"]

@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    msg = req.message.strip() if req.message else ""

    if not msg:
        return {"reply": "Your message was empty"}

    if len(msg) > 500:
        return {"reply": "Message too long (max 500 characters)"}

    blocked_words = ["hack", "exploit", "crash"]
    if any(word in msg.lower() for word in blocked_words):
        return {"reply": "Request not allowed"}

    session_id = request.headers.get("x-session-id")

    if not session_id:
        session_id = request.client.host  # fallback

    with memory_lock:
        memory_data = load_memory()
        user_memory = get_user_memory(memory_data, session_id)

    now = time()

    if session_id not in request_log:
        request_log[session_id] = []

    MAX_REQUESTS = 20
    WINDOW = 10
    COOLDOWN = 1.5

    # Clean old requests
    request_log[session_id] = [
        t for t in request_log[session_id] if now - t < WINDOW
    ]

    # Burst protection
    if len(request_log[session_id]) >= MAX_REQUESTS:
        return {"reply": "Too many requests, slow down"}

    # Rapid spam protection
    if request_log[session_id]:
        if now - request_log[session_id][-1] < COOLDOWN:
            return {"reply": "You're sending messages too fast"}

    request_log[session_id].append(now)

    if is_feedback(msg) and len(msg) < 200:
        update_memory(msg, session_id)
        with memory_lock:
            memory_data = load_memory()
            user_memory = get_user_memory(memory_data, session_id)
    else:
        fact = extract_fact(msg)
        if fact:
            with memory_lock:
                memory_data = load_memory()
                user_memory = get_user_memory(memory_data, session_id)

                if fact not in user_memory["facts"]:
                    user_memory["facts"].append(fact)

                user_memory["facts"] = user_memory["facts"][-20:]
                save_memory(memory_data)

    # INIT CONVERSATION
    with memory_lock:
        convo_data = load_conversations()
        history = get_session_history(convo_data, session_id)

        history.append({"role": "user", "content": msg})

        # trim
        history = history[-10:]
        convo_data["sessions"][session_id] = history

        save_conversations(convo_data)

    # BUILD SYSTEM PROMPT
    prefs = "\n".join(user_memory["preferences"])
    corrections = "\n".join(user_memory["corrections"])
    facts = "\n".join(user_memory.get("facts", []))

    system_prompt = f"""
You are Zippz, a conversational assistant with personality and emotional awareness.

Core behaviour:
- Be natural, conversational, and human-like
- Match the user's tone and energy
- Avoid sounding robotic, overly formal, or overly analytical

Emotional intelligence:
- Recognise emotional tone in the user's message
- Respond appropriately (e.g. excitement, humour, curiosity)
- Do not overreact or exaggerate emotions

Conversation awareness:
- Treat the interaction as an ongoing conversation
- Use previous messages for context and continuity
- Do not act like each message is isolated

Personality:
- You can use light humour, creativity, and wordplay when appropriate
- Be engaging, not just informative
- Avoid excessive praise or artificial enthusiasm

Memory usage:
- Use stored preferences to improve responses
- Avoid repeating mistakes mentioned in corrections

User preferences:
{prefs}

Previous corrections:
{corrections}

Known facts about the user:
{facts}
"""

    # SEND FULL CONVERSATION TO GROQ
    messages = [{"role": "system", "content": system_prompt}] + history

    reply = query_groq_messages(messages)

    with memory_lock:
        convo_data = load_conversations()
        history = get_session_history(convo_data, session_id)

        history.append({"role": "assistant", "content": reply})

        history = history[-10:]
        convo_data["sessions"][session_id] = history

        save_conversations(convo_data)

    cleaned = reply.replace("\\n", "\n").strip()
    return {"reply": cleaned}

from fastapi.responses import FileResponse

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


