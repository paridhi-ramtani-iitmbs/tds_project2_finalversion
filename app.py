

import os
import sys
import subprocess
import time
import importlib.util
import importlib
import json
import base64
import uuid
import logging
import traceback
import nest_asyncio
import shutil
import io
import gzip
import zipfile
import tarfile
from urllib.parse import urlparse, urljoin
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime

# ==========================================
# 0. AUTOMATIC ENVIRONMENT SETUP
# ==========================================
REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "langgraph",
    "langchain",
    "langchain-openai",
    "langchain-community",
    "langchain-core",
    "playwright",
    "pandas",
    "numpy",
    "requests",
    "beautifulsoup4",
    "pypdf",
    "openai",
    "nest-asyncio",
    "pillow",
    "matplotlib",  # Added for Visualization
    "seaborn",      # Added for Visualization
    "networkx",     # Added for network analysis
    "torch",        # Added for ML models (deep learning)
    "statsmodels",  # Added for statistical/ML models
    "geopandas"     # Added for geo-spatial analysis
]

def is_package_installed(package_name):
    return importlib.util.find_spec(package_name) is not None

def setup_environment():
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        import_name = package
        if package == "beautifulsoup4": import_name = "bs4"
        elif package == "pypdf": import_name = "pypdf"
        elif package == "langchain-openai": import_name = "langchain_openai"
        elif package == "nest-asyncio": import_name = "nest_asyncio"
        elif package == "pillow": import_name = "PIL"
        elif package == "langchain-core": import_name = "langchain_core"

        if not is_package_installed(import_name):
            missing_packages.append(package)

    if missing_packages:
        print(f"Installing dependencies... {missing_packages}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("Dependencies installed.")
            importlib.invalidate_caches()
        except subprocess.CalledProcessError as e:
            print(f"Error installing packages: {e}")
            sys.exit(1)

        print(" Checking Playwright browsers...")
        try:
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
            subprocess.check_call([sys.executable, "-m", "playwright", "install-deps"])
        except Exception:
            print(" Playwright install skipped (assumed installed).")
    else:
        print(" Dependencies ready.")

if __name__ == "__main__":
    setup_environment()

# ==========================================
# 1. IMPORTS & CONFIG
# ==========================================
try:
    from fastapi import FastAPI, BackgroundTasks, HTTPException
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langgraph.graph import StateGraph, END, START
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from playwright.async_api import async_playwright, Page
    from openai import AsyncOpenAI
    import pandas as pd
    import numpy as np
    import requests
    from PIL import Image
    from pypdf import PdfReader
    from bs4 import BeautifulSoup
    import matplotlib.pyplot as plt
    import seaborn as sns
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

nest_asyncio.apply()

# --- CONFIGURATION ---
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MY_SECRET = os.environ.get("MY_SECRET")

if OPENAI_API_KEY is None:
    print("WARNING: OPENAI_API_KEY not found in env. Ensure it is set.")

MY_EMAIL = os.environ.get("MY_EMAIL") # Load email from environment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Agent")
TASK_STORE: Dict[str, Dict[str, Any]] = {}

# ==========================================
# 2. CUSTOM PYTHON REPL (Stateless/Safe)
# ==========================================
class LocalPythonREPL:
    def __init__(self, task_id: str):
        self.task_id = task_id
        # Define the namespace for this execution only
        self.namespace = {
            "pd": pd, "np": np, "requests": requests,
            "os": os, "json": json, "base64": base64,
            "gzip": gzip, "io": io,
            "PdfReader": PdfReader,
            "BeautifulSoup": BeautifulSoup,
            "zipfile": zipfile,
            "tarfile": tarfile,
            "Image": Image,
            "urljoin": urljoin,
            "urlparse": urlparse,
            "subprocess": subprocess,
            "plt": plt, # Matplotlib for visualization
            "sns": sns  # Seaborn for visualization
        }

    def execute(self, code: str) -> str:
        import io
        from contextlib import redirect_stdout, redirect_stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Ensure matplotlib uses a non-interactive backend to prevent errors
                plt.switch_backend('Agg')
                exec(code, self.namespace)

            output = stdout_capture.getvalue()
            error = stderr_capture.getvalue()

            result = ""
            if output: result += f"STDOUT:\n{output}\n"
            if error: result += f"STDERR:\n{error}\n"
            if not result: result = "Code executed successfully (no output)."
            return result
        except Exception as e:
            # --- SMART ERROR HINTS ---
            error_msg = f"EXECUTION ERROR: {str(e)}\nTraceback: {traceback.format_exc()}"
            e_str = str(e).lower()
            if "no columns to parse" in e_str or "emptydataerror" in e_str:
                error_msg += "\nHINT: The file might be empty or malformed. Use `print(requests.get(url).text)` to inspect raw content."
            elif "keyerror" in e_str:
                 error_msg += "\nHINT: Column not found. Check `df.columns` to see actual names. If the file has no headers, use `pd.read_csv(..., header=None)`."
            elif "403" in e_str:
                error_msg += "\nHINT: 403 Forbidden. The URL might be wrong, or you need to visit it with `navigate_web` first to check for protections."
            elif "name" in e_str and "is not defined" in e_str:
                error_msg += "\nHINT: Variable lost? Note that variables persist across calls. Did you define it?"
            return error_msg

BROWSER_SESSIONS: Dict[str, Dict[str, Any]] = {}
REPL_SESSIONS: Dict[str, LocalPythonREPL] = {}  # Global store for REPL sessions

# ==========================================
# 3. TOOLS DEFINITION
# ==========================================
async def get_browser_page(task_id: str) -> Page:
    if task_id not in BROWSER_SESSIONS:
        p = await async_playwright().start()
        # Launch options optimized for SPEED
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            viewport={"width": 1280, "height": 720}
        )
        page = await context.new_page()

        # BLOCK images, fonts, and css to make loading instant
        await page.route("**/*.{png,jpg,jpeg,gif,css,woff,woff2,svg,ico}", lambda route: route.abort())

        page.on("download", lambda download: download.cancel())
        BROWSER_SESSIONS[task_id] = {"playwright": p, "browser": browser, "page": page}
    return BROWSER_SESSIONS[task_id]["page"]

@tool
async def navigate_web(url: str, task_id: str):
    """Navigates to a URL. Returns HTML content preview. OPTIMIZED FOR SPEED."""
    ignored_exts = ['.csv', '.json', '.xml', '.mp3', '.wav', '.zip', '.pdf', '.png', '.jpg', '.mp4']
    parsed = urlparse(url)
    if any(parsed.path.lower().endswith(ext) for ext in ignored_exts):
        return f"STOP! {url} is a file. DO NOT navigate. Use python_interpreter or read_pdf_file."

    try:
        page = await get_browser_page(task_id)
        # domcontentloaded is much faster than networkidle
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        html = await page.content()
        return f"Navigated to {url}.\nPage HTML Preview:\n{html[:1500]}..."
    except Exception as e:
        return f"Navigation Failed: {e}"

@tool
async def read_pdf_file(url: str):
    """Reads a PDF from a URL and returns text."""
    try:
        r = requests.get(url)
        r.raise_for_status()
        f = io.BytesIO(r.content)
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return f"PDF Content Preview (First 2000 chars):\n{text[:2000]}"
    except Exception as e:
        return f"PDF Error: {e}"

@tool
async def click_element(selector: str, task_id: str):
    """Clicks an element (CSS selector)."""
    try:
        page = await get_browser_page(task_id)
        # Short timeout because we want to fail fast if it's not there
        await page.click(selector, timeout=3000)
        return f"Clicked element: {selector}"
    except Exception as e:
        return f"Click Failed: {e}"

@tool
async def type_text(selector: str, text: str, task_id: str):
    """Types text into an input."""
    try:
        page = await get_browser_page(task_id)
        await page.fill(selector, text)
        return f"Typed '{text}' into {selector}"
    except Exception as e:
        return f"Typing Failed: {e}"

@tool
async def scroll_page(task_id: str):
    """Scrolls down."""
    try:
        page = await get_browser_page(task_id)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        return "Scrolled."
    except Exception as e:
        return f"Scroll Failed: {e}"

@tool
async def get_visual_context(task_id: str):
    """Takes a screenshot. SLOW - Use only if text extraction fails."""
    try:
        page = await get_browser_page(task_id)
        screenshot_bytes = await page.screenshot(full_page=False)
        img = Image.open(io.BytesIO(screenshot_bytes))
        max_width = 800
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        output_buffer = io.BytesIO()
        img.save(output_buffer, format='JPEG', quality=50)
        b64_img = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        return f"__IMAGE_CAPTURE__:{b64_img}"
    except Exception as e:
        return f"Screenshot Failed: {e}"

@tool
def python_interpreter(code: str, task_id: str):
    """
    Executes Python code. Use for math, API calls, Parsing CSV/JSON, Visualization.
    Use `plt.savefig('chart.png')` for charts.
    Variables persist across calls for the same task_id.
    """
    if task_id not in REPL_SESSIONS:
        REPL_SESSIONS[task_id] = LocalPythonREPL(task_id)

    repl = REPL_SESSIONS[task_id]
    return repl.execute(code)

@tool
async def analyze_media_file(media_url: str, task_id: str, current_url: str = ""):
    """Transcribes AUDIO or VIDEO files to text. Resolves relative URLs using current_url if provided."""
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        if not media_url.startswith("http") and current_url:
            media_url = urljoin(current_url, media_url)
        elif not media_url.startswith("http"):
            return "Error: Relative URL detected and no current_url provided. Resolve it using urljoin first."

        # Add User-Agent headers to prevent 403/401 from basic bot protection
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Download first to avoid stream issues
        response = requests.get(media_url, headers=headers)

        if response.status_code != 200:
            return f"Failed to fetch media: {response.status_code}"

        # SAFETY CHECK: Ensure we didn't download an HTML page (Soft 404 or Login Page)
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            # If it's HTML, return the text so the agent can see it's an error page
            snippet = response.text[:300].replace("\n", " ")
            return f"ERROR: The URL returned HTML (webpage) instead of an audio file. Server Response: {snippet}..."

        path = urlparse(media_url).path
        ext = os.path.splitext(path)[1]

        # Handle cases where URL has no extension or is weird
        if ext.lower() == '.opus': ext = '.ogg'
        if not ext:
            if "audio/mpeg" in content_type: ext = ".mp3"
            elif "audio/wav" in content_type: ext = ".wav"
            elif "audio/ogg" in content_type: ext = ".ogg"
            else: ext = ".mp3" # Default fallback

        media_file = io.BytesIO(response.content)
        media_file.name = f"media{ext}"

        transcription = await client.audio.transcriptions.create(
            model="whisper-1",
            file=(f"media{ext}", media_file)
        )
        return f"Transcription: {transcription.text}"
    except Exception as e:
        return f"Media Analysis Error: {e}"

@tool
def install_package(package_name: str):
    """Installs a Python package dynamically."""
    try:
        if importlib.util.find_spec(package_name):
            return f"Package {package_name} is already installed."
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        importlib.invalidate_caches()
        return f"Successfully installed {package_name}."
    except Exception as e:
        return f"Failed to install {package_name}: {e}"

# Add new tools to the list
agent_tools = [navigate_web, read_pdf_file, click_element, type_text, scroll_page, get_visual_context, python_interpreter, analyze_media_file, install_package]

# ==========================================
# 4. AGENT STATE & GRAPH
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    task_id: str
    current_url: str
    email: str
    secret: str
    iteration: int

# --- CRITICAL FIX: HIGH SPEED RATE LIMITER ---
# Previous: 9/60 (too slow). New: 100/1 (basically unlimited).
rate_limiter = InMemoryRateLimiter(
    requests_per_second=100,
    check_every_n_seconds=0.1,
    max_bucket_size=100
)

async def agent_reasoning_node(state: AgentState):
    messages = state["messages"]

    # Handle Image Injection
    if isinstance(messages[-1], ToolMessage) and "__IMAGE_CAPTURE__:" in messages[-1].content:
        content = messages[-1].content
        b64_str = content.split("__IMAGE_CAPTURE__:")[1]
        new_content = [
            {"type": "text", "text": "Screenshot captured."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}}
        ]
        messages.append(HumanMessage(content=new_content))

    model = ChatOpenAI(
        model="gpt-5.1", # Using fast reliable model
        temperature=0,
        rate_limiter=rate_limiter
    ).bind_tools(agent_tools)

    # --- SYSTEM PROMPT: OPTIMIZED FOR CONCISENESS & SPEED ---
    system_prompt = (
        "You are a HIGH-SPEED Autonomous Agent. TIME IS CRITICAL (Max 3 mins).\n"
        "GOAL: Solve the data puzzle and Submit.\n\n"
        "RULES:"
        "1. DO NOT TALK. DO NOT EXPLAIN. JUST ACT."
        "2. IF you see a PDF, use `read_pdf_file`."
        "3. IF you see an Audio or Video file, ALWAYS use `analyze_media_file` to transcribe it—it may contain critical information like codes, cutoffs, or data. If the page mentions audio, video or media, fetch and parse the full HTML using python_interpreter with requests and BeautifulSoup to find the media URL, then analyze it."
        "4. IF you get a CSV/JSON/XML/other data URL, use `python_interpreter` to download, parse and analyze it."
        "5. IF instructions say 'POST', use `python_interpreter` to send requests. DO NOT fill forms."
        "6. ALWAYS check for `correct: false`. If false, read reason and RETRY IMMEDIATELY. DO NOT QUIT."
        "7. SUBMISSION: POST to the URL given in the prompt."
        "8. FINISH: **ONLY** output 'ALL TASKS COMPLETED' if you receive `{'correct': true, 'url': null}`. If `correct` is false, YOU MUST FIX IT."
        "9. For analysis, use `python_interpreter` for filtering, sorting, aggregating, reshaping, statistical/ML models (use torch or statsmodels; install scikit-learn etc. via `install_package` if needed), geo-spatial (use geopandas), network analysis (use networkx)."
        "10. ALWAYS check if required libraries are available; if not, use `install_package` to install them automatically (e.g., scikit-learn for ML, geopandas for geo-spatial, networkx for networks, etc.) before using them in code. Reload imports after install."
        f" NOTE: 'email': '{MY_EMAIL}', 'secret': '{MY_SECRET}'."
    )

    if not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=system_prompt))

    response = await model.ainvoke(messages)
    return {"messages": [response], "iteration": state["iteration"] + 1}

# --- FEEDBACK NODE ---
def feedback_node(state: AgentState):
    return {
        "messages": [
            HumanMessage(
                content="SYSTEM ALERT: Do not stop. If you submitted and got 'correct: false', TRY AGAIN. "
                        "If you got a new URL, navigate to it. Speed is key."
            )
        ]
    }

# --- ROUTING LOGIC (FIXED TO PREVENT EARLY STOP) ---
def route(state: AgentState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Check last tool output for end condition (robust against LLM hallucinations)
    end_condition_met = False
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            content = str(msg.content).lower().replace(" ", "")
            if 'correct":true' in content and 'url":null' in content:
                end_condition_met = True
            break  # Only check the most recent tool output

    if end_condition_met:
        return END

    if state["iteration"] > 50:
        return END

    return "feedback"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_reasoning_node)
workflow.add_node("tools", ToolNode(agent_tools))
workflow.add_node("feedback", feedback_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    route,
    {
        "tools": "tools",
        "feedback": "feedback",
        END: END
    }
)

workflow.add_edge("tools", "agent")
workflow.add_edge("feedback", "agent")

app_graph = workflow.compile()

# ==========================================
# Custom Middleware to Strip Leading and Trailing Spaces from Path
# ==========================================
class StripSpaceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.scope['path']
        request.scope['path'] = path.strip(' ')
        response = await call_next(request)
        return response

# ==========================================
# 5. FASTAPI SERVER
# ==========================================
app = FastAPI()
app.add_middleware(StripSpaceMiddleware)  # Add the custom middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskRequest(BaseModel):
    email: str
    secret: str
    url: str

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": "Invalid JSON"})

async def run_agent_background(task_id: str, start_url: str):
    logger.info(f"Task {task_id} started: {start_url}")
    TASK_STORE[task_id] = {"status": "running", "logs": [], "result": None}

    inputs = {
        "messages": [HumanMessage(content=f"Start at: {start_url}. task_id='{task_id}'.")],
        "task_id": task_id,
        "current_url": start_url,
        "email": MY_EMAIL,
        "secret": MY_SECRET,
        "iteration": 0
    }

    try:
        async for output in app_graph.astream(inputs, {"recursion_limit": 5000}):
            for key, val in output.items():
                if "messages" in val:
                    last_msg = val["messages"][-1]
                    if isinstance(last_msg, ToolMessage):
                        log_msg = f"Tool Output: {str(last_msg.content)[:200]}..."
                    elif hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        log_msg = f"Agent calling tool: {last_msg.tool_calls[0]['name']}"
                    else:
                        log_msg = f"Agent thought: {str(last_msg.content)[:200]}..."

                    logger.info(log_msg)
                    TASK_STORE[task_id]["logs"].append(log_msg)

        TASK_STORE[task_id]["status"] = "finished"
    except Exception as e:
        logger.error(f"Crash: {e}")
        TASK_STORE[task_id]["status"] = "failed"
        TASK_STORE[task_id]["result"] = str(e)
    finally:
        if task_id in BROWSER_SESSIONS:
            try:
                await BROWSER_SESSIONS[task_id]["browser"].close()
                await BROWSER_SESSIONS[task_id]["playwright"].stop()
            except: pass
            del BROWSER_SESSIONS[task_id]
        if task_id in REPL_SESSIONS:
            del REPL_SESSIONS[task_id]

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/submit_task")
async def submit(req: TaskRequest, bt: BackgroundTasks):
    # --- MODIFIED CHECK HERE ---
    if req.secret.strip().lower() != MY_SECRET.strip().lower():
        raise HTTPException(403, "Bad Secret")
    if req.email.strip().lower() != MY_EMAIL.strip().lower():
        raise HTTPException(403, "Bad Email")
    # --- END MODIFIED CHECK ---

    task_id = str(uuid.uuid4())
    bt.add_task(run_agent_background, task_id, req.url)
    return {"message": "OK", "task_id": task_id}

@app.get("/status/{task_id}")
async def status(task_id: str):
    return TASK_STORE.get(task_id, {"status": "not_found"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
