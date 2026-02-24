import os
import sys
import json
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai

# ==============================
# Load environment variables
# ==============================
load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

app = FastAPI()

# ==============================
# Enable CORS
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Request Model
# ==============================
class CodeRequest(BaseModel):
    code: str


# ==============================
# Tool Function: Execute Python
# ==============================
def execute_python_code(code: str) -> dict:
    """
    Executes Python code and returns exact stdout or traceback.
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


# ==============================
# AI Error Analyzer
# ==============================
def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    """
    Calls Gemini to extract error line numbers.
    """

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY")
    )

    prompt = f"""
Return ONLY valid JSON.
Do NOT include explanation.
Do NOT wrap in markdown.

Format:
{{"error_lines": [numbers]}}

CODE:
{code}

TRACEBACK:
{tb}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    raw_text = response.text.strip()

    # Remove markdown wrapping if model adds it
    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "")
        raw_text = raw_text.replace("```", "").strip()

    try:
        data = json.loads(raw_text)
        return data.get("error_lines", [])
    except json.JSONDecodeError:
        # If AI fails, return empty list safely
        return []


# ==============================
# Endpoint
# ==============================
@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    # If success → no AI call
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    # If error → invoke AI
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"],
    )

    return {
        "error": error_lines,
        "result": execution["output"],
    }