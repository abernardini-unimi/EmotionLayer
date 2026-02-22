import os
from dotenv import load_dotenv  # type: ignore

load_dotenv()

# ==============================
# API KEYS
# ==============================

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Lista per controllo centralizzato
REQUIRED_KEYS = {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "HF_TOKEN": HF_TOKEN,
    "GROQ_API_KEY": GROQ_API_KEY,
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}

missing_keys = [key for key, value in REQUIRED_KEYS.items() if not value]

if missing_keys:
    raise RuntimeError(
        f"❌ Missing required API keys in .env file: {', '.join(missing_keys)}"
    )

# ==============================
# FILE PATHS
# ==============================

OUTPUT_EXCEL = "file/output/benchmark_report.xlsx"
DATASET_FILE = "file/test_case/test_case.json"
MODEL_FILE = "file/models_price.xlsx"
PROMPT_FILE_PATH = "prompt/llm_prompt.txt"
LLM_OUTPUT_JSON = "file/output/final_benchmark_results.json"

# ==============================
# JUDGE CONFIG
# ==============================

JUDGE_OUTPUT_FILE = "file/output/judge_evaluation_report.xlsx"
JUDGE_PROMPT_PATH = "prompt/judge_prompt.txt"
JUDGE_PROVIDER = "Claude"
JUDGE_MODEL = "claude-opus-4-6"