import streamlit as st
import os
import requests
from dotenv import load_dotenv

# -------------------------------
# Load ENV
# -------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY or not GEMINI_API_KEY:
    st.error("API keys not found. Check your .env file.")
    st.stop()

# -------------------------------
# API ENDPOINTS
# -------------------------------
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

# -------------------------------
# MODEL CONFIG
# -------------------------------
LLM_MODELS = {
    "gpt-4": {"latency": 0.7, "cost": 0.9, "domain_relevance": 0.95},
    "gemini-pro": {"latency": 0.5, "cost": 0.6, "domain_relevance": 0.88}
    
}

WEIGHTS = {"latency": 0.3, "cost": 0.3, "domain_relevance": 0.4}

# -------------------------------
# SCORING
# -------------------------------
def score_models():
    scores = {}
    for model, v in LLM_MODELS.items():
        score = (
            WEIGHTS["latency"] * (1 - v["latency"]) +
            WEIGHTS["cost"] * (1 - v["cost"]) +
            WEIGHTS["domain_relevance"] * v["domain_relevance"]
        )
        scores[model] = round(score, 4)
    return scores

def select_best_model(scores):
    return max(scores, key=scores.get)

# -------------------------------
# API CALLS
# -------------------------------
def call_openai(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(OPENAI_URL, headers=headers, json=payload)
        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"OpenAI Error: {str(e)}"


def call_gemini(prompt):
    headers = {"Content-Type": "application/json"}
