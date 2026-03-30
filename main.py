import streamlit as st
import time
import requests
from dataclasses import dataclass
from typing import List, Any
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# ---------------- CONFIG ----------------
st.set_page_config(page_title="NIIVA", layout="wide")

st.title("NIIVA - AI Decision Engine")

# ---------------- WEIGHTS ----------------
W_RELEVANCE = 0.5
W_LATENCY = 0.3
W_COST = 0.2

# ---------------- INPUT ----------------
user_input = st.text_area("Enter your prompt")
run_btn = st.button("Run")

# ---------------- API KEYS ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

# ---------------- DATA ----------------
@dataclass
class ModelResponse:
    model_name: str
    response_text: str
    latency: float
    tokens_used: int
    raw_response: Any
    relevance_score: float = 0.0
    latency_score: float = 0.0
    cost_score: float = 0.0
    final_score: float = 0.0

# ---------------- SAFE REQUEST ----------------
def safe_request(url, headers=None, json=None, retries=2):
    for _ in range(retries):
        try:
            res = requests.post(url, headers=headers, json=json, timeout=15)
            if res.status_code == 200:
                return res
            else:
                st.error(f"API Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
            time.sleep(1)
    return None

# ---------------- DOMAIN ----------------
def detect_domain(prompt):
    p = prompt.lower()

    if any(k in p for k in ["doctor", "hospital", "disease", "medicine", "symptom", "treatment"]):
        return "Healthcare"
    if any(k in p for k in ["stock", "investment", "finance", "money", "crypto"]):
        return "Finance"
    if any(k in p for k in ["law", "legal", "contract"]):
        return "Legal"
    if any(k in p for k in ["code", "python", "api", "bug", "program"]):
        return "Coding"
    if any(k in p for k in ["business", "startup", "market", "growth"]):
        return "Business"

    return "General"

# ---------------- INTENT ----------------
def detect_intent(prompt, domain):
    p = prompt.lower()

    if domain == "Healthcare":
        if "symptom" in p:
            return "Symptoms Analysis"
        if "treatment" in p:
            return "Treatment Guidance"
        return "Medical Advice"

    if domain == "Finance":
        if "invest" in p:
            return "Investment Advice"
        if "risk" in p:
            return "Risk Analysis"
        return "Financial Guidance"

    if domain == "Coding":
        if "error" in p or "bug" in p:
            return "Debugging"
        return "Code Assistance"

    if domain == "Business":
        return "Business Strategy"

    return "General"

# ---------------- CONTEXT ----------------
def enrich_prompt(prompt, domain, intent):
    return f"""
You are an expert in {domain}.
Task: {intent}
Provide a clear structured answer.

User Query:
{prompt}
"""

# ---------------- RELEVANCE ----------------
def relevance(prompt, response):
    try:
        score = len(set(prompt.lower().split()) & set(response.lower().split()))
        return min(score / 10, 1.0)
    except:
        return 0.0

# ---------------- MODEL CALLS ----------------
def call_chatgpt(prompt, domain, intent):
    start = time.time()
    enriched = enrich_prompt(prompt, domain, intent)

    res = safe_request(
        OPENAI_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": enriched}],
            "max_tokens": 300
        }
    )

    latency = time.time() - start
    if not res:
        return None

    data = res.json()

    return ModelResponse(
        "ChatGPT",
        data.get("choices", [{}])[0].get("message", {}).get("content", ""),
        latency,
        data.get("usage", {}).get("total_tokens", 0),
        data
    )

def call_gemini(prompt, domain, intent):
    start = time.time()
    enriched = enrich_prompt(prompt, domain, intent)

    res = safe_request(
        GEMINI_URL,
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": enriched}]}]}
    )

    latency = time.time() - start
    if not res:
        return None

    data = res.json()

    return ModelResponse(
        "Gemini",
        data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", ""),
        latency,
        data.get("usageMetadata", {}).get("totalTokenCount", 0),
        data
    )

# ---------------- SCORING ----------------
def compute_scores(results: List[ModelResponse]):
    max_latency = max(r.latency for r in results)
    max_tokens = max(r.tokens_used for r in results)

    for r in results:
        r.latency_score = 1 - (r.latency / max_latency) if max_latency else 0
        r.cost_score = 1 - (r.tokens_used / max_tokens) if max_tokens else 0

        r.final_score = (
            W_RELEVANCE * r.relevance_score +
            W_LATENCY * r.latency_score +
            W_COST * r.cost_score
        )

# ---------------- MAIN ----------------
if run_btn:

    if not user_input.strip():
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Processing..."):

            domain = detect_domain(user_input)
            intent = detect_intent(user_input, domain)

            st.success(f"Domain: {domain} | Intent: {intent}")

            with ThreadPoolExecutor() as executor:
                gpt = executor.submit(call_chatgpt, user_input, domain, intent).result()
                gem = executor.submit(call_gemini, user_input, domain, intent).result()

            results = [r for r in [gpt, gem] if r]

            if not results:
                st.error("No response from models")
            else:
                for r in results:
                    r.relevance_score = relevance(user_input, r.response_text)

                compute_scores(results)
                best = max(results, key=lambda x: x.final_score)

                st.subheader("Best Model: " + best.model_name)
                st.write(best.response_text)
    
                                    
        
    
