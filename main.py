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
st.title("🤖 NIIVA - AI Decision Engine")

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

# ---------------- DOMAIN DETECTION ----------------
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

# ---------------- INTENT DETECTION (DOMAIN-AWARE) ----------------
def detect_intent(prompt, domain):
    p = prompt.lower()

    # -------- HEALTHCARE --------
    if domain == "Healthcare":
        if any(k in p for k in ["symptom", "cause"]):
            return "Symptoms Analysis"
        if any(k in p for k in ["treatment", "cure", "medicine"]):
            return "Treatment Guidance"
        if any(k in p for k in ["diagnose", "diagnosis"]):
            return "Diagnosis"
        return "Medical Advice"

    # -------- FINANCE --------
    if domain == "Finance":
        if any(k in p for k in ["invest", "stock", "crypto"]):
            return "Investment Advice"
        if any(k in p for k in ["risk"]):
            return "Risk Analysis"
        if any(k in p for k in ["budget"]):
            return "Budget Planning"
        return "Financial Guidance"

    # -------- CODING --------
    if domain == "Coding":
        if any(k in p for k in ["error", "bug", "fix"]):
            return "Debugging"
        if any(k in p for k in ["build", "create"]):
            return "Code Generation"
        return "Code Explanation"

    # -------- BUSINESS --------
    if domain == "Business":
        if any(k in p for k in ["strategy"]):
            return "Strategy Planning"
        if any(k in p for k in ["marketing"]):
            return "Marketing Advice"
        return "Business Insights"

    return "General"

# ---------------- CONTEXT ENRICHMENT ----------------
def enrich_prompt(prompt, domain, intent):
    return f"""
You are an expert in {domain}.

Task: {intent}

Provide a clear, structured, and helpful response.

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
        json={
            "contents": [{"parts": [{"text": enriched}]}]
        }
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

            st.success(f"🧠 Domain: {domain} | Intent: {intent}")

            with ThreadPoolExecutor() as executor:
                gpt = executor.submit(call_chatgpt, user_input, domain, intent).result()
                gem = executor.submit(call_gemini, user_input, domain, intent).result()

            results = [r for r in [gpt, gem] if r]

            if not results:
                st.error("❌ No models returned response")
            else:
                for r in results:
                    r.relevance_score = relevance(user_input, r.response_text)

                compute_scores(results)
                best = max(results, key=lambda x: x.final_score)

                # -------- OUTPUT --------
                st.markdown("---")
                st.markdown("## 🚀 Best Model Output")

                st.subheader(best.model_name)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latency", f"{best.latency:.2f}s")
                c2.metric("Tokens", best.tokens_used)
                c3.metric("Relevance", f"{best.relevance_score:.2f}")
                c4.metric("Score", f"{best.final_score:.3f}")

                st.success("🏆 Best Model Selected")

                st.write(best.response_text)
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

# ---------------- DOMAIN DETECTION ----------------
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

# ---------------- INTENT DETECTION (DOMAIN-AWARE) ----------------
def detect_intent(prompt, domain):
    p = prompt.lower()

    # -------- HEALTHCARE --------
    if domain == "Healthcare":
        if any(k in p for k in ["symptom", "cause"]):
            return "Symptoms Analysis"
        if any(k in p for k in ["treatment", "cure", "medicine"]):
            return "Treatment Guidance"
        if any(k in p for k in ["diagnose", "diagnosis"]):
            return "Diagnosis"
        return "Medical Advice"

    # -------- FINANCE --------
    if domain == "Finance":
        if any(k in p for k in ["invest", "stock", "crypto"]):
            return "Investment Advice"
        if any(k in p for k in ["risk"]):
            return "Risk Analysis"
        if any(k in p for k in ["budget"]):
            return "Budget Planning"
        return "Financial Guidance"

    # -------- CODING --------
    if domain == "Coding":
        if any(k in p for k in ["error", "bug", "fix"]):
            return "Debugging"
        if any(k in p for k in ["build", "create"]):
            return "Code Generation"
        return "Code Explanation"

    # -------- BUSINESS --------
    if domain == "Business":
        if any(k in p for k in ["strategy"]):
            return "Strategy Planning"
        if any(k in p for k in ["marketing"]):
            return "Marketing Advice"
        return "Business Insights"

    return "General"

# ---------------- CONTEXT ENRICHMENT ----------------
def enrich_prompt(prompt, domain, intent):
    return f"""
You are an expert in {domain}.

Task: {intent}

Provide a clear, structured, and helpful response.

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
        json={
            "contents": [{"parts": [{"text": enriched}]}]
        }
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

            st.success(f"🧠 Domain: {domain} | Intent: {intent}")

            with ThreadPoolExecutor() as executor:
                gpt = executor.submit(call_chatgpt, user_input, domain, intent).result()
                gem = executor.submit(call_gemini, user_input, domain, intent).result()

            results = [r for r in [gpt, gem] if r]

            if not results:
                st.error("❌ No models returned response")
            else:
                for r in results:
                    r.relevance_score = relevance(user_input, r.response_text)

                compute_scores(results)
                best = max(results, key=lambda x: x.final_score)

                # -------- OUTPUT --------
                st.markdown("---")
                st.markdown("## 🚀 Best Model Output")

                st.subheader(best.model_name)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latency", f"{best.latency:.2f}s")
                c2.metric("Tokens", best.tokens_used)
                c3.metric("Relevance", f"{best.relevance_score:.2f}")
                c4.metric("Score", f"{best.final_score:.3f}")

                st.success("🏆 Best Model Selected")

                st.write(best.response_text)
