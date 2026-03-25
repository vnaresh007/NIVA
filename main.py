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
            res = requests.post(url, headers=headers, json=json, timeout=10)
            if res.status_code == 200:
                return res
        except:
            time.sleep(1)
    return None

# ---------------- KEYWORD FALLBACK ----------------
def keyword_domain(prompt):
    p = prompt.lower()

    if any(k in p for k in ["doctor", "medicine", "patient", "disease"]):
        return "Healthcare"
    if any(k in p for k in ["stock", "investment", "bank", "crypto", "money"]):
        return "Finance"
    if any(k in p for k in ["law", "legal", "contract", "agreement"]):
        return "Legal"
    if any(k in p for k in ["code", "python", "java", "bug", "program"]):
        return "Coding"

    return "General"

# ---------------- DOMAIN DETECTION ----------------
def detect_domain(prompt):
    try:
        res = safe_request(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": f"""
Classify this prompt into EXACTLY one category:

Healthcare
Finance
Legal
Coding
General

Return ONLY the category.

Prompt: {prompt}
"""
                }],
                "max_tokens": 5
            }
        )

        if res:
            text = res.json()["choices"][0]["message"]["content"].strip().capitalize()

            if text in ["Healthcare", "Finance", "Legal", "Coding", "General"]:
                return text

        return keyword_domain(prompt)

    except:
        return keyword_domain(prompt)

# ---------------- RELEVANCE ----------------
def relevance(prompt, response, domain):
    res = safe_request(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": f"Score relevance 0-1\nDomain:{domain}\nPrompt:{prompt}\nResponse:{response}"
            }],
            "max_tokens": 10
        }
    )

    try:
        return float(res.json()["choices"][0]["message"]["content"])
    except:
        return 0.0

# ---------------- MODEL CALLS ----------------
def chatgpt(prompt, domain):
    start = time.time()

    res = safe_request(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": f"Expert in {domain}"},
                {"role": "user", "content": prompt}
            ]
        }
    )

    latency = time.time() - start
    if not res:
        return None

    data = res.json()

    return ModelResponse(
        "ChatGPT",
        data.get("choices", [{}])[0].get("message", {}).get("content", "No response"),
        latency,
        data.get("usage", {}).get("total_tokens", 0),
        data
    )

def gemini(prompt, domain):
    start = time.time()

    res = safe_request(
        GEMINI_URL,
        json={
            "contents": [{
                "parts": [{"text": f"[{domain}] {prompt}"}]
            }]
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
            .get("text", "No response"),
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
if run_btn and user_input.strip():

    with st.spinner("Processing..."):

        domain = detect_domain(user_input)
        st.success(f"🧠 Detected Domain: {domain}")

        # Parallel execution
        with ThreadPoolExecutor() as executor:
            gpt = executor.submit(chatgpt, user_input, domain).result()
            gem = executor.submit(gemini, user_input, domain).result()

        results = [r for r in [gpt, gem] if r is not None]

        if not results:
            st.error("No responses received")
        else:
            for r in results:
                r.relevance_score = relevance(user_input, r.response_text, domain)

            compute_scores(results)

            best = max(results, key=lambda x: x.final_score)

            # -------- FINAL OUTPUT (ONLY BEST MODEL) --------
            st.markdown("## 🚀 Best Model Output")

            st.subheader(f"🤖 {best.model_name}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⚡ Latency", f"{best.latency:.2f}s")
            c2.metric("💰 Tokens", best.tokens_used)
            c3.metric("🎯 Relevance", f"{best.relevance_score:.2f}")
            c4.metric("🏆 Score", f"{best.final_score:.3f}")

            st.success("🏆 Selected by Multi-Factor Scoring")

            st.markdown("### 💬 Response")
            st.write(best.response_text)
