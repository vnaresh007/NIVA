# NIIVA - Multi Model AI Comparator
# ---------------------------------
# Enhanced UI Version with proper descriptions and usability

import streamlit as st
import time
import requests
from dataclasses import dataclass
from typing import List, Any
import os
from dotenv import load_dotenv


load_dotenv()


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NIIVA - AI Comparator",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #4CAF50;
    }
    .subtitle {
        font-size: 18px;
        color: #888;
    }
    .card {
        padding: 15px;
        border-radius: 12px;
        background-color: #111;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🤖 NIIVA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next-Gen AI Model Comparison Platform</div>', unsafe_allow_html=True)

st.markdown("""
### 🚀 About NIIVA
NAIVA helps you compare multiple AI models like ChatGPT and Gemini in real-time.

🔍 **What you can do:**
- Compare responses from different AI models
- Analyze latency (speed)
- Track token usage (cost efficiency)
- View raw backend responses for debugging

💡 Ideal for developers, researchers, and AI enthusiasts.
""")

st.markdown("---")

# ---------------- API CONFIG ----------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

# ---------------- DATA STRUCTURE ----------------
@dataclass
class ModelResponse:
    model_name: str
    response_text: str
    latency: float
    tokens_used: int
    raw_response: Any

# ---------------- CHATGPT ----------------
def call_chatgpt(prompt: str):
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200
        }

        start = time.time()
        response = requests.post(OPENAI_URL, headers=headers, json=payload)
        latency = time.time() - start

        data = response.json()

        text = data.get("choices", [{}])[0].get("message", {}).get("content", "Error in response")
        tokens = data.get("usage", {}).get("total_tokens", 0)

        return ModelResponse("ChatGPT", text, latency, tokens, data)

    except Exception as e:
        st.error(f"ChatGPT Error: {e}")
        return None

# ---------------- GEMINI ----------------
def call_gemini(prompt: str):
    try:
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 200
            }
        }

        start = time.time()
        response = requests.post(GEMINI_URL, headers=headers, json=payload)
        latency = time.time() - start

        data = response.json()

        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "Error in response")
        )

        tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)

        return ModelResponse("Gemini Flash", text, latency, tokens, data)

    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return None

# ---------------- COMPARISON ----------------
def compare_models(prompt: str) -> List[ModelResponse]:
    results = []

    gpt = call_chatgpt(prompt)
    gemini = call_gemini(prompt)

    if gpt:
        results.append(gpt)
    if gemini:
        results.append(gemini)

    return results

# ---------------- INPUT SECTION ----------------
st.markdown("### 🧠 Enter Your Prompt")
user_input = st.text_area("Type your question or task:", height=120, placeholder="Example: Explain AI in simple terms")

run_btn = st.button("🚀 Run Comparison")

# ---------------- RESULTS ----------------
if run_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter a prompt")
    else:
        with st.spinner("Analyzing models..."):
            results = compare_models(user_input)

        if len(results) == 0:
            st.error("❌ No valid responses. Check API keys.")
        else:
            st.markdown("---")
            st.markdown("## 📊 Model Comparison Results")

            cols = st.columns(len(results))

            for i, res in enumerate(results):
                with cols[i]:
                    st.markdown(f"### {res.model_name}")
                    st.metric("⚡ Latency", f"{res.latency:.2f} sec")
                    st.metric("🔢 Tokens", res.tokens_used)

                    st.markdown("#### 💬 Response")
                    st.write(res.response_text)

                    st.markdown("#### 🛠 Debug Data")
                    with st.expander("View Raw Response"):
                        st.json(res.raw_response)

            if len(results) > 1:
                fastest = min(results, key=lambda x: x.latency)
                cheapest = min(results, key=lambda x: x.tokens_used)

                st.markdown("---")
                st.markdown("## 🏆 Insights")
                st.success(f"⚡ Fastest Model: {fastest.model_name}")
                st.info(f"💰 Most Efficient (Tokens): {cheapest.model_name}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
### 📌 How to Use
1. Enter your prompt
2. Click **Run Comparison**
3. View results side-by-side
4. Expand debug section if needed

---
Built with ❤️ by NIIVA Team 
""")    latency: float
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

                st.write(best.response_text)    latency: float
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
