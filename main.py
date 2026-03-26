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

# ---------------- DOMAIN ----------------
def keyword_domain(prompt):
    p = prompt.lower()
    if "doctor" in p or "medicine" in p:
        return "Healthcare"
    if "stock" in p or "money" in p:
        return "Finance"
    if "law" in p or "legal" in p:
        return "Legal"
    if "code" in p or "python" in p:
        return "Coding"
    return "General"

def detect_domain(prompt):
    try:
        res = safe_request(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": f"Classify into one: Healthcare, Finance, Legal, Coding, General. Prompt: {prompt}"
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

# ---------------- INTENT ----------------
def detect_intent(prompt):
    p = prompt.lower()
    if "build" in p or "create" in p:
        return "Generate"
    if "explain" in p or "what is" in p:
        return "Explain"
    if "compare" in p:
        return "Compare"
    if "error" in p or "fix" in p:
        return "Debug"
    return "General"

# ---------------- CONTEXT ENRICHMENT ----------------
def enrich_prompt(prompt, domain, intent):

    base = f"You are an expert in {domain}."

    if intent == "Generate":
        inst = "Provide complete structured solution."
    elif intent == "Explain":
        inst = "Explain clearly with examples."
    elif intent == "Compare":
        inst = "Compare with pros and cons."
    elif intent == "Debug":
        inst = "Fix issue and explain."
    else:
        inst = "Provide helpful answer."

    return f"{base}\nTask:{intent}\nInstruction:{inst}\nQuery:{prompt}"

# ---------------- RELEVANCE ----------------
def relevance(prompt, response, domain):
    res = safe_request(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": f"Score 0-1 relevance\nPrompt:{prompt}\nResponse:{response}"
            }],
            "max_tokens": 10
        }
    )
    try:
        return float(res.json()["choices"][0]["message"]["content"])
    except:
        return 0.0

# ---------------- MODELS ----------------
def chatgpt(prompt, domain, intent):
    start = time.time()
    enriched = enrich_prompt(prompt, domain, intent)

    res = safe_request(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": enriched}]
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

def gemini(prompt, domain, intent):
    start = time.time()
    enriched = enrich_prompt(prompt, domain, intent)

    res = safe_request(
        GEMINI_URL,
        json={"contents": [{"parts": [{"text": enriched}]}]}
    )

    latency = time.time() - start
    if not res:
        return None

    data = res.json()

    return ModelResponse(
        "Gemini",
        data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", ""),
        latency,
        data.get("usageMetadata", {}).get("totalTokenCount", 0),
        data
    )

# ---------------- SCORING ----------------
def compute_scores(results):
    max_lat = max(r.latency for r in results)
    max_tok = max(r.tokens_used for r in results)

    for r in results:
        r.latency_score = 1 - r.latency / max_lat if max_lat else 0
        r.cost_score = 1 - r.tokens_used / max_tok if max_tok else 0

        r.final_score = (
            W_RELEVANCE * r.relevance_score +
            W_LATENCY * r.latency_score +
            W_COST * r.cost_score
        )

# ---------------- MAIN ----------------
if run_btn and user_input.strip():

    with st.spinner("Processing..."):

        domain = detect_domain(user_input)
        intent = detect_intent(user_input)

        st.success(f"🧠 Domain: {domain} | Intent: {intent}")

        with ThreadPoolExecutor() as ex:
            gpt = ex.submit(chatgpt, user_input, domain, intent).result()
            gem = ex.submit(gemini, user_input, domain, intent).result()

        results = [r for r in [gpt, gem] if r]

        if not results:
            st.error("No response")
        else:
            for r in results:
                r.relevance_score = relevance(user_input, r.response_text, domain)

            compute_scores(results)
            best = max(results, key=lambda x: x.final_score)

            # -------- OUTPUT --------
            st.markdown("## 🚀 Best Model Output")

            st.subheader(best.model_name)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latency", f"{best.latency:.2f}s")
            c2.metric("Tokens", best.tokens_used)
            c3.metric("Relevance", f"{best.relevance_score:.2f}")
            c4.metric("Score", f"{best.final_score:.3f}")

            st.success("🏆 Best Model Selected")

            st.write(best.response_text)

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

# ---------------- DOMAIN ----------------
def keyword_domain(prompt):
    p = prompt.lower()
    if "doctor" in p or "medicine" in p:
        return "Healthcare"
    if "stock" in p or "money" in p:
        return "Finance"
    if "law" in p or "legal" in p:
        return "Legal"
    if "code" in p or "python" in p:
        return "Coding"
    return "General"

def detect_domain(prompt):
    try:
        res = safe_request(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": f"Classify into one: Healthcare, Finance, Legal, Coding, General. Prompt: {prompt}"
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

# ---------------- INTENT ----------------
def detect_intent(prompt):
    p = prompt.lower()
    if "build" in p or "create" in p:
        return "Generate"
    if "explain" in p or "what is" in p:
        return "Explain"
    if "compare" in p:
        return "Compare"
    if "error" in p or "fix" in p:
        return "Debug"
    return "General"

# ---------------- CONTEXT ENRICHMENT ----------------
def enrich_prompt(prompt, domain, intent):

    base = f"You are an expert in {domain}."

    if intent == "Generate":
        inst = "Provide complete structured solution."
    elif intent == "Explain":
        inst = "Explain clearly with examples."
    elif intent == "Compare":
        inst = "Compare with pros and cons."
    elif intent == "Debug":
        inst = "Fix issue and explain."
    else:
        inst = "Provide helpful answer."

    return f"{base}\nTask:{intent}\nInstruction:{inst}\nQuery:{prompt}"

# ---------------- RELEVANCE ----------------
def relevance(prompt, response, domain):
    res = safe_request(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": f"Score 0-1 relevance\nPrompt:{prompt}\nResponse:{response}"
            }],
            "max_tokens": 10
        }
    )
    try:
        return float(res.json()["choices"][0]["message"]["content"])
    except:
        return 0.0

# ---------------- MODELS ----------------
def chatgpt(prompt, domain, intent):
    start = time.time()
    enriched = enrich_prompt(prompt, domain, intent)

    res = safe_request(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": enriched}]
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

def gemini(prompt, domain, intent):
    start = time.time()
    enriched = enrich_prompt(prompt, domain, intent)

    res = safe_request(
        GEMINI_URL,
        json={"contents": [{"parts": [{"text": enriched}]}]}
    )

    latenc
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

# ---------------- DOMAIN DETECTION ----------------
def keyword_domain(prompt):
    p = prompt.lower()
    if any(k in p for k in ["doctor", "medicine", "patient", "disease"]):
        return "Healthcare"
    if any(k in p for k in ["stock", "investment", "bank", "crypto", "money"]):
        return "Finance"
    if any(k in p for k in ["law", "legal", "contract"]):
        return "Legal"
    if any(k in p for k in ["code", "python", "bug", "api"]):
        return "Coding"
    return "General"

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
Classify into one:
Healthcare, Finance, Legal, Coding, General
Return only one word.

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

# ---------------- INTENT DETECTION (NEW) ----------------
def detect_intent(prompt):
    p = prompt.lower()

    if any(k in p for k in ["build", "create", "write code", "api", "function"]):
        return "Generate"
    if any(k in p for k in ["explain", "what is", "define"]):
        return "Explain"
    if any(k in p for k in ["compare", "difference", "vs"]):
        return "Compare"
    if any(k in p for k in ["fix", "debug", "error"]):
        return "Debug"

    return "General"

# ---------------- CONTEXT ENRICHMENT (LEVEL 2) ----------------
def enrich_prompt(prompt, domain, intent):

    base = f"You are an expert in {domain}."

    if intent == "Generate":
        instruction = "Provide a complete, structured solution."
    elif intent == "Explain":
        instruction = "Explain clearly in simple terms with examples."
    elif intent == "Compare":
        instruction = "Compare clearly with pros and cons."
    elif intent == "Debug":
        instruction = "Identify the issue and provide a fix."
    else:
        instruction = "Provide a helpful answer."

    return f"""
{base}
Task: {intent}

Instruction:
{instruction}

User Query:
{prompt}
"""

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
def chatgpt(prompt, domain, intent):
    start = time.time()
    enriched = enrich_prompt(prompt, domain, intent)

    res = safe_request(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": f"Expert in {domain}"},
                {"role": "user", "content": enriched}
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

def gemini(prompt, domain, intent):
    start = time.time()
    enriched = enrich_prompt(prompt, domain, intent)

    res = safe_request(
        GEMINI_URL,
        json={
            "contents": [{
                "parts": [{"text": enriched}]
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
        intent = detect_intent(user_input)

        st.success(f"🧠 Domain: {domain} | Intent: {intent}")

        with ThreadPoolExecutor() as executor:
            gpt = executor.submit(chatgpt, user_input, domain, intent).result()
            gem = executor.submit(gemini, user_input, domain, intent).result()

        results = [r for r in [gpt, gem] if r]

        if not results:
            st.error("No responses received")
        else:
            for r in results:
                r.relevance_score = relevance(user_input, r.response_text, domain)

            compute_scores(results)
            best = max(results, key=lambda x: x.final_score)

            # -------- FINAL OUTPUT --------
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
""")
