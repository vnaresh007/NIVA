import streamlit as st
import time
import requests
from dataclasses import dataclass
from typing import List, Any
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

st.set_page_config(page_title="NIIVA", layout="wide")
st.title("🤖 NIIVA - AI Decision Engine")

user_input = st.text_area("Enter your prompt")
run_btn = st.button("Run")

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

# ---------------- RETRY ----------------
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
def detect_domain(prompt):
    res = safe_request(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"Classify: {prompt} into Healthcare, Finance, Legal, Coding, General"}],
            "max_tokens": 5
        }
    )
    if not res:
        return "General"

    try:
        return res.json()["choices"][0]["message"]["content"].strip()
    except:
        return "General"

# ---------------- RELEVANCE ----------------
def relevance(prompt, response, domain):
    res = safe_request(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"Score relevance 0-1\nDomain:{domain}\nPrompt:{prompt}\nResponse:{response}"}],
            "max_tokens": 10
        }
    )
    try:
        return float(res.json()["choices"][0]["message"]["content"])
    except:
        return 0.0

# ---------------- MODELS ----------------
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
        data.get("choices", [{}])[0].get("message", {}).get("content", ""),
        latency,
        data.get("usage", {}).get("total_tokens", 0),
        data
    )


def gemini(prompt, domain):
    start = time.time()

    res = safe_request(
        GEMINI_URL,
        json={"contents": [{"parts": [{"text": f"[{domain}] {prompt}"}]}]}
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
def score(results):
    max_lat = max(r.latency for r in results)
    max_tok = max(r.tokens_used for r in results)

    for r in results:
        r.latency_score = 1 - (r.latency / max_lat if max_lat else 0)
        r.cost_score = 1 - (r.tokens_used / max_tok if max_tok else 0)

        r.final_score = (
            0.5 * r.relevance_score +
            0.3 * r.latency_score +
            0.2 * r.cost_score
        )

# ---------------- MAIN ----------------
if run_btn and user_input:

    with st.spinner("Processing..."):

        domain = detect_domain(user_input)
        st.info(f"Detected Domain: {domain}")

        # PARALLEL CALLS
        with ThreadPoolExecutor() as executor:
            gpt_future = executor.submit(chatgpt, user_input, domain)
            gem_future = executor.submit(gemini, user_input, domain)

            gpt = gpt_future.result()
            gem = gem_future.result()

        results = [r for r in [gpt, gem] if r]

        if not results:
            st.error("No responses")
        else:
            for r in results:
                r.relevance_score = relevance(user_input, r.response_text, domain)

            score(results)

            best = max(results, key=lambda x: x.final_score)

            # OUTPUT
            for r in results:
                st.markdown("---")
                st.subheader(r.model_name)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latency", f"{r.latency:.2f}s")
                c2.metric("Tokens", r.tokens_used)
                c3.metric("Relevance", f"{r.relevance_score:.2f}")
                c4.metric("Score", f"{r.final_score:.3f}")

                if r == best:
                    st.success("Best Model Selected")

                st.write(r.response_text)

            st.markdown("---")
            st.success(f"🚀 Recommended Model: {best.model_name}")
total_w = w_relevance + w_latency + w_cost
w_relevance /= total_w
w_latency /= total_w
w_cost /= total_w

# ---------------- INPUT ----------------
user_input = st.text_area("🧠 Enter Prompt")
run_btn = st.button("🚀 Run Comparison")

# ---------------- API CONFIG ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

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
    relevance_score: float = 0.0
    latency_score: float = 0.0
    cost_score: float = 0.0
    final_score: float = 0.0

# ---------------- LLM RELEVANCE ----------------
def llm_relevance_score(prompt: str, response: str, domain: str) -> float:
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        eval_prompt = f"""
Rate relevance (0 to 1)

Domain: {domain}
Prompt: {prompt}
Response: {response}

Return only number.
"""

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": eval_prompt}],
            "max_tokens": 10
        }

        res = requests.post(OPENAI_URL, headers=headers, json=payload)
        score = res.json()["choices"][0]["message"]["content"].strip()

        return float(score)

    except:
        return 0.0

# ---------------- MODEL CALLS ----------------
def call_chatgpt(prompt: str, domain: str):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": f"You are an expert in {domain}."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200
    }

    start = time.time()
    res = requests.post(OPENAI_URL, headers=headers, json=payload)
    latency = time.time() - start

    data = res.json()
    text = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", 0)

    return ModelResponse("ChatGPT", text, latency, tokens, data)

def call_gemini(prompt: str, domain: str):
    payload = {
        "contents": [{
            "parts": [{"text": f"[Domain: {domain}] {prompt}"}]
        }]
    }

    start = time.time()
    res = requests.post(GEMINI_URL, json=payload)
    latency = time.time() - start

    data = res.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)

    return ModelResponse("Gemini", text, latency, tokens, data)

# ---------------- MULTI-FACTOR SCORING ----------------
def compute_scores(results: List[ModelResponse]):
    latencies = [r.latency for r in results]
    tokens = [r.tokens_used for r in results]

    max_latency = max(latencies)
    max_tokens = max(tokens)

    for r in results:
        # Normalize (lower is better → invert)
        r.latency_score = 1 - (r.latency / max_latency)
        r.cost_score = 1 - (r.tokens_used / max_tokens)

        r.final_score = (
            w_relevance * r.relevance_score +
            w_latency * r.latency_score +
            w_cost * r.cost_score
        )

# ---------------- MAIN ----------------
if run_btn and user_input.strip():

    with st.spinner("Running models..."):
        gpt = call_chatgpt(user_input, domain)
        gemini = call_gemini(user_input, domain)

        results = [gpt, gemini]

        for r in results:
            r.relevance_score = llm_relevance_score(
                user_input, r.response_text, domain
            )

        compute_scores(results)

    # ---------------- DISPLAY ----------------
    best = max(results, key=lambda x: x.final_score)

    for r in results:
        st.markdown("---")
        st.markdown(f"## 🤖 {r.model_name}")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("⚡ Latency", f"{r.latency:.2f}s")
        c2.metric("💰 Tokens", r.tokens_used)
        c3.metric("🎯 Relevance", f"{r.relevance_score:.2f}")
        c4.metric("🏆 Score", f"{r.final_score:.3f}")

        if r.model_name == best.model_name:
            st.success("🏆 Best Model")

        st.write(r.response_text)

    st.markdown("---")
    st.success(f"🏆 Best Overall: {best.model_name}")

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
    relevance_score: float = 0.0
    final_score: float = 0.0

# ---------------- LLM RELEVANCE ----------------
def llm_relevance_score(prompt: str, response: str, domain: str) -> float:
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        eval_prompt = f"""
You are an expert evaluator.

Rate how relevant the response is to the given domain.

Domain: {domain}
User Prompt: {prompt}
Model Response: {response}

Return ONLY a number between 0 and 1.
"""

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": eval_prompt}],
            "max_tokens": 10
        }

        res = requests.post(OPENAI_URL, headers=headers, json=payload)
        data = res.json()

        score_text = data["choices"][0]["message"]["content"].strip()
        return float(score_text)

    except:
        return 0.0

# ---------------- FINAL SCORE ----------------
def compute_final_score(res: ModelResponse) -> float:
    latency_score = 1 / (res.latency + 0.01)
    token_score = 1 / (res.tokens_used + 1)

    return (
        0.5 * res.relevance_score +
        0.3 * latency_score +
        0.2 * token_score
    )

# ---------------- CHATGPT ----------------
def call_chatgpt(prompt: str, domain: str):
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": f"You are an expert in {domain}."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200
        }

        start = time.time()
        response = requests.post(OPENAI_URL, headers=headers, json=payload)
        latency = time.time() - start

        data = response.json()

        text = data.get("choices", [{}])[0].get("message", {}).get("content", "Error")
        tokens = data.get("usage", {}).get("total_tokens", 0)

        return ModelResponse("ChatGPT", text, latency, tokens, data)

    except Exception as e:
        st.error(f"ChatGPT Error: {e}")
        return None

# ---------------- GEMINI ----------------
def call_gemini(prompt: str, domain: str):
    try:
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{
                "parts": [{"text": f"[Domain: {domain}] {prompt}"}]
            }],
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
            .get("text", "Error")
        )

        tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)

        return ModelResponse("Gemini Flash", text, latency, tokens, data)

    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return None

# ---------------- COMPARISON ----------------
def compare_models(prompt: str, domain: str) -> List[ModelResponse]:
    results = []

    gpt = call_chatgpt(prompt, domain)
    gemini = call_gemini(prompt, domain)

    for model in [gpt, gemini]:
        if model:
            model.relevance_score = llm_relevance_score(
                prompt, model.response_text, domain
            )
            model.final_score = compute_final_score(model)
            results.append(model)

    return results

# ---------------- RESULTS (UPDATED UI) ----------------
if run_btn:
    if not user_input.strip():
        st.warning("Enter a prompt")
    else:
        with st.spinner("Running models..."):
            results = compare_models(user_input, domain)

        if results:
            st.markdown("## 📊 Model Comparison Results")

            fastest = min(results, key=lambda x: x.latency)
            cheapest = min(results, key=lambda x: x.tokens_used)
            best = max(results, key=lambda x: x.final_score)

            for res in results:
                st.markdown("---")
                st.markdown(f"## 🤖 {res.model_name}")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("⚡ Latency", f"{res.latency:.2f}s")
                col2.metric("🔢 Tokens", res.tokens_used)
                col3.metric("🎯 Relevance", f"{res.relevance_score:.2f}")
                col4.metric("🏆 Score", f"{res.final_score:.3f}")

                if res.model_name == fastest.model_name:
                    st.success("⚡ Fastest Model")

                if res.model_name == cheapest.model_name:
                    st.info("💰 Most Cost Efficient")

                if res.model_name == best.model_name:
                    st.success("🏆 Best Overall Model")

                st.markdown("### 💬 Response")
                st.write(res.response_text)

                with st.expander("🛠 Debug Data"):
                    st.json(res.raw_response)

            st.markdown("---")
            st.markdown("## 🧠 Final Insights")
            st.success(f"🏆 Best Overall Model: {best.model_name}")
            st.info(f"⚡ Fastest Model: {fastest.model_name}")
            st.info(f"💰 Most Cost Efficient: {cheapest.model_name}")

        else:
            st.error("No responses received")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ by NIIVA")
