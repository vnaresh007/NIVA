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

# ---------------- HEADER ----------------
st.markdown("# 🤖 NIIVA")
st.markdown("### AI Model Evaluation & Comparison Platform")

# ---------------- DOMAIN SELECTION ----------------
st.markdown("### 🌐 Select Domain")
domain = st.selectbox(
    "Choose domain:",
    ["General", "Healthcare", "Finance", "Legal", "Coding"]
)

# ---------------- INPUT ----------------
st.markdown("### 🧠 Enter Prompt")
user_input = st.text_area("Type your query:", height=120)
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
