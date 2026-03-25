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
Built with ❤️ by NAIVA Team
""")
