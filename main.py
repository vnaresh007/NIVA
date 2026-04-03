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

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=payload)

        # Debug (optional)
        print("Status:", response.status_code)
        print("Response:", response.text)

        data = response.json()

        # ✅ Success
        if "candidates" in data:
            candidate = data["candidates"][0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            if parts and "text" in parts[0]:
                return parts[0]["text"]
            else:
                return "Gemini returned empty response."

        # ❌ API error
        elif "error" in data:
            return f"Gemini API Error: {data['error'].get('message')}"

        # ❌ Unexpected
        else:
            return f"Unexpected Gemini response: {data}"

    except Exception as e:
        return f"Gemini request failed: {str(e)}"


# -------------------------------
# EXECUTION (Single Model Only)
# -------------------------------
def execute_llm(model, prompt):
    if model == "gpt-4":
        return call_openai(prompt)

    elif model == "gemini-pro":
        return call_gemini(prompt)

    else:
        return f"[{model}] not connected yet"


# -------------------------------
# CHAT PIPELINE
# -------------------------------
def chatbot(prompt, metadata):

    enriched_prompt = f"""
    [User Tier: {metadata['user_tier']}]
    [Region: {metadata['region']}]
    [Domain: {metadata['domain']}]

    {prompt}
    """

    scores = score_models()
    best_model = select_best_model(scores)

    response = execute_llm(best_model, enriched_prompt)

    return response, best_model, scores


# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="LLM Router", layout="wide")

st.title("🌍 Intelligent LLM Routing Chatbot")

# Sidebar
st.sidebar.header("⚙️ Configuration")

domain = st.sidebar.selectbox(
    "Domain",
    ["coding", " legal ", " healthcare ","finance"]   # ❌ removed "general"
)

user_tier = st.sidebar.selectbox(
    "User Tier",
    ["free", "pro"]
)

region = st.sidebar.selectbox(
    "Region",
    ["global", "asia"]
)

# Chat
st.subheader("💬 Chat")

user_input = st.text_input("Enter your prompt:")

if st.button("Send"):

    metadata = {
        "domain": domain,
        "user_tier": user_tier,
        "region": region
    }

    response, selected_model, scores = chatbot(user_input, metadata)

    # Response
    st.subheader("🤖 Response")
    st.write(response)

    # Selected Model
    st.subheader("🏆 Selected Model")
    st.success(selected_model)

    # Scores
    st.subheader("📊 Model Scores")

    score_data = []
    for model, score in scores.items():
        score_data.append({
            "Model": model,
            "Score": score,
            "Latency": LLM_MODELS[model]["latency"],
            "Cost": LLM_MODELS[model]["cost"],
            "Domain Relevance": LLM_MODELS[model]["domain_relevance"]
        })

    st.dataframe(score_data, use_container_width=True)

    # Chart
    st.subheader("📈 Score Visualization")
    st.bar_chart({model: score for model, score in scores.items()})
