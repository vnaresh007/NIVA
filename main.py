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
    raise ValueError("API keys not found. Check your .env file.")

# -------------------------------
# API ENDPOINTS
# -------------------------------
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

# -------------------------------
# LLM CONFIG
# -------------------------------
LLM_MODELS = {
    "gpt-4": {"latency": 0.7, "cost": 0.9, "domain_relevance": 0.95},
   # "claude-3": {"latency": 0.6, "cost": 0.85, "domain_relevance": 0.92},
    "gemini-pro": {"latency": 0.5, "cost": 0.6, "domain_relevance": 0.88},
   # "mistral-large": {"latency": 0.4, "cost": 0.5, "domain_relevance": 0.80},
    # "llama-3": {"latency": 0.3, "cost": 0.4, "domain_relevance": 0.75}
}

WEIGHTS = {"latency": 0.3, "cost": 0.3, "domain_relevance": 0.4}

# -------------------------------
# SCORING
# -------------------------------
def score_models():
    scores = {}
    for model, values in LLM_MODELS.items():
        score = (
            WEIGHTS["latency"] * (1 - values["latency"]) +
            WEIGHTS["cost"] * (1 - values["cost"]) +
            WEIGHTS["domain_relevance"] * values["domain_relevance"]
        )
        scores[model] = round(score, 4)
    return scores

def select_best_model(scores):
    return max(scores, key=scores.get)

# -------------------------------
# REAL API CALLS
# -------------------------------
def call_openai(prompt):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(OPENAI_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


def call_gemini(prompt):
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(GEMINI_URL, headers=headers, json=payload)
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


# -------------------------------
# EXECUTION (SINGLE MODEL ONLY)
# -------------------------------
def execute_llm(model_name, prompt):

    if model_name == "gpt-4":
        return call_openai(prompt)

    elif model_name == "gemini-pro":
        return call_gemini(prompt)

    elif model_name == "claude-3":
        return "[Claude API not connected yet]"

    elif model_name == "mistral-large":
        return "[Mistral API not connected yet]"

    elif model_name == "llama-3":
        return "[LLaMA API not connected yet]"

    else:
        return "No valid model selected."


# -------------------------------
# CHATBOT PIPELINE
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

domain = st.sidebar.selectbox("Domain", ["general", "finance", "coding"])
user_tier = st.sidebar.selectbox("User Tier", ["free", "pro"])
region = st.sidebar.selectbox("Region", ["global", "asia"])

# Chat input
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

    # Selected model
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
