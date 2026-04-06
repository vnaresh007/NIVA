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
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# -------------------------------
# MODEL CONFIG
# -------------------------------
LLM_MODELS = {
    "gpt-4o-mini": {"latency": 0.7, "cost": 0.9, "domain_relevance": 0.95},
    "gemini-pro": {"latency": 0.5, "cost": 0.6, "domain_relevance": 0.88},
    "claude-3": {"latency": 0.6, "cost": 0.85, "domain_relevance": 0.92},
    "mistral-large": {"latency": 0.4, "cost": 0.5, "domain_relevance": 0.80},
    "llama-3": {"latency": 0.3, "cost": 0.4, "domain_relevance": 0.75}
}

WEIGHTS = {
    "latency": 0.3,
    "cost": 0.3,
    "domain_relevance": 0.4
}

# -------------------------------
# INPUT VALIDATION
# -------------------------------
def validate_prompt(prompt: str) -> str:
    if not prompt or len(prompt.strip()) == 0:
        raise ValueError("Prompt cannot be empty")

    if len(prompt) > 2000:
        raise ValueError("Prompt too long (max 2000 chars)")

    return prompt.strip()

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
# SAFE REQUEST HANDLER
# -------------------------------
def safe_post(url, headers, payload):
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=10
        )

        if response.status_code != 200:
            return f"API Error ({response.status_code})"

        return response.json()

    except requests.exceptions.Timeout:
        return "Request timed out"
    except requests.exceptions.RequestException:
        return "Network error occurred"
    except Exception:
        return "Unexpected error"

# -------------------------------
# OPENAI CALL
# -------------------------------
def call_openai(prompt):
    try:
        prompt = validate_prompt(prompt)

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        data = safe_post(OPENAI_URL, headers, payload)

        if isinstance(data, str):
            return data

        return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")

    except ValueError as ve:
        return str(ve)

# -------------------------------
# GEMINI CALL
# -------------------------------
def call_gemini(prompt):
    try:
        prompt = validate_prompt(prompt)

        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }

        data = safe_post(GEMINI_URL, headers, payload)

        if isinstance(data, str):
            return data

        return (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "No response")
        )

    except ValueError as ve:
        return str(ve)

# -------------------------------
# PLACEHOLDER APIs (Extend Later)
# -------------------------------
def call_claude(prompt):
    return "Claude API not integrated yet"


def call_mistral(prompt):
    return "Mistral API not integrated yet"


def call_llama(prompt):
    return "LLaMA API not integrated yet"

# -------------------------------
# ROUTER
# -------------------------------
def route_request(model, prompt):
    if model == "gpt-4o-mini":
        return call_openai(prompt)

    elif model == "gemini-pro":
        return call_gemini(prompt)

    elif model == "claude-3":
        return call_claude(prompt)

    elif model == "mistral-large":
        return call_mistral(prompt)

    elif model == "llama-3":
        return call_llama(prompt)

    else:
        return "Model not supported"

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("Multi-LLM Chatbot Router")

user_input = st.text_input("Enter your prompt:")

if st.button("Submit"):
    if user_input:
        try:
            validated_prompt = validate_prompt(user_input)

            scores = score_models()
            best_model = select_best_model(scores)

            st.write("### Model Scores")
            st.json(scores)

            st.success(f"Selected Model: {best_model}")

            response = route_request(best_model, validated_prompt)

            st.write("### Response")
            st.write(response)

        except ValueError as e:
            st.error(str(e))
    else:
        st.warning("Please enter a prompt")
