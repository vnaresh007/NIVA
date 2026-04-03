import time
import os

# OpenAI
from openai import OpenAI

# Anthropic
import anthropic

# Gemini
import google.generativeai as genai


# -------------------------------
# 1. Initialize Clients
# -------------------------------

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# -------------------------------
# 2. Static LLM Configuration
# -------------------------------

LLM_MODELS = {
    "gpt-4": {
        "latency": 0.7,
        "cost": 0.9,
        "domain_relevance": 0.95
    },
    "claude-3": {
        "latency": 0.6,
        "cost": 0.85,
        "domain_relevance": 0.92
    },
    "gemini-pro": {
        "latency": 0.5,
        "cost": 0.6,
        "domain_relevance": 0.88
    },
    "mistral-large": {
        "latency": 0.4,
        "cost": 0.5,
        "domain_relevance": 0.80
    },
    "llama-3": {
        "latency": 0.3,
        "cost": 0.4,
        "domain_relevance": 0.75
    }
}


# -------------------------------
# 3. Weights
# -------------------------------

WEIGHTS = {
    "latency": 0.3,
    "cost": 0.3,
    "domain_relevance": 0.4
}


# -------------------------------
# 4. Prompt Ingestion
# -------------------------------

def ingest_prompt(user_input, metadata):
    return {
        "prompt": user_input,
        "metadata": metadata
    }


# -------------------------------
# 5. Context Enrichment
# -------------------------------

def enrich_context(data):
    return f"""
    [User Tier: {data['metadata'].get('user_tier')}]
    [Region: {data['metadata'].get('region')}]
    [Domain: {data['metadata'].get('domain')}]

    {data['prompt']}
    """.strip()


# -------------------------------
# 6. Scoring Engine
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


# -------------------------------
# 7. Model Selection
# -------------------------------

def select_best_model(scores):
    return max(scores, key=scores.get)


# -------------------------------
# 8. Real LLM Execution (Single Call)
# -------------------------------

def execute_llm(model_name, prompt):

    if model_name == "gpt-4":
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # cost-effective GPT
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    elif model_name == "claude-3":
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    elif model_name == "gemini-pro":
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text

    elif model_name == "mistral-large":
        # Placeholder (you can integrate Mistral API later)
        return "[Mistral Placeholder Response]"

    elif model_name == "llama-3":
        # Placeholder (use Together.ai / Groq later)
        return "[LLaMA Placeholder Response]"

    else:
        return "No valid model selected."


# -------------------------------
# 9. Chatbot Pipeline
# -------------------------------

def chatbot(user_input):

    metadata = {
        "user_tier": "free",
        "region": "global",
        "domain": "general"
    }

    data = ingest_prompt(user_input, metadata)
    enriched_prompt = enrich_context(data)

    scores = score_models()
    best_model = select_best_model(scores)

    print(f"\nSelected Model: {best_model}")  # optional debug

    response = execute_llm(best_model, enriched_prompt)

    return response


# -------------------------------
# 10. Run Chatbot
# -------------------------------

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        reply = chatbot(user_input)
        print("Bot:", reply)
             
