
import os
import requests
from memory import redis, SESSION_TTL

from groq import Groq
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


# ----------------------------------
# Provider 1: GROQ
# ----------------------------------

def ask_groq(messages):

    if os.getenv("SIMULATE_GROQ_FAILURE") == "true":
        raise ConnectionError("Simulated Groq outage")

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
        timeout=15,
        max_retries=0
    )

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages
        )

        return response.choices[0].message.content

    finally:
        client.close()




# ----------------------------------
# Provider 2: GEMINI 
# ----------------------------------

def ask_gemini(messages):

    contents = []
    system_instructions = []

    for message in messages:

        role = message["role"]
        content = message["content"]

        # Handle system instructions separately
        if role == "system":
            system_instructions.append(content)
            continue

        # Convert OpenAI roles to Gemini roles
        gemini_role = (
            "model" if role == "assistant" else "user"
        )

        contents.append(
            types.Content(
                role=gemini_role,
                parts=[
                    types.Part.from_text(text=content)
                ]
            )
        )

    # Combine system instructions
    system_prompt = "\n\n".join(system_instructions)

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(
            timeout=15000
        )
    )

    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL"),
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )

        return response.text

    finally:
        client.close()

# ----------------------------------
# Provider 3: OPENROUTER
# ----------------------------------

def ask_openrouter(messages):

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": (
                f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
            ),
            "Content-Type": "application/json"
        },
        json={
            "model": "openrouter/free",
            "messages": messages
        },
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]


# ----------------------------------
# AUTOMATIC LLM ROUTER
# ----------------------------------

def ask_ai(messages, conversation_id):

    providers = {
        "Groq": ("GROQ_API_KEY", ask_groq),
        "Gemini": ("GEMINI_API_KEY", ask_gemini),
        "OpenRouter": ("OPENROUTER_API_KEY", ask_openrouter)
    }

    provider_key = f"chat:{conversation_id}:provider"
    active_provider = redis.get(provider_key)

    if isinstance(active_provider, bytes):
        active_provider = active_provider.decode()

    order = list(providers.keys())

    if active_provider in providers:
        order.remove(active_provider)
        order.insert(0, active_provider)

    for name in order:
        key_name, handler = providers[name]

        if not os.getenv(key_name):
            continue

        try:
            answer = handler(messages)

            if not isinstance(answer, str) or not answer.strip():
                continue

            redis.setex(
                provider_key,
                SESSION_TTL,
                name
            )

            return answer, name

        except Exception:
            continue

    return None, None
