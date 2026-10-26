
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    timeout=20,
    max_retries=0
)

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": "Hello! Explain RAG in two sentences."
            }
        ]
    )

    print("Groq response:")
    print(response.choices[0].message.content)

except Exception as error:
    print("Groq Error:", error)

finally:
    client.close()
