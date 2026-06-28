import os
from dotenv import load_dotenv

load_dotenv()


def test_groq():
    try:
        from groq import Groq

        key = os.getenv("GROQ_API_KEYS").split(",")[0].strip()
        client = Groq(api_key=key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Reply with API WORKING"}],
            max_tokens=10,
        )

        print("\n=== GROQ ===")
        print("✅ API Working")
        print("Response:", response.choices[0].message.content)

    except Exception as e:
        print("\n=== GROQ ===")
        print("❌ Failed:", e)


def test_cerebras():
    try:
        from cerebras.cloud.sdk import Cerebras

        key = os.getenv("CEREBRAS_API_KEYS").split(",")[0].strip()
        client = Cerebras(api_key=key)

        response = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": "Reply with API WORKING"}],
            max_tokens=10,
        )

        print("\n=== CEREBRAS ===")
        print("✅ API Working")
        print(response)

    except Exception as e:
        print("\n=== CEREBRAS ===")
        print("❌ Failed:", e)


def test_openrouter():
    try:
        from openai import OpenAI

        key = os.getenv("OPENROUTER_API_KEYS").split(",")[0].strip()

        client = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
        )

        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",
            messages=[{"role": "user", "content": "Reply with API WORKING"}],
            max_tokens=10,
        )

        print("\n=== OPENROUTER ===")
        print("✅ API Working")
        print("Response:", response.choices[0].message.content)

    except Exception as e:
        print("\n=== OPENROUTER ===")
        print("❌ Failed:", e)


if __name__ == "__main__":
    test_groq()
    test_cerebras()
    test_openrouter()