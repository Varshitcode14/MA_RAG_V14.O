import os
import time

from groq import Groq
from cerebras.cloud.sdk import Cerebras

from dotenv import load_dotenv

load_dotenv()


class ProviderManager:

    def __init__(self):

        self.groq_keys = [
            k.strip()
            for k in os.getenv(
                "GROQ_API_KEYS", ""
            ).split(",")
            if k.strip()
        ]

        self.cerebras_keys = [
            k.strip()
            for k in os.getenv(
                "CEREBRAS_API_KEYS", ""
            ).split(",")
            if k.strip()
        ]

        self.groq_idx = 0
        self.cerebras_idx = 0

    def _next_groq(self):

        key = self.groq_keys[self.groq_idx]

        self.groq_idx = (
            self.groq_idx + 1
        ) % len(self.groq_keys)

        return key

    def _next_cerebras(self):

        key = self.cerebras_keys[
            self.cerebras_idx
        ]

        self.cerebras_idx = (
            self.cerebras_idx + 1
        ) % len(self.cerebras_keys)

        return key

    def generate(
        self,
        prompt,
        temperature=0
    ):

        # -------------------
        # Try Groq Keys
        # -------------------

        for _ in range(
            len(self.groq_keys)
        ):

            try:

                key = self._next_groq()

                client = Groq(
                    api_key=key
                )

                response = (
                    client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        temperature=temperature,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                )

                return (
                    response
                    .choices[0]
                    .message
                    .content
                )

            except Exception as e:

                print(
                    f"[Groq Failed] {str(e)[:100]}"
                )

        # -------------------
        # Try Cerebras Keys
        # -------------------

        for _ in range(
            len(self.cerebras_keys)
        ):

            try:

                key = self._next_cerebras()

                client = Cerebras(
                    api_key=key
                )

                response = (
                    client.chat.completions.create(
                        model="gpt-oss-120b",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                )

                return (
                    response
                    .choices[0]
                    .message
                    .content
                )

            except Exception as e:

                print(
                    f"[Cerebras Failed] {str(e)[:100]}"
                )

        raise Exception(
            "All Providers Exhausted"
        )