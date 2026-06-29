"""
ProviderManager — LLM provider with automatic fallback.

Priority order:
  1. Groq          (fastest, primary)
  2. Cerebras      (fallback 1)
  3. AWS Bedrock   (fallback 2 — kicks in when Groq/Cerebras rate-limit)

Bedrock models tried in order (all confirmed working in your account):
  1. amazon.nova-pro-v1:0          (strongest)
  2. meta.llama3-70b-instruct-v1:0 (good quality)
  3. mistral.mixtral-8x7b-instruct-v0:1
  4. meta.llama3-8b-instruct-v1:0  (fast)
  5. mistral.mistral-7b-instruct-v0:2
  6. amazon.nova-lite-v1:0         (lightest fallback)
"""

import json
import os
import time
from dotenv import load_dotenv

load_dotenv()


class ProviderManager:

    def __init__(self) -> None:
        # ── Groq ─────────────────────────────────────────────────────
        self.groq_keys = [
            k.strip()
            for k in os.getenv("GROQ_API_KEYS", "").split(",")
            if k.strip()
        ]
        self.groq_idx = 0

        # ── Cerebras ─────────────────────────────────────────────────
        self.cerebras_keys = [
            k.strip()
            for k in os.getenv("CEREBRAS_API_KEYS", "").split(",")
            if k.strip()
        ]
        self.cerebras_idx = 0

        # ── Bedrock (confirmed working in your account) ───────────────
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
        self.aws_region     = os.getenv("AWS_REGION", "us-east-1").strip()

        self.bedrock_models = [
            "amazon.nova-pro-v1:0",
            "meta.llama3-70b-instruct-v1:0",
            "mistral.mixtral-8x7b-instruct-v0:1",
            "meta.llama3-8b-instruct-v1:0",
            "mistral.mistral-7b-instruct-v0:2",
            "amazon.nova-lite-v1:0",
        ]

        self._bedrock_client = None  # lazy init

    # ── Key rotation ──────────────────────────────────────────────────

    def _next_groq(self) -> str:
        key = self.groq_keys[self.groq_idx]
        self.groq_idx = (self.groq_idx + 1) % len(self.groq_keys)
        return key

    def _next_cerebras(self) -> str:
        key = self.cerebras_keys[self.cerebras_idx]
        self.cerebras_idx = (self.cerebras_idx + 1) % len(self.cerebras_keys)
        return key

    def _get_bedrock_client(self):
        if self._bedrock_client is None:
            import boto3
            self._bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
            )
        return self._bedrock_client

    # ── Provider calls ────────────────────────────────────────────────

    def _try_groq(self, prompt: str, temperature: float) -> str | None:
        if not self.groq_keys:
            return None
        for _ in range(len(self.groq_keys)):
            try:
                from groq import Groq
                client = Groq(api_key=self._next_groq())
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[Groq Failed] {str(e)[:120]}")
        return None

    def _try_cerebras(self, prompt: str) -> str | None:
        if not self.cerebras_keys:
            return None
        for _ in range(len(self.cerebras_keys)):
            try:
                from cerebras.cloud.sdk import Cerebras
                client = Cerebras(api_key=self._next_cerebras())
                response = client.chat.completions.create(
                    model="gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[Cerebras Failed] {str(e)[:120]}")
        return None

    def _try_bedrock(self, prompt: str, temperature: float) -> str | None:
        if not self.aws_access_key or not self.aws_secret_key:
            return None
        try:
            import boto3
        except ImportError:
            print("[Bedrock] boto3 not installed. Run: pip install boto3")
            return None

        client = self._get_bedrock_client()

        for model_id in self.bedrock_models:
            try:
                result = self._invoke_bedrock(client, model_id, prompt, temperature)
                if result:
                    print(f"[Bedrock ✓] {model_id}")
                    return result
            except Exception as e:
                print(f"[Bedrock ✗] {model_id}: {str(e)[:100]}")
                time.sleep(0.5)

        return None

    def _invoke_bedrock(
        self, client, model_id: str, prompt: str, temperature: float
    ) -> str | None:

        # ── Amazon Nova ───────────────────────────────────────────────
        if "amazon.nova" in model_id:
            body = json.dumps({
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]}
                ],
                "inferenceConfig": {
                    "max_new_tokens": 1024,
                    "temperature": max(temperature, 1e-6),
                },
            })
            resp   = client.invoke_model(modelId=model_id, body=body,
                                         contentType="application/json",
                                         accept="application/json")
            result = json.loads(resp["body"].read())
            return result["output"]["message"]["content"][0]["text"]

        # ── Meta Llama ────────────────────────────────────────────────
        elif "meta.llama" in model_id:
            body = json.dumps({
                "prompt": (
                    "<|begin_of_text|>"
                    "<|start_header_id|>user<|end_header_id|>\n"
                    f"{prompt}"
                    "<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n"
                ),
                "max_gen_len": 1024,
                "temperature": max(temperature, 1e-6),
            })
            resp   = client.invoke_model(modelId=model_id, body=body,
                                         contentType="application/json",
                                         accept="application/json")
            result = json.loads(resp["body"].read())
            return result.get("generation", "")

        # ── Mistral ───────────────────────────────────────────────────
        elif "mistral" in model_id:
            body = json.dumps({
                "prompt": f"<s>[INST]{prompt}[/INST]",
                "max_tokens": 1024,
                "temperature": max(temperature, 1e-6),
            })
            resp   = client.invoke_model(modelId=model_id, body=body,
                                         contentType="application/json",
                                         accept="application/json")
            result = json.loads(resp["body"].read())
            return result["outputs"][0]["text"]

        else:
            raise ValueError(f"Unknown model family: {model_id}")

    # ── Public interface ──────────────────────────────────────────────

    def generate(self, prompt: str, temperature: float = 0) -> str:
        """
        Generate using first available provider.
        Groq → Cerebras → Bedrock
        """
        result = self._try_groq(prompt, temperature)
        if result:
            return result

        result = self._try_cerebras(prompt)
        if result:
            return result

        print("[ProviderManager] Groq + Cerebras exhausted → AWS Bedrock")
        result = self._try_bedrock(prompt, temperature)
        if result:
            return result

        raise RuntimeError(
            "All providers exhausted: Groq, Cerebras, and AWS Bedrock all failed."
        )