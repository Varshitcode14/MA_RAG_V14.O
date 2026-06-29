"""
Detect available AWS Bedrock models and test a live invocation.

Run from repo root:
    python scripts/test_bedrock.py

This reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION from .env
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

import boto3

ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
REGION     = os.getenv("AWS_REGION", "us-east-1")

print("=" * 60)
print("AWS Bedrock Diagnostics")
print("=" * 60)
print(f"Region          : {REGION}")
print(f"Access Key ID   : {ACCESS_KEY[:8]}... (first 8 chars)")
print(f"Secret Key set  : {'YES' if SECRET_KEY else 'NO'}")

# ── Step 1: Verify identity ───────────────────────────────────────────
print("\n[1] Verifying AWS identity...")
try:
    sts = boto3.client(
        "sts",
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    identity = sts.get_caller_identity()
    print(f"    Account  : {identity['Account']}")
    print(f"    User ARN : {identity['Arn']}")
    print("    Identity OK ✓")
except Exception as e:
    print(f"    FAILED: {e}")
    print("\n    Your credentials are invalid. Go to:")
    print("    https://console.aws.amazon.com/iam/home#/security_credentials")
    print("    and create a new Access Key.")
    sys.exit(1)

# ── Step 2: List foundation models ───────────────────────────────────
print("\n[2] Listing available foundation models in Bedrock...")
try:
    bedrock = boto3.client(
        "bedrock",
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    response = bedrock.list_foundation_models(byOutputModality="TEXT")
    models = response.get("modelSummaries", [])

    # Filter to on-demand inference (no provisioned throughput needed)
    usable = [
        m for m in models
        if "ON_DEMAND" in m.get("inferenceTypesSupported", [])
    ]

    print(f"    Found {len(usable)} text models with ON_DEMAND inference:\n")
    for m in usable:
        print(f"    {m['modelId']:<60} {m['modelName']}")

except Exception as e:
    print(f"    FAILED: {e}")
    usable = []

# ── Step 3: Try invoking candidate models ────────────────────────────
print("\n[3] Testing live invocation with candidate models...")

CANDIDATES = [
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.titan-text-express-v1",
    "amazon.titan-text-lite-v1",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
]

runtime = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

TEST_PROMPT = "Reply with only the word: WORKING"
working_models = []

for model_id in CANDIDATES:
    try:
        # Build body based on model family
        if "anthropic" in model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": TEST_PROMPT}],
            })
        elif "amazon.nova" in model_id:
            body = json.dumps({
                "messages": [{"role": "user", "content": [{"text": TEST_PROMPT}]}],
                "inferenceConfig": {"max_new_tokens": 10},
            })
        elif "amazon.titan" in model_id:
            body = json.dumps({
                "inputText": TEST_PROMPT,
                "textGenerationConfig": {"maxTokenCount": 10},
            })
        elif "meta.llama" in model_id:
            body = json.dumps({
                "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{TEST_PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
                "max_gen_len": 10,
            })
        elif "mistral" in model_id:
            body = json.dumps({
                "prompt": f"<s>[INST]{TEST_PROMPT}[/INST]",
                "max_tokens": 10,
            })
        else:
            continue

        resp = runtime.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())

        # Extract text from response
        if "anthropic" in model_id:
            text = result["content"][0]["text"]
        elif "amazon.nova" in model_id:
            text = result["output"]["message"]["content"][0]["text"]
        elif "amazon.titan" in model_id:
            text = result["results"][0]["outputText"]
        elif "meta.llama" in model_id:
            text = result.get("generation", "")
        elif "mistral" in model_id:
            text = result["outputs"][0]["text"]
        else:
            text = str(result)

        print(f"    ✓  {model_id:<55} → '{text.strip()[:30]}'")
        working_models.append(model_id)

    except Exception as e:
        err = str(e)[:80]
        print(f"    ✗  {model_id:<55} → {err}")

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Working models ({len(working_models)}):")
for m in working_models:
    print(f"  {m}")

if working_models:
    print(f"\nAdd these to provider_manager.py bedrock_models list.")
    print(f"Best candidate: {working_models[0]}")
else:
    print("\nNo working models found. Check:")
    print("  1. Your AWS_REGION in .env matches Bedrock's region")
    print("  2. IAM permissions include bedrock:InvokeModel")
print("=" * 60)