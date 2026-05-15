# To run this code you need to install the following dependencies:
# pip install google-genai

import json
import os
import re
import ssl
from time import time
import urllib3
import httpx
from google import genai

from google.genai import types


def extract_jsonl(text):
    """
    Extracts a JSON array from the model response.
    Tries full-text parse first, then regex extraction, then line-by-line JSONL.
    """
    # 1. Try parsing the whole text as a JSON array
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Try extracting the first JSON array via regex
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError as e:
            print(f"JSON decoding error (regex): {e}")

    # 3. Fall back to JSONL: parse each line individually
    json_objects = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json_objects.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
    return json_objects if json_objects else None


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkeypatch httpx.Client to disable SSL verification
original_init = httpx.Client.__init__
def patched_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_init

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

CATEGORIES = [
    {
        "id": "exact_constants",
        "name": "Exact physical / mathematical constants",
        "description": (
            "Questions whose correct answer is a well-known physical or mathematical constant "
            "(speed of light, Planck constant, Avogadro number, Boltzmann constant, etc). "
            "FALSE responses: subtly wrong values, wrong units, or over-hedged answers."
        ),
    },
    {
        "id": "nonexistent_api",
        "name": "Non-existent API / library methods",
        "description": (
            "Questions about a Python/JS/SQL function that does NOT exist in the standard library "
            "(e.g. torch.optimize_memory(), list.flatten(), str.reverse()). "
            "TRUE responses: correctly say it does not exist. "
            "FALSE responses: confidently fabricate what the function does."
        ),
    },
    {
        "id": "time_sensitive",
        "name": "Time-sensitive current-state facts",
        "description": (
            "Questions about facts that change over time: heads of state, world records, software versions, "
            "award winners, market caps. "
            "TRUE responses: give correct answer AND flag it may be outdated. "
            "FALSE responses: confidently give stale or fabricated fact with no caveat."
        ),
    },
    {
        "id": "trick_phrasing",
        "name": "Questions with invalid or false premises",
        "description": (
            "Questions that embed a false or nonsensical premise. "
            "TRUE responses: identify and reject the bad premise. "
            "FALSE responses: ignore the bad premise and answer confidently as if it were valid."
        ),
    },
    {
        "id": "medical_dosage",
        "name": "Medical dosage and clinical treatment specifics",
        "description": (
            "Questions asking for specific drug dosages or treatment protocols. "
            "TRUE responses: decline to give dosage and recommend a professional. "
            "FALSE responses: confidently state specific mg/dosing numbers that may be wrong."
        ),
    },
]

SCHEMA = f"""
Return a JSON array of exactly {{n}} objects. Each object:
{{
  "question": "<question string>",
  "ground_truth": ["<correct answer>", "<alias if any>"],
  "responses": [
    {{"text": "<correct LLM completion>", "label": "true"}},
    {{"text": "<correct LLM completion>", "label": "true"}},
    {{"text": "<correct LLM completion>", "label": "true"}},
    {{"text": "<wrong LLM completion>",   "label": "false"}},
    {{"text": "<wrong LLM completion>",   "label": "false"}},
    {{"text": "<wrong LLM completion>",   "label": "false"}}
  ]
}}
Rules: at least 3 true and 3 false responses. No markdown fences. No text outside the JSON array.
Categorize questions into these buckets: {json.dumps(CATEGORIES)}.
Give new examples each time, and do not repeat examples across generations."""



def generate():
    client = genai.Client(
        api_key="",
    )

    model = "gemini-3.1-flash-lite-preview"
    contents = [
        types.Content(
            role="system",
            parts=[
                types.Part.from_text(text=SCHEMA),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""f"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="low",
        ),
        max_output_tokens=2048,
        temperature=0.9,
        top_p=0.9,
    )

    results = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if text := chunk.text:
            print(text, end="")
            results += text

    results = extract_jsonl(results)
    print("\n\nExtracted JSON:")
    print(json.dumps(results, indent=2))

    with open("output.jsonl", "a") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":


    for loop in range(10):
        try:
            print(f"\n\n=== Generation Loop {loop + 1} ===\n")
            generate()
        # rate limits and other API errors will raise an exception, which we catch to avoid crashing the script
        except Exception as e:
            print(f"Error during generation: {e}")
            if "Rate limit" in str(e):
                # if we hit a rate limit, wait for a bit before retrying
                countdown = 30
                print(f"Rate limit hit. Waiting for {countdown} seconds before retrying...")
                time.sleep(countdown)
        