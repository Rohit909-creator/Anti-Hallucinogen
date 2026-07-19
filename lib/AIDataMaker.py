# pip install google-genai

import json
import os
import re
import ssl
import sys
import time
import urllib3
import httpx
from google import genai
from google.genai import types

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

original_init = httpx.Client.__init__
def patched_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_init

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

API_KEY       = ""
MODEL         = "gemini-2.0-flash-lite"
OUTPUT_FILE   = "RepoMakingHere\\generated_data.txt"
BATCH_SIZE    = 5
PER_CATEGORY  = 20

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

SCHEMA = """
Return a JSON array of exactly {n} objects. Each object:
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
"""


def build_prompt(category, n):
    return (
        f"You are a dataset generator for hallucination detection research.\n\n"
        f"Category: {category['name']}\n"
        f"Instructions: {category['description']}\n\n"
        f"{SCHEMA.format(n=n)}"
    )


def call_gemini(client, prompt):
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")
    )
    result = ""
    for chunk in client.models.generate_content_stream(model=MODEL, contents=contents, config=cfg):
        if chunk.text:
            result += chunk.text
    return result


def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("[")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "[": depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception:
                        break
    raise ValueError("Could not parse JSON from response")


if __name__ == "__main__":
    client = genai.Client(api_key=API_KEY)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        total = 0
        for cat in CATEGORIES:
            print(f"\n=== {cat['name']} ===", flush=True)
            out.write(f"\n{'='*60}\n{cat['name']}\n{'='*60}\n")
            saved = 0
            counter = 0

            while saved < PER_CATEGORY:
                remaining = PER_CATEGORY - saved
                n = min(BATCH_SIZE, remaining)

                print(f"  Calling Gemini for {n} samples [{saved}/{PER_CATEGORY}] ...", end=" ", flush=True)
                try:
                    raw = call_gemini(client, build_prompt(cat, n))
                    time.sleep(10)
                except Exception as e:
                    print(f"FAILED: {e}", flush=True)
                    break

                try:
                    samples = extract_json(raw)
                    if isinstance(samples, dict):
                        samples = [samples]
                except Exception as e:
                    print(f"JSON error: {e}", flush=True)
                    continue

                batch_saved = 0
                for sample in samples:
                    q = sample.get("question", "").strip()
                    gt = sample.get("ground_truth", [])
                    responses = sample.get("responses", [])
                    if not q or not gt or len(responses) < 2:
                        continue
                    qid = f"synth_{cat['id']}_{counter:05d}"
                    counter += 1
                    saved += 1
                    batch_saved += 1
                    total += 1

                    out.write(f"\n[{qid}]\n")
                    out.write(f"Question: {q}\n")
                    out.write(f"Ground Truth: {', '.join(gt)}\n")
                    out.write("Responses:\n")
                    for r in responses:
                        out.write(f"  [{r.get('label','?')}] {r.get('text','').strip()}\n")
                    out.flush()

                print(f"OK +{batch_saved}  total={saved}", flush=True)

            print(f"  Category done: {saved} samples saved.", flush=True)

    print(f"\nDone. {total} total samples written to {OUTPUT_FILE}", flush=True)
