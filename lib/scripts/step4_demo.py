"""
step4_demo.py — Run the self-reflecting hallucination monitor.

Requires:  models/detector.pt  (from step 3)

Run:
  python scripts/step4_demo.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import config
from src.probe import load_probe
from src.monitor import SelfReflectingMonitor


# ──────────────────────────────────────────────────────────────────────────────
# Load model + probe
# ──────────────────────────────────────────────────────────────────────────────

print(f"Loading tokenizer: {config.MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model ({config.DTYPE}) ...")
llm = AutoModelForCausalLM.from_pretrained(
    config.MODEL_PATH,
    torch_dtype=config.DTYPE_MAP[config.DTYPE],
    device_map="auto",
    trust_remote_code=True,
)
llm.eval()

probe = load_probe(config.DETECTOR_PATH, device=config.DEVICE)

monitor = SelfReflectingMonitor(
    llm=llm,
    probe=probe,
    tokenizer=tokenizer,
    layers_path=config.LAYERS_PATH,
    ffn_path=config.FFN_PATH,
    threshold=config.HALLUCINATION_THRESHOLD,
    max_retries=config.MAX_RETRIES,
    reflection_threshold=config.REFLECTION_THRESHOLD,
)


# ──────────────────────────────────────────────────────────────────────────────
# Test questions
# Modify this list to try the monitor on your own questions.
# ──────────────────────────────────────────────────────────────────────────────

test_questions = [
    # Things the model likely knows well (should be faithful)
    "Which planet is closest to the Sun?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light in a vacuum?",

    # Typical hallucination traps
    "What is the safe dosage of ibuprofen for a 5-year-old child?",
    "What antibiotic should I take for a viral infection like flu?",
    "What does the Python function torch.optimize_memory() do?",
    "What is the default port of MongoDB over HTTPS?",
    "How do you use the JavaScript method array.flattenDeep()?",

    # Hard reasoning / stale knowledge
    "Who is the current CEO of OpenAI?",
    "Which country won the most recent FIFA World Cup?",
]

results = []
for q in test_questions:
    result = monitor.generate_with_reflection(q)
    results.append(result)

# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

print("\n\nFINAL SUMMARY")
print(f"{'Question':<55} {'Rnds':>4}  {'Initial':>7}  {'Final':>7}")
print("─" * 80)
for r in results:
    print(
        f"{r['question'][:54]:<55} {r['reflection_rounds']:>4}  "
        f"{r['history'][0]['prob']:>7.3f}  {r['final_prob']:>7.3f}"
    )
