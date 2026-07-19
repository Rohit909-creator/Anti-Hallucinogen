"""
step1_collect_data.py — Sample LLM responses and judge them for correctness.

Output: data/consistency_samples.jsonl
  Each line: {"<qid>": {"question": ..., "responses": [...], "judges": [...], "ground_truth": [...]}}

Run:
  python scripts/step1_collect_data.py

  # If Gemini API calls fail with SSL errors on your network, set:
  #   DISABLE_SSL_VERIFY=1 python scripts/step1_collect_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# SSL bypass must be set before any http imports — judge.py reads the env var.
# Uncomment the next line (or set the env var in your shell) if needed:
# os.environ["DISABLE_SSL_VERIFY"] = "1"

from src.data_pipeline import ConsistencySampler

sampler = ConsistencySampler()
sampler.process_data()

# ── Sanity check ────────────────────────────────────────────────────────────
import json, config as cfg
output = cfg.OUTPUT_PATH
if os.path.exists(output):
    with open(output, encoding="utf-8") as f:
        n = sum(1 for l in f if l.strip())
    print(f"\n[step1] Output file: {output}  ({n} records total)")
