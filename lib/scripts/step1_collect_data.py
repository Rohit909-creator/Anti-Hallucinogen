"""
step1_collect_data.py — Sample LLM responses and judge them for correctness.

Output: data/consistency_samples.jsonl
  Each line: {"<qid>": {"question": ..., "responses": [...], "judges": [...], "ground_truth": [...]}}

Run:
  python scripts/step1_collect_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_pipeline import ConsistencySampler

sampler = ConsistencySampler()
sampler.process_data()
