"""
utils.py — Shared helper functions.
"""

import os
import re
import string
from typing import Set


def normalize_answer(s: str) -> str:
    """
    Lowercase, remove articles/punctuation, and collapse whitespace.
    Used for rule-based answer matching against TriviaQA ground-truth aliases.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def handle_punc(text: str) -> str:
        exclude = set(string.punctuation + "''´`")
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    if not s:
        return ""
    return ' '.join(
        remove_articles(handle_punc(str(s).lower().replace('_', ' '))).split()
    ).strip()


def load_existing_qids(path: str) -> Set[str]:
    """
    Read already-processed question IDs from a .jsonl file.
    Allows resuming a crashed collection run without re-processing questions.
    """
    import json
    if not os.path.exists(path):
        return set()
    qids: Set[str] = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                qids.update(data.keys())
            except Exception:
                continue
    return qids
