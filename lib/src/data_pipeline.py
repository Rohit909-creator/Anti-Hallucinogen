"""
data_pipeline.py — Data collection: sample LLM responses and label them.

Classes:
  HFSampler          — wraps a HuggingFace CausalLM for batched sampling
  ConsistencySampler — full pipeline from dataset → labelled .jsonl
"""

import json
import os
from collections import defaultdict
from typing import List, Set

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import config
from src.judge import GeminiJudge
from src.utils import normalize_answer, load_existing_qids


# ──────────────────────────────────────────────────────────────────────────────

class HFSampler:
    """
    Wraps a HuggingFace CausalLM for batched chat-style sampling.

    Tokenizes the prompt once, tiles it `n` times along the batch dimension,
    and calls `generate()` a single time — giving `n` independent samples
    in one forward pass instead of `n` separate calls.
    """

    def __init__(self):
        print(f"Loading tokenizer: {config.MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_PATH, trust_remote_code=True
        )

        # Ensure BOS/EOS tokens are set (important for Llama 3)
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = "<|begin_of_text|>"
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|end_of_text|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model ({config.DTYPE}) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_PATH,
            dtype=config.DTYPE_MAP[config.DTYPE],
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("Model ready.\n")

    @torch.no_grad()
    def sample(self, messages: List[dict], n: int) -> List[str]:
        """
        Generate `n` independent completions for the given chat messages.

        Args:
            messages: Chat-format list e.g. [{"role": "user", "content": "..."}]
            n:        Number of completions.

        Returns:
            List of `n` decoded response strings (prompt tokens stripped).
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(prompt, return_tensors="pt", padding=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        input_ids      = enc["input_ids"].repeat(n, 1)
        attention_mask = enc["attention_mask"].repeat(n, 1)
        prompt_len     = input_ids.shape[1]

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        new_tokens = outputs[:, prompt_len:]
        decoded    = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [r.strip() for r in decoded]


# ──────────────────────────────────────────────────────────────────────────────

class ConsistencySampler:
    """
    Full data-collection pipeline.

    For each HaluEval QA question:
      1. Sample SAMPLE_NUM responses from the LLM.
      2. Judge each response using Gemini (or rule-based fallback).
      3. Append the result to OUTPUT_PATH as newline-delimited JSON.

    Already-processed qids are skipped automatically, so interrupted
    runs can be resumed safely.
    """

    _UNCERTAIN_TERMS = [
        "don't know", "cannot", "not provided",
        "no information", "i'm not sure", "unclear",
    ]

    def __init__(self):
        self.sampler = HFSampler()
        self.judge   = GeminiJudge(
            api_key=config.GEMINI_API_KEY,
            model_name=config.JUDGE_MODEL,
        ) if config.JUDGE_TYPE == "llm" else None

    # ------------------------------------------------------------------

    def process_data(self) -> None:
        from datasets import load_dataset

        print(f"Loading HaluEval ({config.DATASET_SUBSET}) ...")
        ds = load_dataset(config.DATASET_NAME, config.DATASET_SUBSET, split=config.DATASET_SPLIT)
        if config.MAX_SAMPLES is not None:
            ds = ds.select(range(config.MAX_SAMPLES))
        print(f"Loaded {len(ds)} questions.\n")

        processed_qids = load_existing_qids(config.OUTPUT_PATH)
        already_done   = len(processed_qids)
        if already_done:
            print(f"Resume: {already_done} questions already written to "
                  f"{config.OUTPUT_PATH} — skipping those.")
        os.makedirs(os.path.dirname(config.OUTPUT_PATH) or ".", exist_ok=True)

        suffix  = "Respond with the answer only, without any explanation."
        counts  = {"written": 0, "true": 0, "false": 0, "error": 0}
        buffer: list = []

        def flush(buf: list) -> None:
            if not buf:
                return
            with open(config.OUTPUT_PATH, 'a', encoding='utf-8') as f:
                # Build unique (question, response) pairs needing LLM judging
                cache_map = [{} for _ in buf]
                llm_items = []

                for bi, entry in enumerate(buf):
                    for resp in entry["responses"]:
                        if resp in cache_map[bi]:
                            continue
                        if self._is_uncertain(resp):
                            cache_map[bi][resp] = "uncertain"
                        elif config.JUDGE_TYPE == "rule":
                            cache_map[bi][resp] = self._rule_judge(resp, entry["norm_gts"])
                        else:
                            llm_items.append({
                                "buf_idx":  bi,
                                "response": resp,
                                "question": entry["question"],
                                # Gemini sees a readable string, not a Python list repr
                                "answers":  ", ".join(entry["raw_aliases"]),
                            })

                # Call LLM judge in chunks
                if config.JUDGE_TYPE == "llm" and llm_items and self.judge:
                    for start in range(0, len(llm_items), config.JUDGE_BATCH_SIZE):
                        chunk    = llm_items[start: start + config.JUDGE_BATCH_SIZE]
                        verdicts = self.judge.judge_batch(chunk)
                        for item, verdict in zip(chunk, verdicts):
                            cache_map[item["buf_idx"]][item["response"]] = verdict

                for bi, entry in enumerate(buf):
                    judges = [cache_map[bi].get(r, "error") for r in entry["responses"]]
                    counts["true"]  += judges.count("true")
                    counts["false"] += judges.count("false")
                    counts["error"] += sum(
                        1 for j in judges if j not in ("true", "false")
                    )
                    counts["written"] += 1

                    record = {
                        entry["qid"]: {
                            "question":     entry["question"],
                            "responses":    entry["responses"],
                            "judges":       judges,
                            "ground_truth": list(set(entry["raw_aliases"])),
                        }
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    processed_qids.add(entry["qid"])

        with tqdm(total=len(ds), desc="Processing questions") as pbar:
            for idx, item in enumerate(ds):
                pbar.update(1)
                qid = str(idx)
                if qid in processed_qids:
                    continue

                question = item.get("question", "").strip()
                if not question:
                    continue

                # HaluEval qa_samples schema:
                #   knowledge  — background context
                #   question   — the question
                #   answer     — correct answer
                #   hallucination — a hallucinated answer (unused here)
                right_answer = item.get("answer", "").strip()
                if not right_answer:
                    continue

                knowledge = item.get("knowledge", "").strip()

                raw_aliases: List[str] = [right_answer]I
                norm_gts = [normalize_answer(right_answer)]

                # Include knowledge context so LLaMA has the same info
                if knowledge:
                    user_content = (
                        f"Context: {knowledge}\n\n"
                        f"Question: {question} {suffix}"
                    )
                else:
                    user_content = f"{question} {suffix}"

                messages = [{"role": "user", "content": user_content}]
                try:
                    responses = self.sampler.sample(messages, n=config.SAMPLE_NUM)
                except Exception as e:
                    tqdm.write(f"  [sampling error] qid={qid}: {e}")
                    continue

                if len(responses) < config.SAMPLE_NUM:
                    continue

                buffer.append({
                    "qid":       qid,
                    "question":  user_content,
                    "responses": responses,
                    "raw_aliases": raw_aliases,
                    "norm_gts":  norm_gts,
                })

                if len(buffer) >= config.JUDGE_BATCH_SIZE:
                    flush(buffer)
                    buffer.clear()
                    tqdm.write(
                        f"  Written: {counts['written']} | "
                        f"true: {counts['true']} | "
                        f"false: {counts['false']} | "
                        f"error/uncertain: {counts['error']}"
                    )

            flush(buffer)

        if counts["written"] == 0 and already_done:
            print(f"\nAll {already_done} questions were already processed — nothing new to write.")
            print(f"Delete or rename {config.OUTPUT_PATH} to reprocess from scratch.")
        else:
            t = counts['true']
            f = counts['false']
            e = counts['error']
            total = t + f + e or 1
            print(
                f"\nDone.  Records written: {counts['written']}\n"
                f"  Judge breakdown — "
                f"true: {t} ({100*t//total}%) | "
                f"false: {f} ({100*f//total}%) | "
                f"error/uncertain: {e} ({100*e//total}%)"
            )

    # ------------------------------------------------------------------

    def _rule_judge(self, response: str, norm_gts: List[str]) -> str:
        norm_res = normalize_answer(response)
        return "true" if any(gt and gt in norm_res for gt in norm_gts) else "false"

    def _is_uncertain(self, response: str) -> bool:
        lower = response.lower()
        return any(term in lower for term in self._UNCERTAIN_TERMS)
