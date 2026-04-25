"""
step2_extract_activations.py — Extract CETT feature vectors for each sample.

Reads:  data/consistency_samples.jsonl  (from step 1)
Writes: data/answer_tokens.jsonl        (flattened per-response records)
        data/activations/activations.pt (tensor of shape [N, L*D])

Run:
  python scripts/step2_extract_activations.py
"""

import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import config
from src.extraction import ActivationExtractor, get_neuron_activations, compute_cett


# ──────────────────────────────────────────────────────────────────────────────
# Step A: Flatten consistency_samples.jsonl → answer_tokens.jsonl
# ──────────────────────────────────────────────────────────────────────────────

def flatten_to_answer_tokens(
    input_path: str = config.OUTPUT_PATH,
    output_path: str = config.ANSWER_TOKENS_PATH,
) -> None:
    """
    Convert the per-question consistency records into per-response records
    that `extract_activations` can process.

    Output fields per line:
      qid, question, response, label (0=faithful / 1=hallucinated), answer_tokens
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            for qid, data in raw.items():
                for response, judge in zip(data["responses"], data["judges"]):
                    record = {
                        "qid":          qid,
                        "question":     data["question"],
                        "response":     response,
                        "label":        1 if judge == "false" else 0,
                        "answer_tokens": [],   # [] = use all response tokens
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"answer_tokens.jsonl written → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Step B: Extract activations
# ──────────────────────────────────────────────────────────────────────────────

def extract_activations(
    jsonl_path: str = config.ANSWER_TOKENS_PATH,
    output_dir: str = config.ACTIVATIONS_DIR,
) -> None:
    """Load the LLM, hook into its FFN layers, and save CETT feature tensors."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {config.MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model ({config.DTYPE}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH,
        torch_dtype=config.DTYPE_MAP[config.DTYPE],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model ready.\n")

    extractor  = ActivationExtractor(model, config.LAYERS_PATH, config.FFN_PATH)
    features, labels, qids = [], [], []
    errors = 0

    with open(jsonl_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in tqdm(lines, desc="Extracting activations"):
        sample      = json.loads(line)
        prompt      = sample.get("question") or sample.get("prompt", "")
        response    = sample.get("response", "")
        label       = sample.get("label", -1)
        qid         = sample.get("qid", "")
        tok_indices = sample.get("answer_tokens") or None

        try:
            layer_acts  = get_neuron_activations(extractor, tokenizer, prompt, response, tok_indices)
            feature_vec = compute_cett(layer_acts, method=config.CETT_METHOD)
            features.append(feature_vec)
            labels.append(label)
            qids.append(qid)
        except Exception as e:
            tqdm.write(f"  [error] qid={qid}: {e}")
            errors += 1

    if not features:
        print("No features extracted. Check your input file.")
        return

    features_t = torch.stack(features)
    labels_t   = torch.tensor(labels)

    save_path = f"{output_dir}/activations.pt"
    torch.save({"features": features_t, "labels": labels_t, "qids": qids}, save_path)

    print(f"\nSaved → {save_path}")
    print(f"  Samples     : {len(features_t)}")
    print(f"  Feature dim : {features_t.shape[1]}")
    print(f"  Hallucinated: {labels_t.sum().item()}  |  Faithful: {(labels_t == 0).sum().item()}")
    print(f"  Errors skipped: {errors}")

    # Free GPU memory
    del model, extractor
    gc.collect()
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    flatten_to_answer_tokens()
    extract_activations()
