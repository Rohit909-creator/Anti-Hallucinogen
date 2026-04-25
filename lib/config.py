"""
config.py — All tunable knobs in one place.

Edit this file before running any pipeline step.
"""

import torch

# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"

# Dtype for loading the model weights.
# "float16" is the best default for most consumer/cloud GPUs.
# Use "bfloat16" for Ampere/Ada GPUs (A100, H100, RTX 4090).
DTYPE = "float16"   # "float16" | "bfloat16" | "float32"

# ──────────────────────────────────────────────────────────────
# Model Architecture Config
# ──────────────────────────────────────────────────────────────
#
# These two strings tell ActivationExtractor where to plant hooks.
# Use `print(model)` after loading to find the right path for your model.
#
# Common presets:
#
#   Llama 3 / Llama 2 / Mistral / Gemma / Qwen / Phi-3:
#     LAYERS_PATH = "model.layers"
#     FFN_PATH    = "mlp.down_proj"
#
#   GPT-2 / GPT-Neo / GPT-J:
#     LAYERS_PATH = "transformer.h"
#     FFN_PATH    = "mlp.c_proj"
#
#   Falcon (older):
#     LAYERS_PATH = "transformer.h"
#     FFN_PATH    = "mlp.dense_4h_to_h"
#
#   OPT:
#     LAYERS_PATH = "model.decoder.layers"
#     FFN_PATH    = "fc2"
#
#   BLOOM:
#     LAYERS_PATH = "transformer.h"
#     FFN_PATH    = "mlp.dense_4h_to_h"
#

LAYERS_PATH = "model.layers"   # dotted path from model object to the list of transformer layers
FFN_PATH    = "mlp.down_proj"  # dotted path from each layer to the FFN module to hook

# ──────────────────────────────────────────────────────────────
# Step 1 — Data Collection
# ──────────────────────────────────────────────────────────────

OUTPUT_PATH     = "data/consistency_samples.jsonl"

# How many independent completions to sample per question.
# More samples → cleaner consistency labels, but slower.
SAMPLE_NUM      = 10

# Cap the dataset to this many questions (None = full dataset).
MAX_SAMPLES     = 1000

# Generation parameters
MAX_NEW_TOKENS  = 50
TEMPERATURE     = 1.0
TOP_P           = 0.9
TOP_K           = 50

# Judge type: "llm" → Gemini  |  "rule" → exact string match
JUDGE_TYPE      = "llm"

# Gemini API key  (only used when JUDGE_TYPE = "llm")
GEMINI_API_KEY  = ""

# Gemini model for judging
JUDGE_MODEL     = "gemini-2.0-flash-lite"

# How many (question, response) pairs to send in a single judge API call.
# Lower values are safer against rate limits.
JUDGE_BATCH_SIZE = 20

# ──────────────────────────────────────────────────────────────
# Step 2 — Feature Extraction
# ──────────────────────────────────────────────────────────────

ANSWER_TOKENS_PATH  = "data/answer_tokens.jsonl"
ACTIVATIONS_DIR     = "data/activations"
CETT_METHOD         = "mean"   # "mean" | "max"

# ──────────────────────────────────────────────────────────────
# Step 3 — Probe Training
# ──────────────────────────────────────────────────────────────

TRAIN_QIDS_PATH     = "data/train_qids.json"
DETECTOR_PATH       = "models/detector.pt"

# "l1" → sparse model, identifies interpretable H-Neurons
# "l2" → dense model, typically higher accuracy
PENALTY             = "l2"
LAMBDA              = 1e-5   # regularization strength

LR                  = 1e-4
EPOCHS              = 30
BATCH_SIZE          = 512
PATIENCE            = 10     # early stopping patience

NUM_SAMPLES_PER_CLASS = 500  # balanced samples per label for train/test split

# ──────────────────────────────────────────────────────────────
# Step 4 — Inference / Monitor
# ──────────────────────────────────────────────────────────────

# Probe score above which the response is flagged as a hallucination.
HALLUCINATION_THRESHOLD = 0.5

# Self-reflection: maximum rounds of reflection per question.
MAX_RETRIES = 2

# If the corrected response scores below this, stop reflecting early.
REFLECTION_THRESHOLD = 0.3

# ──────────────────────────────────────────────────────────────
# Runtime
# ──────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DTYPE_MAP = {
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "float32":  torch.float32,
}
