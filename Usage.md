# H-Neurons: Self-Reflecting Hallucination Detector

An end-to-end system that detects LLM hallucinations using the model's **own internal FFN activations**, then triggers the model to self-correct — no external knowledge base or second model required.

---

## The Core Idea

When a language model hallucinates, specific neurons in its feed-forward layers ("**H-Neurons**") activate differently than when it gives a faithful answer. By training a lightweight linear probe on these activation patterns, you can catch hallucinations at inference time and prompt the model to reconsider.

```
User question
      │
      ▼
┌─────────────────────────────────────────────────────┐
│           LLM generates a response                  │
│                                                     │
│   [Layer 0 FFN] → [Layer 1 FFN] → ... → [Layer N]   │
│       ↓               ↓                   ↓         │
│   activations     activations         activations   │
│            ↘         ↓         ↙                    │
│               CETT Feature Vector                   │
│                  [L × D dims]                       │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
             ┌──────────────────┐
             │  Linear Probe    │  ◄── Trained on labelled (prompt, response) pairs
             │  (H-Neurons)     │
             └──────────────────┘
                        │
             ┌──────────┴──────────┐
             │                     │
        prob < 0.5            prob ≥ 0.5
             │                     │
      ✓ Faithful           ⚠ Hallucination detected
                                   │
                        ┌──────────▼──────────┐
                        │  Self-Reflection     │
                        │  Inject metacognitive│
                        │  prompt → re-score   │
                        └─────────────────────┘
```

**CETT (Cross-token Excitation Telemetry)**: For each FFN down-projection layer, take the absolute activation of every neuron, averaged across all response tokens. Concatenate all layers → one dense feature vector per response.

---

## Pipeline

```
Step 1 — Collect Data         step1_collect_data.py
         TriviaQA  →  LLM (10 samples/question)  →  Gemini judge
         → consistency_samples.jsonl

Step 2 — Extract Activations  step2_extract_activations.py
         (prompt, response, label)  →  CETT features
         → activations.pt  [N × (layers × intermediate_size)]

Step 3 — Train Probe          step3_train_probe.py
         activations.pt  →  LinearProbe(L1/L2)
         → detector.pt   +  H-Neuron report

Step 4 — Deploy Monitor       step4_demo.py
         question  →  response  →  probe  →  (reflect if needed)  →  final answer
```

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Log in to HuggingFace (required for gated models like Llama):
```bash
huggingface-cli login
```

### Run the full pipeline

```bash
# 1. Collect consistency-labelled samples
python scripts/step1_collect_data.py

# 2. Extract FFN activations for each sample
python scripts/step2_extract_activations.py

# 3. Train the hallucination probe
python scripts/step3_train_probe.py

# 4. Run the self-reflecting monitor
python scripts/step4_demo.py
```

### Use the monitor directly in Python

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from src.extraction import ActivationExtractor
from src.probe import load_probe
from src.monitor import SelfReflectingMonitor
import config

# Load your LLM
model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# Load the trained probe
probe = load_probe(config.DETECTOR_PATH, device=config.DEVICE)

# Build the self-reflecting monitor
monitor = SelfReflectingMonitor(
    llm=model,
    probe=probe,
    tokenizer=tokenizer,
    layers_path=config.LAYERS_PATH,
    ffn_path=config.FFN_PATH,
)

result = monitor.generate_with_reflection("Which planet has the most moons?")
print(result["final_answer"])
```

---

## Adapting to Your Own LLM

Edit `config.py`, section **Model Architecture Config**:

```python
# Examples:

# Llama 3 / Mistral / Gemma / Qwen / Phi-3
LAYERS_PATH = "model.layers"
FFN_PATH    = "mlp.down_proj"

# GPT-2 / GPT-Neo
LAYERS_PATH = "transformer.h"
FFN_PATH    = "mlp.c_proj"

# Falcon
LAYERS_PATH = "transformer.h"
FFN_PATH    = "mlp.dense_4h_to_h"

# OPT
LAYERS_PATH = "model.decoder.layers"
FFN_PATH    = "fc2"
```

The `LAYERS_PATH` and `FFN_PATH` tell the `ActivationExtractor` where to plant the forward hooks. Use `print(model)` after loading to find the right path for any unsupported model.

---

## Project Structure

```
├── config.py                     All knobs in one place
├── requirements.txt
├── src/
│   ├── extraction.py             ActivationExtractor + CETT feature extraction
│   ├── judge.py                  GeminiJudge for LLM-as-judge labelling
│   ├── data_pipeline.py          HFSampler + ConsistencySampler
│   ├── probe.py                  HallucinationProbe, training loop, evaluation
│   ├── monitor.py                HallucinationMonitor + SelfReflectingMonitor
│   └── utils.py                  Shared helpers
├── scripts/
│   ├── step1_collect_data.py
│   ├── step2_extract_activations.py
│   ├── step3_train_probe.py
│   └── step4_demo.py
├── data/                         Created at runtime (gitignored)
└── models/                       Saved probes (gitignored)
```

---

## How the Self-Reflection Loop Works

```python
# Round 0: generate initial response
response, prob = monitor._generate_and_score(question)

# If probe fires → inject metacognitive nudge
if prob >= HALLUCINATION_THRESHOLD:
    reflection_prompt = (
        f'Your previous answer was: "{response}"\n'
        f'The hallucination detector flagged this with probability {prob:.3f}.\n'
        f'Please carefully reconsider and provide a corrected answer.'
    )
    # Append to conversation history and re-generate
    response, prob = monitor._generate_and_score(question, history=...)
    # Repeat up to max_retries rounds
```

The model is never told *what* is wrong — only that *something* is likely wrong. This encourages genuine re-evaluation rather than just rephrasing.

---

## Key Configuration

| Parameter | Default | Description |
|---|---|---|
| `MODEL_PATH` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID or local path |
| `SAMPLE_NUM` | `10` | Responses sampled per question for consistency labelling |
| `JUDGE_TYPE` | `"llm"` | `"llm"` (Gemini) or `"rule"` (string match) |
| `PENALTY` | `"l2"` | L1 → sparse H-Neurons, L2 → best accuracy |
| `HALLUCINATION_THRESHOLD` | `0.5` | Probe score above which a warning is issued |
| `MAX_RETRIES` | `2` | Self-reflection rounds on a flagged response |

---

## Results

> Fill in after running step3_train_probe.py and step4_demo.py

| Metric | Value |
|---|---|
| Test Accuracy | — |
| AUROC | — |
| H-Neurons found (L1) | — |
| Self-reflection improvement rate | — |

---

## Citation / Inspiration

This project is inspired by the **H-Neurons** line of research on mechanistic interpretability of hallucinations in large language models. The self-reflection loop is an original extension that uses the probe's output as a metacognitive feedback signal during generation.
