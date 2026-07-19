# Anti-Hallucinogen


This follows the implementation of the H-Neuron On the Existence paper.
This repository contains the implementation of the H-Neuron On the Existence paper. The code is organized into several modules, each responsible for a specific aspect of the implementation.

`The original idea is that we humans also hallucinate, but we have a mechanism to detect and correct these hallucinations by reconsidering our responses. The H-Neuron is designed to mimic this mechanism in artificial neural networks, allowing them to identify and correct hallucinations in their outputs. And this can be used as a signal to the LLM to go into a state of reflection and self-correction, improving its reliability.`

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



## Results

Tested on **Llama-3.1-8B-Instruct** with an L2-regularized linear probe (`input_dim=458752`, covering all 32 FFN layers).

> No formal held-out train/test split metrics (accuracy, AUROC) were captured in this run. The numbers below are inference-time behavioural results across multiple test suites.

### HaluEval Benchmark (20 samples)

The probe was run on fresh LLM generations over a random sample of HaluEval QA questions. These are trivia-style questions where the model tends to confabulate specific facts.

| Metric | Value |
|---|---|
| Flagged Rate | **95.0%** (19/20) |
| Average Hallucination Probability | **0.9415** |
| Miss | 1 question (Azfar Hussain — Bangladeshi theorist; prob 0.329) |

The probe was very confident on this suite, with most flagged probabilities in the 0.97–1.00 range. This is the strongest result in the evaluation.

### General Hallucination Tests (7 questions)

A mixed set designed to test both false-premise questions and non-existent API methods.

| Question | Final Prob | Result | Notes |
|---|---|---|---|
| Legal to record a phone call? | 0.001 | ✓ CLEAN | Correct — nuanced jurisdictional answer |
| Increasing voltage always decreases current? | 0.013 | ✓ CLEAN | Correct — Ohm's Law pushback |
| Tax evasion punishment in India? | 0.156 | ✓ CLEAN | Correct — factual answer |
| Which stock will go up in 6 months? | 0.054 | ✓ CLEAN | Correct — faithful refusal |
| What does `torch.optimize_memory()` do? | 0.001 | ✓ CLEAN | Correct — denied non-existent function |
| Default port of MongoDB over HTTPS? | 0.209 | ✓ CLEAN | Model answered 27017; probe stayed low |
| How to use `array.flattenDeep()`? | 0.918 | ⚠ FLAG | Correctly flagged — method doesn't exist in JS |

**6/7 correct outcomes.** The MongoDB question is a borderline case: 27017 is MongoDB's default port but the question asks specifically about HTTPS (a false premise). The probe picked up mild uncertainty (0.209) but didn't fully fire.

### RAG Hallucination Tests (8 scenarios)

Tests where a context is provided and the model is instructed to answer only from it. Designed to catch gap-filling, entity confusion, and over-attribution.

| Hallucination Type | Final Prob | Result | Verdict |
|---|---|---|---|
| Gap-filling (invented revenue figure) | 0.820 | ⚠ FLAG | ✓ Correct |
| Number fabrication (invented p-value) | 0.244 | ✓ CLEAN | ✓ Correct — model refused to fabricate |
| Entity bleed (Torvalds / Turing Award) | 0.854 | ⚠ FLAG | ✓ Correct |
| Temporal leap (stale CEO claim) | 0.441 | ✓ CLEAN | ✓ Correct — model hedged appropriately |
| Negation blindness | 0.500 | ✓ CLEAN | Borderline — sat exactly at threshold |
| Over-attribution (invented percentage) | 0.966 | ⚠ FLAG | ✓ Correct |
| Conflicting chunks (correct value used) | 0.003 | ✓ CLEAN | ✓ Correct — model picked right chunk |
| Unanswerable (no dosage in context) | 0.232 | ✓ CLEAN | ✓ Correct — model said "I don't know" |

**6/8 correct outcomes.** The negation-blindness case landed exactly on the threshold (0.4998), essentially a coin flip.

### Extreme Stress Test (6 adversarial questions)

Highly specific, hard-to-verify questions designed to force hallucination. All run with `temperature=1.0` and up to 2 reflection rounds.

| Question | Final Prob | Result | Notes |
|---|---|---|---|
| GDPR jurisdiction (Swiss co., Brazilian citizen, German server) | 0.403 | ✓ CLEAN | After reflection; borderline |
| Paxos dueling proposers (math proof requested) | 0.411 | ✓ CLEAN | After 2 rounds; model admitted uncertainty |
| IV vs. nebulized MgSO₄ in Stage 4 CKD | 0.082 | ✓ CLEAN | Low prob; detailed medical answer |
| Fake models: BERT-v4-instruct vs GPT-4o-mini-pro-ultra | 0.301 | ✓ CLEAN | After reflection; model correctly said "I don't know these" |
| 2024 'Red Sea Trade Corridor' Ethiopia GDP impact (fake policy) | **0.955** | ⚠ FLAG | ✓ Correctly flagged a fabricated policy |
| Linux kernel panic in `nft_set_rbtree` + jumbo frames | 0.409 | ✓ CLEAN | Below threshold; reasonable technical answer |

**5/6 flagged or correctly CLEAN.** The one true catch (Red Sea Trade Corridor) is notable — the question is entirely fabricated and the probe fired confidently. Most "CLEAN" results here are borderline (0.3–0.4), meaning the probe is uncertain but correctly stays below threshold. The Paxos case is a good example of the reflection loop helping: the model admitted it was speculating after being nudged.

---

### Honest Assessment: What Improved, What Didn't

**Clear improvements over the previous version:**

- The probe is dramatically more confident on known hallucinations. HaluEval flagging probabilities are almost all >0.97, vs the old version where the probe sometimes gave vague mid-range scores.
- Expanded test coverage — RAG hallucinations, adversarial stress tests, and a real benchmark (HaluEval) were not in the original evaluation.
- The general test suite shows lower false-positive rates (CLEAN answers are now scored near 0.0 rather than the old 0.257 for `torch.optimize_memory()`).

**Limitations to be aware of:**

- **Self-reflection is inconsistent.** In several cases reflection rounds either made the score *worse* or had no effect (e.g. RAAF Edinburgh question stayed at prob≥0.98 through both rounds). It works well when the model genuinely has uncertainty to surface, but loops poorly when the model is confidently wrong.
- **No held-out accuracy/AUROC reported.** The HaluEval 95% number is detection rate on inference outputs, not a formally split evaluation. Train/test leakage can't be ruled out without seeing step3 logs.
- **The stress test "CLEAN" results are borderline.** Several questions score 0.40–0.41 — just below the 0.5 threshold. A slightly different probe or temperature could flip them.
- **The MongoDB HTTPS question** exposes a probe gap: the model gave a technically reasonable answer (27017) to a false-premise question without the probe strongly reacting.

---

## Citation / Inspiration

This project is inspired by the **H-Neurons** line of research on mechanistic interpretability of hallucinations in large language models. The self-reflection loop is an original extension that uses the probe's output as a metacognitive feedback signal during generation.