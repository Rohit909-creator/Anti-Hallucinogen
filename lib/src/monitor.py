"""
monitor.py — Inference-time hallucination detection and self-reflection.

Classes:
  HallucinationMonitor   — generate + score; warn if probe fires
  SelfReflectingMonitor  — generate + score + reflect if needed
"""

from typing import List, Optional

import torch

import config
from src.extraction import ActivationExtractor, compute_cett


_REFLECTION_SYSTEM_PROMPT = (
    "You are a careful and accurate assistant. "
    "When you are informed that your previous answer may contain a hallucination, "
    "you MUST critically re-examine it.  Ask yourself:\n"
    "  - Am I confident this fact is correct?\n"
    "  - Could I be confusing similar names, dates, or places?\n"
    "  - What is the most accurate answer I can give?\n"
    "Then provide a corrected, more careful answer."
)


# ──────────────────────────────────────────────────────────────────────────────

class HallucinationMonitor:
    """
    Generates a response to a question and scores it for hallucination.

    Workflow:
      1. Generate the response with greedy decoding.
      2. Run ONE forward pass on the full (prompt + response) sequence to
         capture FFN activations — identical to how features were extracted
         during probe training.
      3. Compute the CETT feature vector and pass it to the probe.
      4. Print a warning if the probability exceeds `threshold`.

    Args:
        llm:         Loaded HuggingFace CausalLM.
        probe:       Trained HallucinationProbe (from probe.py).
        tokenizer:   Matching HuggingFace tokenizer.
        layers_path: See ActivationExtractor docs.
        ffn_path:    See ActivationExtractor docs.
        threshold:   Probe score above which a response is flagged.
    """

    def __init__(
        self,
        llm,
        probe,
        tokenizer,
        layers_path: str = config.LAYERS_PATH,
        ffn_path: str = config.FFN_PATH,
        threshold: float = config.HALLUCINATION_THRESHOLD,
    ):
        self.llm         = llm
        self.probe       = probe
        self.tokenizer   = tokenizer
        self.layers_path = layers_path
        self.ffn_path    = ffn_path
        self.threshold   = threshold

    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_with_warning(
        self,
        question: str,
        max_new_tokens: int = 100,
    ) -> tuple:
        """
        Generate a response and print a hallucination warning if needed.

        Returns:
            (response_str, hallucination_probability)
        """
        suffix   = "Respond with the answer only, without any explanation."
        messages = [{"role": "user", "content": f"{question} {suffix}"}]

        response, output_ids, prompt_len = self._generate(messages, max_new_tokens)

        if output_ids.shape[1] <= prompt_len:
            print("(no response generated)")
            return response, 0.0

        prob = self._score(output_ids, prompt_len)

        print(f"\n{'─'*55}")
        print(f"  Q: {question}")
        print(f"  A: {response}")
        print(f"{'─'*55}")
        if prob >= self.threshold:
            tag = "HIGH" if prob > 0.8 else "MODERATE"
            print(f"  ⚠  HALLUCINATION WARNING [{tag}]  prob={prob:.3f}")
        else:
            print(f"  ✓  Response looks faithful  prob={prob:.3f}")
        print(f"{'─'*55}\n")

        return response, prob

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate(self, messages: List[dict], max_new_tokens: int):
        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc        = self.tokenizer(prompt_str, return_tensors="pt").to(self.llm.device)
        prompt_len = enc["input_ids"].shape[1]

        output_ids = self.llm.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(
            output_ids[0, prompt_len:], skip_special_tokens=True
        ).strip()

        return response, output_ids, prompt_len

    @torch.no_grad()
    def _score(self, output_ids: torch.Tensor, prompt_len: int) -> float:
        """Run one forward pass and return probe hallucination probability."""
        extractor = ActivationExtractor(self.llm, self.layers_path, self.ffn_path)
        extractor.register_hooks()
        self.llm(output_ids)
        extractor.remove_hooks()

        resp_indices = list(range(prompt_len, output_ids.shape[1]))
        sliced = {
            idx: act[0, resp_indices, :]
            for idx, act in extractor.activations.items()
        }

        feat = compute_cett(sliced, method="mean").unsqueeze(0).to(config.DEVICE)
        self.probe.eval()
        return torch.sigmoid(self.probe(feat)).item()


# ──────────────────────────────────────────────────────────────────────────────

class SelfReflectingMonitor(HallucinationMonitor):
    """
    Extends HallucinationMonitor with a metacognitive self-reflection loop.

    If the probe flags the initial response, this monitor injects a
    reflection prompt into the conversation history, asking the model to
    reconsider WITHOUT telling it what is wrong — only that something likely
    is wrong.  This encourages genuine re-evaluation rather than rephrasing.

    Args:
        max_retries:          Maximum reflection rounds per question.
        reflection_threshold: If the corrected response scores below this,
                              stop reflecting early (confidence restored).
    """

    def __init__(
        self,
        llm,
        probe,
        tokenizer,
        layers_path: str = config.LAYERS_PATH,
        ffn_path: str = config.FFN_PATH,
        threshold: float = config.HALLUCINATION_THRESHOLD,
        max_retries: int = config.MAX_RETRIES,
        reflection_threshold: float = config.REFLECTION_THRESHOLD,
    ):
        super().__init__(llm, probe, tokenizer, layers_path, ffn_path, threshold)
        self.max_retries          = max_retries
        self.reflection_threshold = reflection_threshold

    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_with_reflection(
        self,
        question: str,
        max_new_tokens: int = 100,
    ) -> dict:
        """
        Generate a response, score it, and reflect up to `max_retries` times
        if the probe fires.

        Returns:
            {
              "question":          str,
              "final_answer":      str,
              "final_prob":        float,
              "reflection_rounds": int,
              "history": [
                  {"answer": str, "prob": float, "reflected": bool}, ...
              ]
            }
        """
        suffix   = "Respond with the answer only, without any explanation."
        messages = [
            {"role": "system", "content": _REFLECTION_SYSTEM_PROMPT},
            {"role": "user",   "content": f"{question} {suffix}"},
        ]

        history = []
        rounds  = 0

        # ── Round 0 ──────────────────────────────────────────────────
        response, output_ids, prompt_len = self._generate(messages, max_new_tokens)
        prob = self._score(output_ids, prompt_len)
        history.append({"answer": response, "prob": prob, "reflected": False})
        self._print_round(0, question, response, prob)

        # ── Reflection loop ───────────────────────────────────────────
        while prob >= self.threshold and rounds < self.max_retries:
            rounds += 1
            tag = "HIGH" if prob > 0.8 else "MODERATE"
            reflection = (
                f'Your previous answer was: "{response}"\n\n'
                f"The hallucination detector flagged this response with "
                f"{tag} confidence (probability={prob:.3f}).\n"
                f"This means there is a significant chance your answer contains "
                f"an inaccuracy — perhaps a wrong name, date, place, or fact.\n\n"
                f"Please carefully reconsider.  If you are not certain, say so.  "
                f"Provide your best corrected answer."
            )
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user",      "content": reflection})

            response, output_ids, prompt_len = self._generate(messages, max_new_tokens)
            prob = self._score(output_ids, prompt_len)
            history.append({"answer": response, "prob": prob, "reflected": True})
            self._print_round(rounds, question, response, prob, is_reflection=True)

            if prob < self.reflection_threshold:
                break

        self._print_summary(history)

        return {
            "question":          question,
            "final_answer":      response,
            "final_prob":        prob,
            "reflection_rounds": rounds,
            "history":           history,
        }

    # ------------------------------------------------------------------

    def _print_round(
        self, round_n, question, response, prob, is_reflection=False
    ) -> None:
        label = f"Reflection round {round_n}" if is_reflection else "Initial response"
        print(f"\n{'─'*60}")
        print(f"  [{label}]")
        if round_n == 0:
            print(f"  Q: {question}")
        print(f"  A: {response}")
        if prob >= self.threshold:
            tag = "HIGH" if prob > 0.8 else "MODERATE"
            print(f"  ⚠  Hallucination detected [{tag}]  prob={prob:.3f}")
        else:
            print(f"  ✓  Looks faithful  prob={prob:.3f}")
        print(f"{'─'*60}")

    def _print_summary(self, history: list) -> None:
        if len(history) == 1:
            return
        print(f"\n{'═'*60}")
        print("  SELF-REFLECTION SUMMARY")
        for i, h in enumerate(history):
            tag   = "(initial)" if not h["reflected"] else f"(reflection {i})"
            arrow = "⚠" if h["prob"] >= self.threshold else "✓"
            print(f"  {arrow}  Round {i} {tag:20s}  prob={h['prob']:.3f}  →  {h['answer'][:60]}")
        improved = history[-1]["prob"] < history[0]["prob"]
        print(
            f"\n  {'↓ Improved' if improved else '→ No improvement'}: "
            f"{history[0]['prob']:.3f} → {history[-1]['prob']:.3f}"
        )
        print(f"{'═'*60}\n")
