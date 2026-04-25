"""
judge.py — LLM-as-judge for labelling model responses.

GeminiJudge sends batches of (question, response, ground-truth answers) to
the Gemini API and returns "true"/"false" verdicts.
"""

import re
import time
from typing import List, Optional


class GeminiJudge:
    """
    Judges model responses for correctness using Google Gemini.

    Sends up to `batch_size` items per API call using a structured prompt
    that asks for a comma-separated "t"/"f" list.  Includes exponential
    backoff on rate-limit errors.

    Args:
        api_key:    Google Gemini API key.
        model_name: Gemini model to use for judging.
    """

    _BATCH_HEADER = (
        "For each item below, judge whether the model response correctly "
        "answers the question given the correct answers.\n"
        "Reply with ONLY a comma-separated list of 't' or 'f', one per item, "
        "in the same order. Nothing else.\n"
        "Example output for 3 items: t,f,t\n\n"
    )

    _ITEM_TEMPLATE = (
        "[{idx}] Question: {question}\n"
        "     Response: {response}\n"
        "     Correct Answers: {answers}\n"
    )

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-lite"):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model  = model_name

    # ------------------------------------------------------------------

    def judge_batch(
        self,
        items: List[dict],
        retries: int = 5,
    ) -> List[str]:
        """
        Judge a batch of items.

        Args:
            items:   List of dicts, each with keys:
                       "question"  — the question string
                       "response"  — the model response to judge
                       "answers"   — list/str of correct answers
            retries: Number of retry attempts on API failure.

        Returns:
            List of "true" / "false" / "error" in the same order as `items`.
        """
        from google.genai import types

        prompt = self._BATCH_HEADER
        for i, item in enumerate(items):
            prompt += self._ITEM_TEMPLATE.format(
                idx=i + 1,
                question=item["question"],
                response=item["response"],
                answers=item["answers"],
            )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        )

        for attempt in range(retries):
            try:
                raw = ""
                for chunk in self._client.models.generate_content_stream(
                    model=self._model,
                    contents=contents,
                    config=config,
                ):
                    if chunk.text:
                        raw += chunk.text

                verdicts = self._parse(raw.strip(), len(items))
                if verdicts is not None:
                    return verdicts

            except Exception as e:
                wait = self._retry_delay(e, attempt)
                print(f"  [judge] attempt {attempt + 1} failed: {e}")
                print(f"  [judge] retrying in {wait}s ...")
                time.sleep(wait)

        return ["error"] * len(items)

    # ------------------------------------------------------------------

    def _parse(self, raw: str, expected: int) -> Optional[List[str]]:
        """Parse 't,f,t' → ['true','false','true']. Returns None if malformed."""
        tokens = [t.strip().lower() for t in raw.split(",")]
        if len(tokens) != expected:
            return None
        result = []
        for t in tokens:
            if t.startswith('t'):
                result.append("true")
            elif t.startswith('f'):
                result.append("false")
            else:
                return None
        return result

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        match = re.search(r"retryDelay.*?(\d+)s", str(exc))
        return float(match.group(1)) if match else float(2 ** attempt)
