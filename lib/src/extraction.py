"""
extraction.py — FFN activation hooks and CETT feature computation.

Key classes/functions:
  ActivationExtractor   — plants forward hooks on FFN layers
  get_neuron_activations — runs a forward pass and returns per-layer activations
  compute_cett           — reduces token-level activations to a flat feature vector
"""

from operator import attrgetter
from typing import Dict, List, Optional

import torch


class ActivationExtractor:
    """
    Attaches PyTorch forward hooks to every FFN down-projection layer in a
    HuggingFace CausalLM and stores the pre-activation tensors.

    The extractor is model-agnostic: pass `layers_path` and `ffn_path` to
    point it at the right modules for your architecture.

    Args:
        model:        A loaded HuggingFace AutoModelForCausalLM.
        layers_path:  Dotted attribute path from `model` to the list of
                      transformer layers.
                      e.g. "model.layers"  for Llama / Mistral / Gemma / Phi
                           "transformer.h" for GPT-2 / Falcon
        ffn_path:     Dotted attribute path from each layer to the FFN
                      sub-module to hook (the input to this module is captured).
                      e.g. "mlp.down_proj" for Llama family
                           "mlp.c_proj"    for GPT-2
    """

    def __init__(
        self,
        model,
        layers_path: str = "model.layers",
        ffn_path: str = "mlp.down_proj",
    ):
        self.model = model
        self.layers_path = layers_path
        self.ffn_path = ffn_path
        self.activations: Dict[int, torch.Tensor] = {}
        self._hooks: list = []

    # ------------------------------------------------------------------
    def _get_layers(self):
        return attrgetter(self.layers_path)(self.model)

    def _get_ffn_module(self, layer):
        return attrgetter(self.ffn_path)(layer)

    # ------------------------------------------------------------------
    def register_hooks(self) -> None:
        """
        Attach a hook to every FFN layer.

        The hook captures `input[0]` — the tensor going *into* the projection
        matrix — which is the intermediate (post-activation) hidden state and
        carries the H-Neuron signal.

        Calling this method also clears `self.activations` so each call starts
        fresh.
        """
        self.remove_hooks()
        self.activations = {}

        for layer_idx, layer in enumerate(self._get_layers()):
            def _make_hook(idx: int):
                def _hook(module, input, output):
                    # input[0]: [batch, seq_len, intermediate_size]
                    self.activations[idx] = input[0].detach().cpu()
                return _hook

            ffn_module = self._get_ffn_module(layer)
            handle = ffn_module.register_forward_hook(_make_hook(layer_idx))
            self._hooks.append(handle)

    def remove_hooks(self) -> None:
        """
        Remove all registered hook handles.
        Activations are intentionally *not* cleared here so callers can read
        them after unhooking. They are cleared at the start of
        `register_hooks()` instead.
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks = []


# ──────────────────────────────────────────────────────────────────────────────

def get_neuron_activations(
    extractor: ActivationExtractor,
    tokenizer,
    prompt: str,
    response: str,
    token_indices: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor]:
    """
    Run a single forward pass on (prompt + response) and return the
    FFN activations sliced to the response token positions.

    Args:
        extractor:     An ActivationExtractor with hooks registered or not
                       (hooks are re-registered inside this function).
        tokenizer:     HuggingFace tokenizer matching the model.
        prompt:        The user prompt string (including any chat template).
        response:      The model's response string.
        token_indices: Specific token positions (absolute, from the full
                       prompt+response sequence) to slice to.
                       None → all response tokens.

    Returns:
        Dict mapping layer index → Tensor [n_tokens, intermediate_size].
        Returns {} if the response tokenizes to zero tokens.
    """
    prompt_ids   = tokenizer.encode(prompt,   add_special_tokens=True)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    if not response_ids:
        return {}

    full_ids     = prompt_ids + response_ids
    input_tensor = torch.tensor([full_ids]).to(extractor.model.device)

    extractor.register_hooks()
    with torch.no_grad():
        extractor.model(input_tensor)
    extractor.remove_hooks()

    response_start = len(prompt_ids)
    if token_indices is None:
        token_indices = list(range(response_start, len(full_ids)))

    if not token_indices:
        return {}

    return {
        layer_idx: act[0, token_indices, :]   # [n_tokens, intermediate_size]
        for layer_idx, act in extractor.activations.items()
    }


def compute_cett(
    layer_activations: Dict[int, torch.Tensor],
    method: str = "mean",
) -> torch.Tensor:
    """
    Reduce token-level activations to a single feature vector.

    CETT (Cross-token Excitation Telemetry): for each layer, compute the
    mean (or max) absolute activation across all response tokens, then
    concatenate all layers into one flat vector.

    Args:
        layer_activations: {layer_idx: Tensor[n_tokens, intermediate_size]}
        method:            "mean" (default) or "max"

    Returns:
        Tensor of shape [num_layers * intermediate_size].
        This is the feature vector passed to the hallucination probe.
    """
    vectors = []
    for layer_idx in sorted(layer_activations.keys()):
        act = layer_activations[layer_idx]   # [n_tokens, intermediate_size]
        if method == "mean":
            score = act.abs().mean(dim=0)
        elif method == "max":
            score = act.abs().max(dim=0).values
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'mean' or 'max'.")
        vectors.append(score)

    return torch.cat(vectors, dim=0)   # [num_layers * intermediate_size]
