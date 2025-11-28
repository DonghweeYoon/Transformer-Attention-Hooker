# This module provides a wrapper class for hooking into PyTorch Transformer models to extract attention weights.
#
# Original Inspiration:
# https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91
#
# Enhanced and modified to support:
# - Automatic `need_weights=True` forcing
# - Layer reuse (loops/weight sharing)
# - Cross-attention (rectangular matrices)

import re
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn


class TransformerAttentionHooker:
    """
    Hooks into PyTorch Transformer models to capture attention weights.

    This class registers forward hooks to extract attention maps from
    `nn.MultiheadAttention` modules.

    Features:
    - Forces `need_weights=True` for PyTorch native modules.
    - Supports capturing weights from layers called multiple times (e.g. weight sharing).
    - Validates output tensor shape to avoid false positives.
    """

    def __init__(self, model: nn.Module, layer_regex: Optional[str] = None):
        self.model = model
        self.layer_regex = layer_regex if layer_regex else r"self_attn$"
        self.hooks: List[Any] = []
        # Stores list of tensors per layer to handle multiple calls (e.g. in generation or shared weights)
        self.attentions: OrderedDict[str, List[torch.Tensor]] = OrderedDict()

    def setup(self) -> "TransformerAttentionHooker":
        """Registers hooks to the model layers matching the regex."""
        self.remove_hooks()
        self.attentions.clear()
        pattern = re.compile(self.layer_regex)

        for name, module in self.model.named_modules():
            if pattern.search(name):
                # 1. Pre-hook: Force calculation of attention weights
                # Only applies to standard PyTorch MultiheadAttention
                if isinstance(module, nn.MultiheadAttention):
                    self.hooks.append(
                        module.register_forward_pre_hook(
                            self._pre_hook_fn, with_kwargs=True
                        )
                    )

                # 2. Hook: Capture the output weights
                self.hooks.append(
                    module.register_forward_hook(partial(self._hook_fn, name))
                )
        return self

    def remove_hooks(self):
        """Removes all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @property
    def values(self) -> Dict[str, List[torch.Tensor]]:
        """
        Returns the captured attention weights.
        Dictionary values are Lists of Tensors to support multiple calls per layer.
        Shape of each tensor: (Batch, Heads, Seq_Target, Seq_Source)
        """
        return self.attentions

    def _pre_hook_fn(
        self, module: nn.Module, args: Tuple, kwargs: Dict[str, Any]
    ) -> Tuple[Tuple, Dict]:
        if isinstance(module, nn.MultiheadAttention):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
        return args, kwargs

    def _hook_fn(self, name: str, module: nn.Module, input: Any, output: Any):
        candidates = output if isinstance(output, (tuple, list)) else [output]

        for item in candidates:
            if self._is_attention_tensor(item):
                if name not in self.attentions:
                    self.attentions[name] = []
                # Detach and move to CPU immediately to save GPU memory
                self.attentions[name].append(item.detach().cpu())
                # Stop after finding the first valid attention tensor in the output tuple
                break

    def _is_attention_tensor(self, item: Any) -> bool:
        """
        Checks if the item is a valid attention weight tensor.
        Criteria:
        1. Is a Tensor
        2. Is 4D (Batch, Heads, Seq, Seq)
        3. Last two dimensions are equal (Square matrix for self-attention)
           or simply 4D if we want to support cross-attention (Seq_T != Seq_S).
           PyTorch MultiheadAttention returns (B, H, L, S).
        """
        return (
            isinstance(item, torch.Tensor)
            and item.dim() == 4
            # We remove the square matrix check to support Cross Attention
            # where Target Seq Len != Source Seq Len
        )

    def __del__(self):
        self.remove_hooks()
