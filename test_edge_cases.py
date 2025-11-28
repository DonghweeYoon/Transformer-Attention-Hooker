import sys
import os

# Ensure src is in path for local execution without install
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
from transformer_attention_hooker import TransformerAttentionHooker


def test_all():
    print("=" * 60)
    print("Running TransformerAttentionHooker Tests")
    print("=" * 60)

    # Common setup
    d_model = 32
    nhead = 4
    batch_size = 2
    torch.manual_seed(42)

    # --- Test 1: Basic Self-Attention ---
    print("\n[1/3] 🧪 Testing Standard Self-Attention (PyTorch Native)")
    print(
        "      Goal: Capture square attention matrices (Seq x Seq) from TransformerEncoder."
    )

    seq_len = 10
    num_layers = 2

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, batch_first=True
    )
    model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    model.eval()  # Critical: Disable dropout for deterministic attention weights

    print(f"      Model: TransformerEncoder with {num_layers} layers.")
    print("      Action: Running forward pass...")

    hooker = TransformerAttentionHooker(model).setup()
    x = torch.randn(batch_size, seq_len, d_model)
    _ = model(x)

    assert (
        len(hooker.values) == num_layers
    ), f"Expected {num_layers} layers, got {len(hooker.values)}"

    for name, attn_list in hooker.values.items():
        assert len(attn_list) == 1
        attn_tensor = attn_list[0]
        print(f"      - Layer '{name}' shape: {tuple(attn_tensor.shape)}")
        assert attn_tensor.shape == (batch_size, nhead, seq_len, seq_len)

        # Check probability sum (Softmax)
        row_sums = attn_tensor.sum(dim=-1)

        # Debug output if assertion fails
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
            print(f"      (!) Assertion Failed! Row sums stats:")
            print(f"          Min: {row_sums.min().item():.6f}")
            print(f"          Max: {row_sums.max().item():.6f}")
            print(f"          Mean: {row_sums.mean().item():.6f}")

        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-4
        ), "Attention weights do not sum to 1"

    hooker.remove_hooks()
    print("      Passed")

    # --- Test 2: Cross-Attention ---
    print("\n[2/3] Testing Cross-Attention (Seq Length Mismatch)")
    print("      Goal: Capture rectangular matrices (Target_Seq x Source_Seq).")

    tgt_len = 5
    src_len = 8

    class CrossAttnModel(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d, h, batch_first=True)

        def forward(self, q, k, v):
            return self.self_attn(q, k, v)

    model = CrossAttnModel(d_model, nhead)
    model.eval()
    hooker = TransformerAttentionHooker(model).setup()

    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, d_model)
    value = torch.randn(batch_size, src_len, d_model)

    print(f"      Input: Query={tgt_len}, Key/Value={src_len}")
    _ = model(query, key, value)

    attns = list(hooker.values.values())[0]
    print(f"      - Captured Shape: {tuple(attns[0].shape)}")
    assert attns[0].shape == (
        batch_size,
        nhead,
        tgt_len,
        src_len,
    ), f"Shape mismatch: {attns[0].shape}"

    hooker.remove_hooks()
    print("      Passed")

    # --- Test 3: Loop / Layer Reuse ---
    print("\n[3/3] Testing Layer Reuse (Weight Sharing / Loops)")
    print(
        "      Goal: Capture all attention maps when a layer is called multiple times."
    )

    seq_len = 6
    loop_steps = 3

    class LoopedModel(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d, h, batch_first=True)
            self.norm = nn.LayerNorm(d)

        def forward(self, x, steps):
            for _ in range(steps):
                x_attn, _ = self.self_attn(x, x, x)
                x = self.norm(x + x_attn)
            return x

    model = LoopedModel(d_model, nhead)
    model.eval()
    hooker = TransformerAttentionHooker(model).setup()

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"      Action: Calling layer {loop_steps} times in a loop.")
    _ = model(x, steps=loop_steps)

    assert len(hooker.values) == 1
    attn_list = list(hooker.values.values())[0]
    print(f"      - Captured count: {len(attn_list)} (Expected: {loop_steps})")
    assert (
        len(attn_list) == loop_steps
    ), f"Expected {loop_steps} calls, got {len(attn_list)}"

    hooker.remove_hooks()
    print("      Passed")

    print("\n" + "=" * 60)
    print("All tests passed successfully! Ready for production.")
    print("=" * 60)


if __name__ == "__main__":
    test_all()
