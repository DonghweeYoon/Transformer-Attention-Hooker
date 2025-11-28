import torch
import torch.nn as nn
from transformer_attention_hooker import TransformerAttentionHooker


class CustomAttention(nn.Module):
    """Simulates a HuggingFace-style attention layer."""

    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.output_attentions = False  # Simulation of HF config

    def forward(self, x):
        B, L, E = x.shape
        # Fake attention computation
        attn_weights = torch.randn(B, self.nhead, L, L)
        # Make them sum to 1
        attn_weights = torch.softmax(attn_weights, dim=-1)

        output = x

        if self.output_attentions:
            return output, attn_weights
        else:
            return (output,)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = CustomAttention(32, 4)

    def forward(self, x):
        return self.layer(x)


def test_custom_integration():
    print("Testing Custom/HF-style Integration...")
    model = MockModel()

    # Case 1: Default behavior (output_attentions=False)
    # The hooker cannot force this for custom layers.
    print("1. Testing with output_attentions=False (Default)")
    hooker = TransformerAttentionHooker(model, layer_regex="layer").setup()

    x = torch.randn(1, 10, 32)
    _ = model(x)

    if "layer" in hooker.values:
        print("   [Unexpected] Captured attention even though model didn't output it?")
    else:
        print(
            "   [Expected] Did not capture attention (hooker can't force custom layers)."
        )

    hooker.remove_hooks()

    # Case 2: User manually enables output_attentions
    print("\n2. Testing with output_attentions=True (Manual Config)")
    model.layer.output_attentions = True
    hooker = TransformerAttentionHooker(model, layer_regex="layer").setup()

    _ = model(x)

    if "layer" in hooker.values:
        print(f"   [Success] Captured attention: {hooker.values['layer'][0].shape}")
    else:
        print("   [Failure] Failed to capture attention even when output.")


if __name__ == "__main__":
    test_custom_integration()
