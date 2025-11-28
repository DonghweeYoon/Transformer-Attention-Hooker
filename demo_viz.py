import sys
import os

# Ensure src is in path for local execution without install
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
from transformer_attention_hooker import TransformerAttentionHooker
from transformer_attention_hooker import plot_attention_grid


def run_demo():
    print("Running Visualization Demo...")

    # 1. Define a simple model
    d_model = 32
    nhead = 4
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
        num_layers=2,
    )
    model.eval()

    # 2. Setup Hooker
    hooker = TransformerAttentionHooker(model).setup()

    # 3. Forward Pass
    seq_len = 8
    x = torch.randn(1, seq_len, d_model)
    _ = model(x)

    # 4. Visualize
    tokens = [f"Token_{i}" for i in range(seq_len)]

    for name, attn_list in hooker.values.items():
        # Only visualize the first call if called multiple times
        attn = attn_list[0]

        # Save to 'plots' directory
        save_path = f"plots/{name}.png"
        print(f"Plotting {name}...")
        plot_attention_grid(attn, tokens=tokens, layer_name=name, save_path=save_path)

    hooker.remove_hooks()
    print("Done! Check the 'plots' directory.")


if __name__ == "__main__":
    run_demo()
