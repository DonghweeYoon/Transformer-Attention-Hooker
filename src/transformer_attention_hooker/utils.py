import os
from typing import List, Optional
import torch
import matplotlib.pyplot as plt


def plot_attention_grid(
    attention: torch.Tensor,
    tokens: Optional[List[str]] = None,
    layer_name: str = "Layer",
    batch_idx: int = 0,
    save_path: Optional[str] = None,
    cmap: str = "viridis",
):
    """
    Plots all attention heads of a specific layer in a grid.

    Args:
        attention: Tensor of shape (Batch, Heads, Seq_Len, Seq_Len)
        tokens: List of token strings. If None, uses indices.
        layer_name: Title for the plot
        batch_idx: Index of the batch to visualize (default: 0)
        save_path: Path to save the image file.
        cmap: Colormap to use (default: 'viridis')
    """
    if attention.dim() != 4:
        raise ValueError(
            f"Expected 4D tensor (Batch, Heads, Seq, Seq), got {attention.shape}"
        )

    # Select batch and move to CPU
    attn_heads = attention[batch_idx].detach().cpu()  # (Heads, Seq, Seq)
    num_heads = attn_heads.shape[0]

    # Calculate grid dimensions
    cols = int(num_heads**0.5)
    rows = (num_heads + cols - 1) // cols

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), dpi=100)
    if num_heads == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        # Plot heatmap
        # vmin=0, vmax=1 ensures consistent color scale for probabilities
        im = ax.imshow(attn_heads[h], cmap=cmap, vmin=0, vmax=1, aspect="auto")

        ax.set_title(f"Head {h}", fontsize=10)

        # Handle Ticks
        if tokens:
            # Limit ticks to avoid clutter if sequence is too long
            step = 1 if len(tokens) < 25 else len(tokens) // 20
            indices = list(range(0, len(tokens), step))
            labels = [tokens[i] for i in indices]

            ax.set_xticks(indices)
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_yticks(indices)
            ax.set_yticklabels(labels, fontsize=8)
        else:
            ax.tick_params(axis="both", which="both", length=0)
            # Show indices if no tokens provided, but sparse them
            seq_len = attn_heads.shape[-1]
            step = max(1, seq_len // 5)
            ax.set_xticks(range(0, seq_len, step))
            ax.set_yticks(range(0, seq_len, step))

    # Hide unused subplots
    for h in range(num_heads, len(axes)):
        axes[h].axis("off")

    plt.suptitle(f"Attention Maps: {layer_name}", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved attention plot to: {save_path}")

    # Close plot to free memory
    plt.close(fig)
