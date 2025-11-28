# Transformer-Attention-Hooker

A lightweight, robust utility for extracting and visualizing attention weights from PyTorch Transformer models.

This tool simplifies the process of debugging and analyzing Transformer internals by automatically hooking into `nn.MultiheadAttention` modules, handling the `need_weights=True` flag, and managing multiple forward passes (e.g., in generation loops or shared layers).

## Features

- **Automatic Hooking**: Automatically detects `nn.MultiheadAttention` layers using regex.
- **Force Weights**: Automatically sets `need_weights=True` during the forward pass so you don't have to modify your model code.
- **Layer Reuse Support**: Correctly captures attention weights even if a layer is called multiple times (e.g., in a loop or with shared weights).
- **Cross-Attention Support**: Works with both square self-attention and rectangular cross-attention matrices.
- **Visualization Tools**: Includes a built-in visualizer to plot attention heads as heatmaps.

## Requirements

- Python 3.6+
- PyTorch
- Matplotlib

```bash
pip install torch matplotlib
```

## Quick Start

### 1. Extracting Attention Weights

Wrap your model with `TransformerAttentionHooker` before running the forward pass.

```python
import torch
import torch.nn as nn
from src import TransformerAttentionHooker

# 1. Define or load your model
model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True),
    num_layers=2
)

# 2. Setup the hooker
# By default, it hooks layers ending with 'self_attn'
hooker = TransformerAttentionHooker(model, layer_regex=r"self_attn$").setup()

# 3. Run a forward pass
x = torch.randn(1, 10, 32) # (Batch, Seq, Feature)
output = model(x)

# 4. Access the captured attention weights
# hooker.values is a dict: {layer_name: [tensor_call_1, tensor_call_2, ...]}
print("Captured layers:", list(hooker.values.keys()))

for name, attn_list in hooker.values.items():
    print(f"Layer: {name}")
    # Get the tensor from the first call
    attn_tensor = attn_list[0]
    print(f"  Shape: {attn_tensor.shape}") # (Batch, Heads, Seq, Seq)

# 5. Cleanup
hooker.remove_hooks()
```

### 2. Visualizing Attention

Use the included `plot_attention_grid` function to generate heatmaps for all heads in a layer.

```python
from src import plot_attention_grid

# Assuming 'attn_tensor' is captured from the example above
layer_name = "layers.0.self_attn"
attn_tensor = hooker.values[layer_name][0]

plot_attention_grid(
    attn_tensor,
    tokens=[f"Token_{i}" for i in range(10)], # Optional: Add labels
    layer_name=layer_name,
    save_path=f"plots/{layer_name}.png"
)
```

## Advanced Usage

### Custom Layer Selection
If your model names its attention layers differently (e.g., `attn1`, `cross_attention`), you can pass a custom regex pattern.

```python
# Hook all layers containing "attn"
hooker = TransformerAttentionHooker(model, layer_regex=r".*attn.*").setup()
```

### Handling Loops (Generation / Shared Layers)
If a layer is used multiple times during a forward pass (common in recurrent-style generation or weight sharing), `hooker.values[layer_name]` will contain a list of tensors, one for each call.

```python
# Example: A layer called 3 times
output = model(x)

attn_calls = hooker.values['my_layer']
print(len(attn_calls)) # 3
print(attn_calls[0].shape) # Attention from 1st pass
print(attn_calls[1].shape) # Attention from 2nd pass
```

## Project Structure

- `src/attention_hooker.py`: Core hooking logic.
- `src/visualizer.py`: Matplotlib plotting utilities.
- `demo_viz.py`: Runnable demo script.
- `test_edge_cases.py`: Tests ensuring robustness for loops and cross-attention.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Donghwee Yoon
