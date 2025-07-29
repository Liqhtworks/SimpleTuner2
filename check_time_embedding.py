#!/usr/bin/env python3
"""Check time embedding dimensions"""

import torch
from diffusers import FluxTransformer2DModel

# Load model
model = FluxTransformer2DModel.from_pretrained(
    "/home/playerzer0x/ComfyUI/models/unet/FLUX.1-Kontext-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

print("Model dimensions:")
print(f"  inner_dim: {model.inner_dim}")
print(f"  pooled_projection_dim: {model.config.pooled_projection_dim}")

print("\nTime text embed structure:")
print(f"  Type: {type(model.time_text_embed)}")
print(f"  Text embedder linear_1 in: {model.time_text_embed.text_embedder.linear_1.in_features}")
print(f"  Text embedder linear_1 out: {model.time_text_embed.text_embedder.linear_1.out_features}")
print(f"  Text embedder linear_2 out: {model.time_text_embed.text_embedder.linear_2.out_features}")

print("\nChecking AdaLayerNormZeroSingle expected dimensions:")
# The norm expects embedding_dim as input, which should be inner_dim
print(f"  Norm should be initialized with embedding_dim={model.inner_dim}")
print(f"  Norm linear should have shape: [{3 * model.inner_dim}, {model.inner_dim}]")

# Check actual norm dimensions
block = model.single_transformer_blocks[0] 
print(f"\nActual norm.linear shape: {block.norm.linear.weight.shape}")
print(f"  This means norm was initialized with embedding_dim={block.norm.linear.in_features}")

# Test the forward pass
print("\nTesting forward pass:")
batch_size = 2
seq_len = 16
hidden_states = torch.randn(batch_size, seq_len, model.inner_dim, dtype=torch.bfloat16)
timestep = torch.tensor([500.0, 500.0], dtype=torch.float32)
pooled_projections = torch.randn(batch_size, model.config.pooled_projection_dim, dtype=torch.bfloat16)

# Get time embedding
temb = model.time_text_embed(timestep, pooled_projections)
print(f"  Time embedding shape: {temb.shape}")
print(f"  Time embedding expected shape: [batch_size, inner_dim] = [{batch_size}, {model.inner_dim}]")