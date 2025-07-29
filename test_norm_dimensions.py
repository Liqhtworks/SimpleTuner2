#!/usr/bin/env python3
"""Test norm layer dimensions before and after fusion"""

import torch
from diffusers import FluxTransformer2DModel
from helpers.training.diffusers_overrides import fuse_single_block_qkv_mlp

# Load model
model = FluxTransformer2DModel.from_pretrained(
    "/home/playerzer0x/ComfyUI/models/unet/FLUX.1-Kontext-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

# Check first single block norm dimensions before fusion
block = model.single_transformer_blocks[0]
print("Before fusion:")
print(f"  norm type: {type(block.norm)}")
print(f"  norm.linear weight shape: {block.norm.linear.weight.shape}")
print(f"  norm.linear bias shape: {block.norm.linear.bias.shape if block.norm.linear.bias is not None else 'None'}")

# Apply fusion to single block
fuse_single_block_qkv_mlp(block, permanent=True)

print("\nAfter fusion:")
print(f"  norm type: {type(block.norm)}")
print(f"  norm.linear weight shape: {block.norm.linear.weight.shape}")
print(f"  norm.linear bias shape: {block.norm.linear.bias.shape if block.norm.linear.bias is not None else 'None'}")

# Test with time embedding
print("\nTesting norm layer with time embedding:")
batch_size = 2
embedding_dim = model.time_text_embed.text_embedder.linear_1.in_features
print(f"  Expected time embedding dim: {embedding_dim}")

temb = torch.randn(batch_size, embedding_dim)
hidden_states = torch.randn(batch_size, 16, 3072)

try:
    # Test norm layer directly
    norm_out, gate = block.norm(hidden_states, emb=temb)
    print(f"  Norm forward pass successful!")
    print(f"  Norm output shape: {norm_out.shape}")
    print(f"  Gate shape: {gate.shape}")
except Exception as e:
    print(f"  Norm forward pass failed: {e}")
    
# Check if the norm.linear was somehow modified
print("\nChecking norm.linear internals:")
print(f"  norm.linear.in_features: {block.norm.linear.in_features}")
print(f"  norm.linear.out_features: {block.norm.linear.out_features}")