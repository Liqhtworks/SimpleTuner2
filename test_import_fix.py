#!/usr/bin/env python3
"""Quick test to verify import fix."""

from helpers.models.flux.transformer import FluxSingleTransformerBlock, FluxTransformerBlock
from helpers.training.diffusers_overrides import fuse_single_block_qkv_mlp

# Create a single block
block = FluxSingleTransformerBlock(
    dim=3072,
    num_attention_heads=24,
    attention_head_dim=128,
    mlp_ratio=4.0
)

print(f"Block type: {type(block)}")
print(f"Has linear1 before fusion: {hasattr(block, 'linear1')}")

# Try fusion
fuse_single_block_qkv_mlp(block)

print(f"Has linear1 after fusion: {hasattr(block, 'linear1')}")
if hasattr(block, 'linear1'):
    print(f"Linear1 shape: {block.linear1.weight.shape}")
    print(f"Expected: torch.Size([21504, 3072])") 