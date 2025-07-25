#!/usr/bin/env python3
"""Test script for verifying FAL kontext fusion implementation."""

import torch
from diffusers import FluxTransformer2DModel
from helpers.training.diffusers_overrides import (
    fuse_all_blocks_fal_kontext,
    apply_fal_kontext_forward_overrides,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_fal_kontext_fusion():
    """Test the FAL kontext fusion implementation."""
    
    print("Loading Flux transformer model...")
    model = FluxTransformer2DModel(
        patch_size=1,
        in_channels=64,
        num_layers=2,  # Small model for testing
        num_single_layers=3,  # Small model for testing
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        guidance_embeds=False,
    )
    
    print("\nBefore fusion - Single block structure:")
    single_block = model.single_transformer_blocks[0]
    print(f"  - Has linear1: {hasattr(single_block, 'linear1')}")
    print(f"  - Has attn.to_q: {hasattr(single_block.attn, 'to_q')}")
    print(f"  - Has attn.to_qkv: {hasattr(single_block.attn, 'to_qkv')}")
    print(f"  - Has proj_mlp: {hasattr(single_block, 'proj_mlp')}")
    
    print("\nBefore fusion - Double block structure:")
    double_block = model.transformer_blocks[0]
    print(f"  - Has img_attn_qkv: {hasattr(double_block, 'img_attn_qkv')}")
    print(f"  - Has txt_attn_qkv: {hasattr(double_block, 'txt_attn_qkv')}")
    print(f"  - Has attn.to_q: {hasattr(double_block.attn, 'to_q')}")
    
    print("\nApplying FAL kontext fusion...")
    fusion_count = fuse_all_blocks_fal_kontext(model, permanent=True)
    print(f"Fusion count: {fusion_count}")
    
    print("\nAfter fusion - Single block structure:")
    single_block = model.single_transformer_blocks[0]
    print(f"  - Has linear1: {hasattr(single_block, 'linear1')}")
    if hasattr(single_block, 'linear1'):
        print(f"    - Shape: {single_block.linear1.weight.shape}")
        print(f"    - Expected: [21504, 3072] (7x output)")
    print(f"  - Has attn.to_q: {hasattr(single_block.attn, 'to_q')}")
    print(f"  - Has attn.to_qkv: {hasattr(single_block.attn, 'to_qkv')}")
    print(f"  - Has proj_mlp: {hasattr(single_block, 'proj_mlp')}")
    print(f"  - Fused flag: {getattr(single_block, 'fused_qkv_mlp', False)}")
    
    print("\nAfter fusion - Double block structure:")
    double_block = model.transformer_blocks[0]
    print(f"  - Has img_attn_qkv: {hasattr(double_block, 'img_attn_qkv')}")
    if hasattr(double_block, 'img_attn_qkv'):
        print(f"    - Shape: {double_block.img_attn_qkv.weight.shape}")
    print(f"  - Has txt_attn_qkv: {hasattr(double_block, 'txt_attn_qkv')}")
    if hasattr(double_block, 'txt_attn_qkv'):
        print(f"    - Shape: {double_block.txt_attn_qkv.weight.shape}")
    print(f"  - Has img_mod_lin: {hasattr(double_block, 'img_mod_lin')}")
    if hasattr(double_block, 'img_mod_lin'):
        print(f"    - Shape: {double_block.img_mod_lin.weight.shape}")
    print(f"  - Has txt_mod_lin: {hasattr(double_block, 'txt_mod_lin')}")
    print(f"  - Has img_mlp_0: {hasattr(double_block, 'img_mlp_0')}")
    print(f"  - Has txt_mlp_0: {hasattr(double_block, 'txt_mlp_0')}")
    print(f"  - Fused flag: {getattr(double_block, 'fal_kontext_fused', False)}")
    
    print("\nApplying forward overrides...")
    apply_fal_kontext_forward_overrides(model)
    
    print("\nTesting forward pass with fused single block...")
    try:
        # Create dummy inputs
        batch_size = 2
        seq_len = 16
        hidden_dim = 3072
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        temb = torch.randn(batch_size, 256)  # Time embedding dimension
        
        # Test single block forward
        output = single_block(hidden_states, temb)
        print(f"  - Forward pass successful! Output shape: {output.shape}")
        print(f"  - Expected shape: [{batch_size}, {seq_len}, {hidden_dim}]")
        
    except Exception as e:
        print(f"  - Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Single blocks fused: {fusion_count['single']}")
    print(f"Double blocks fused: {fusion_count['double']}")
    
    # List all unique layer names that would be targeted by LoRA
    print("\nLayers available for LoRA targeting:")
    seen_layers = set()
    
    # Check single blocks
    for i, block in enumerate(model.single_transformer_blocks[:2]):  # First 2 blocks
        prefix = f"single_transformer_blocks.{i}"
        if hasattr(block, 'linear1'):
            seen_layers.add(f"{prefix}.linear1")
        if hasattr(block.attn, 'to_out') and hasattr(block.attn.to_out, '__getitem__'):
            seen_layers.add(f"{prefix}.attn.to_out.0")
        if hasattr(block, 'norm') and hasattr(block.norm, 'linear'):
            seen_layers.add(f"{prefix}.norm.linear")
    
    # Check double blocks  
    for i, block in enumerate(model.transformer_blocks[:2]):  # First 2 blocks
        prefix = f"transformer_blocks.{i}"
        for attr in ['img_attn_qkv', 'txt_attn_qkv', 'img_attn_proj', 'txt_attn_proj',
                     'img_mod_lin', 'txt_mod_lin', 'img_mlp_0', 'img_mlp_2',
                     'txt_mlp_0', 'txt_mlp_2']:
            if hasattr(block, attr):
                seen_layers.add(f"{prefix}.{attr}")
    
    print("\nFound layers:")
    for layer in sorted(seen_layers):
        print(f"  - {layer}")
    
    return model


if __name__ == "__main__":
    test_fal_kontext_fusion() 