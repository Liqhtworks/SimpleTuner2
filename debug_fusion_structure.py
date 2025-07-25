#!/usr/bin/env python3
"""Debug script to understand attention module structure after fusion."""

import torch
from diffusers import FluxTransformer2DModel
from diffusers.models.attention import Attention

def inspect_attention_modules(model):
    """Inspect the structure of attention modules in the model."""
    
    print("=== SINGLE TRANSFORMER BLOCKS ===")
    if hasattr(model, 'single_transformer_blocks') and len(model.single_transformer_blocks) > 0:
        block = model.single_transformer_blocks[0]
        attn = block.attn
        
        print(f"Single block type: {type(block)}")
        print(f"Attention type: {type(attn)}")
        print(f"Attention attributes: {[attr for attr in dir(attn) if not attr.startswith('_') and 'weight' in str(getattr(attn, attr, ''))]}")
        
        # Check for QKV layers
        print(f"Has to_q: {hasattr(attn, 'to_q')}")
        print(f"Has to_k: {hasattr(attn, 'to_k')}")
        print(f"Has to_v: {hasattr(attn, 'to_v')}")
        print(f"Has to_qkv: {hasattr(attn, 'to_qkv')}")
        
        # Check dimensions
        if hasattr(attn, 'to_qkv'):
            print(f"to_qkv shape: {attn.to_qkv.weight.shape}")
    
    print("\n=== DOUBLE TRANSFORMER BLOCKS ===")
    if hasattr(model, 'transformer_blocks') and len(model.transformer_blocks) > 0:
        block = model.transformer_blocks[0]
        attn = block.attn
        
        print(f"Double block type: {type(block)}")
        print(f"Attention type: {type(attn)}")
        
        # Check for image QKV
        print(f"Has to_q: {hasattr(attn, 'to_q')}")
        print(f"Has to_qkv: {hasattr(attn, 'to_qkv')}")
        
        # Check for text QKV  
        print(f"Has add_q_proj: {hasattr(attn, 'add_q_proj')}")
        print(f"Has to_added_qkv: {hasattr(attn, 'to_added_qkv')}")
        print(f"Has add_qkv_proj: {hasattr(attn, 'add_qkv_proj')}")

if __name__ == "__main__":
    # Create model
    model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="transformer",
        torch_dtype=torch.float16,
    )
    
    print("Before any fusion:")
    inspect_attention_modules(model)
    
    # Apply standard fusion
    for module in model.modules():
        if isinstance(module, Attention):
            module.fuse_projections(fuse=True)
    
    print("\n\nAfter standard fusion:")
    inspect_attention_modules(model) 