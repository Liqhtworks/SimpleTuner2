#!/usr/bin/env python3
"""
Debug the actual module structure of Flux transformer blocks
to understand why ff.net.2 LoRA weights are missing.
"""

import torch
import torch.nn as nn
from diffusers.models.activations import FeedForward

def examine_feedforward():
    """Examine the FeedForward module structure."""
    print("=== FeedForward Module Structure ===")
    
    # Create a sample FeedForward module similar to Flux
    ff = FeedForward(
        dim=3072,
        dim_out=3072, 
        activation_fn="gelu-approximate"
    )
    
    print(f"FeedForward module: {ff}")
    print("\nSubmodules:")
    for name, module in ff.named_modules():
        if name:  # Skip the root module
            print(f"  {name}: {type(module).__name__} {getattr(module, 'weight', torch.empty(0)).shape if hasattr(module, 'weight') else 'no weight'}")
    
    print("\nNamed parameters:")
    for name, param in ff.named_parameters():
        print(f"  {name}: {param.shape}")
    
    return ff

def check_flux_double_block_structure():
    """Check the structure of FluxTransformerBlock."""
    print("\n=== FluxTransformerBlock Structure ===")
    
    try:
        from helpers.models.flux.transformer import FluxTransformerBlock
        
        block = FluxTransformerBlock(
            dim=3072,
            num_attention_heads=24,
            attention_head_dim=128
        )
        
        print(f"FluxTransformerBlock: {block}")
        print("\nFF module structure:")
        for name, module in block.ff.named_modules():
            if name:
                print(f"  ff.{name}: {type(module).__name__} {getattr(module, 'weight', torch.empty(0)).shape if hasattr(module, 'weight') else 'no weight'}")
        
        print("\nFF_context module structure:")
        for name, module in block.ff_context.named_modules():
            if name:
                print(f"  ff_context.{name}: {type(module).__name__} {getattr(module, 'weight', torch.empty(0)).shape if hasattr(module, 'weight') else 'no weight'}")
                
        print("\nAll double block parameters:")
        for name, param in block.named_parameters():
            if 'ff' in name:
                print(f"  {name}: {param.shape}")
                
    except ImportError as e:
        print(f"Could not import FluxTransformerBlock: {e}")
        return None
    
    return block

if __name__ == "__main__":
    ff = examine_feedforward()
    block = check_flux_double_block_structure()