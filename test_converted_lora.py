#!/usr/bin/env python3
"""
Test script to validate that the converted LoRA can be loaded by SimpleTuner.
"""

import torch
import safetensors.torch
import sys
from pathlib import Path

def test_converted_lora(lora_path):
    """Test that the converted LoRA has the correct structure for SimpleTuner."""
    
    print(f"Testing converted LoRA: {lora_path}")
    
    # Load the converted weights
    state_dict = safetensors.torch.load_file(lora_path)
    
    # Check for expected SimpleTuner patterns
    required_patterns = [
        "transformer.transformer_blocks",
        "transformer.single_transformer_blocks", 
        "lora_A.weight",
        "lora_B.weight",
        ".alpha"
    ]
    
    found_patterns = {pattern: False for pattern in required_patterns}
    
    for key in state_dict.keys():
        for pattern in required_patterns:
            if pattern in key:
                found_patterns[pattern] = True
    
    # Report results
    print("\nStructure Check:")
    all_good = True
    for pattern, found in found_patterns.items():
        status = "✓" if found else "✗"
        print(f"  {status} {pattern}")
        if not found:
            all_good = False
    
    # Check for fused QKV layers (should have to_qkv and add_qkv_proj)
    has_fused_qkv = False
    has_fused_add_qkv = False
    
    for key in state_dict.keys():
        if "to_qkv" in key:
            has_fused_qkv = True
        if "add_qkv_proj" in key:
            has_fused_add_qkv = True
    
    print("\nFused QKV Check (for dsy-kontext compatibility):")
    print(f"  {'✓' if has_fused_qkv else '✗'} to_qkv layers present")
    print(f"  {'✓' if has_fused_add_qkv else '✗'} add_qkv_proj layers present")
    
    # Check FF layer dimensions (they should have been scaled)
    print("\nFF Layer Scaling Check:")
    ff_layers_found = 0
    for key, tensor in state_dict.items():
        if "lora_B.weight" in key and ("ff.net." in key or "ff_context.net." in key):
            ff_layers_found += 1
            if ff_layers_found <= 3:  # Show first 3 as examples
                print(f"  {key}: shape={tensor.shape}")
    print(f"  Total FF B layers found: {ff_layers_found}")
    
    # Check alpha parameters
    alpha_count = sum(1 for k in state_dict.keys() if ".alpha" in k)
    lora_a_count = sum(1 for k in state_dict.keys() if "lora_A.weight" in k)
    print(f"\nAlpha Parameters:")
    print(f"  LoRA A layers: {lora_a_count}")
    print(f"  Alpha parameters: {alpha_count}")
    print(f"  {'✓' if alpha_count == lora_a_count else '✗'} Alpha count matches LoRA A count")
    
    # Detect rank
    rank = None
    for key, tensor in state_dict.items():
        if "lora_A.weight" in key:
            rank = tensor.shape[0]
            break
    print(f"\nLoRA Rank: {rank}")
    
    # Summary
    print("\n" + "="*50)
    if all_good and has_fused_qkv and has_fused_add_qkv and alpha_count == lora_a_count:
        print("✓ Conversion successful! This LoRA is compatible with SimpleTuner dsy-kontext.")
    else:
        print("✗ Issues detected. Please review the conversion.")
    
    return all_good


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_converted_lora.py <path_to_converted_lora>")
        sys.exit(1)
    
    lora_path = sys.argv[1]
    if not Path(lora_path).exists():
        print(f"Error: File not found: {lora_path}")
        sys.exit(1)
    
    success = test_converted_lora(lora_path)
    sys.exit(0 if success else 1)