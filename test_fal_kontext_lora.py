#!/usr/bin/env python3
"""Test script for verifying dsy-kontext LoRA loading and mapping."""

import torch
import safetensors.torch
from helpers.training.adapter import (
    DSY_KONTEXT_KEY_MAPPING, 
    DSY_KONTEXT_SCALING_FACTORS,
    get_dsy_kontext_mapped_key,
    get_dsy_kontext_scaling_factor
)

def test_dsy_kontext_lora(lora_path="/Users/gt/Downloads/labubu_fal.safetensors"):
    """Test the dsy-kontext LoRA loading and verify mappings."""
    
    print("Loading LoRA file...")
    state_dict = safetensors.torch.load_file(lora_path)
    
    # Check if it's detected as dsy-kontext
    is_dsy_kontext = any(
        "img_attn_qkv" in k or "txt_attn_qkv" in k or 
        "img_mlp" in k or "txt_mlp" in k or 
        "modulation_lin" in k or "mod_lin" in k 
        for k in state_dict.keys()
    )
    
    print(f"\nDetected as dsy-kontext LoRA: {is_dsy_kontext}")
    
    # Get unique layer names
    unique_layers = set()
    for k in state_dict.keys():
        base = k.replace(".lora_down.weight", "").replace(".lora_up.weight", "")
        unique_layers.add(base)
    
    print(f"\nTotal unique layers: {len(unique_layers)}")
    
    # Test mappings for a few example layers
    test_keys = [
        "lora_unet_double_blocks_0_img_attn_qkv",
        "lora_unet_double_blocks_0_txt_attn_qkv",
        "lora_unet_double_blocks_0_img_mod_lin",
        "lora_unet_double_blocks_0_txt_mod_lin",
        "lora_unet_double_blocks_0_img_mlp_0",
        "lora_unet_double_blocks_0_img_mlp_2",
        "lora_unet_single_blocks_0_linear1",
        "lora_unet_single_blocks_0_linear2",
        "lora_unet_single_blocks_0_modulation_lin",
        "lora_unet_final_layer_linear"
    ]
    
    print("\n" + "="*80)
    print("Testing key mappings:")
    print("="*80)
    
    for key in test_keys:
        if key + ".lora_up.weight" in state_dict or key + ".lora_down.weight" in state_dict:
            mapped_key = get_dsy_kontext_mapped_key(key)
            scaling_factor = get_dsy_kontext_scaling_factor(mapped_key)
            
            # Get the actual dimensions from the state dict
            up_weight = state_dict.get(key + ".lora_up.weight")
            down_weight = state_dict.get(key + ".lora_down.weight")
            
            print(f"\nOriginal key: {key}")
            print(f"Mapped to: {mapped_key}")
            print(f"Scaling factor: {scaling_factor}x")
            
            if up_weight is not None:
                print(f"Up weight shape: {list(up_weight.shape)}")
                if len(up_weight.shape) >= 2:
                    output_dim = up_weight.shape[0]
                    rank = up_weight.shape[1]
                    # Check if output dimension matches expected scaling
                    if "mod_lin" in key and "single_blocks" not in key:
                        expected_base = output_dim / 6
                        print(f"  Output dim {output_dim} = 6 × {expected_base:.0f} (modulation for double block)")
                    elif "mod_lin" in key and "single_blocks" in key:
                        expected_base = output_dim / 3
                        print(f"  Output dim {output_dim} = 3 × {expected_base:.0f} (modulation for single block)")
                    elif "qkv" in key or "linear1" in key:
                        expected_base = output_dim / 3
                        print(f"  Output dim {output_dim} = 3 × {expected_base:.0f} (QKV projection)")
            
            if down_weight is not None:
                print(f"Down weight shape: {list(down_weight.shape)}")
    
    print("\n" + "="*80)
    print("Testing scaling for different layer types:")
    print("="*80)
    
    # Count layers by type
    layer_counts = {
        "img_attn_qkv": 0,
        "txt_attn_qkv": 0,
        "img_mod_lin": 0,
        "txt_mod_lin": 0,
        "img_mlp": 0,
        "txt_mlp": 0,
        "linear1": 0,
        "linear2": 0,
        "modulation_lin": 0,
        "final_layer": 0
    }
    
    for layer in unique_layers:
        for layer_type in layer_counts:
            if layer_type in layer:
                layer_counts[layer_type] += 1
    
    print("\nLayer type counts:")
    for layer_type, count in layer_counts.items():
        if count > 0:
            print(f"  {layer_type}: {count}")
    
    # Test that all expected mappings exist
    print("\n" + "="*80)
    print("Verifying all mappings exist:")
    print("="*80)
    
    missing_mappings = []
    for layer in unique_layers:
        mapped = get_dsy_kontext_mapped_key(layer)
        if mapped == layer:  # No mapping found
            # Extract the actual layer name part
            parts = layer.split("_")
            if len(parts) > 3:
                layer_name = "_".join(parts[3:])
                if layer_name not in DSY_KONTEXT_KEY_MAPPING:
                    missing_mappings.append(layer_name)
    
    if missing_mappings:
        print(f"\nWarning: Found {len(set(missing_mappings))} unique layer types without mappings:")
        for missing in set(missing_mappings):
            print(f"  - {missing}")
    else:
        print("\n✓ All layer types have mappings!")
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"LoRA file: {lora_path}")
    print(f"Total parameters: {len(state_dict)}")
    print(f"Unique layers: {len(unique_layers)}")
    print(f"Is dsy-kontext style: {is_dsy_kontext}")
    
    return state_dict, unique_layers


if __name__ == "__main__":
    import sys
    
    lora_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/gt/Downloads/labubu_fal.safetensors"
    
    try:
        test_dsy_kontext_lora(lora_path)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc() 