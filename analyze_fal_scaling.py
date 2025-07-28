#!/usr/bin/env python3
"""Analyze FAL-kontext LoRA to verify scaling factors."""

import safetensors.torch
import torch
import statistics

def analyze_fal_lora(lora_path):
    """Analyze a FAL-kontext LoRA file to understand weight magnitudes and scaling."""
    state_dict = safetensors.torch.load_file(lora_path)
    
    print(f"Analyzing FAL-kontext LoRA: {lora_path}")
    print(f"Total keys: {len(state_dict)}")
    
    # Group by layer type
    layer_groups = {
        'img_attn_qkv': [],
        'txt_attn_qkv': [],
        'img_mod_lin': [],
        'txt_mod_lin': [],
        'img_mlp': [],
        'txt_mlp': [],
        'modulation_lin': [],
        'linear1': [],
        'linear2': [],
        'final_layer': []
    }
    
    # Collect layers by type
    for key, tensor in state_dict.items():
        if 'img_attn_qkv' in key:
            layer_groups['img_attn_qkv'].append((key, tensor))
        elif 'txt_attn_qkv' in key:
            layer_groups['txt_attn_qkv'].append((key, tensor))
        elif 'img_mod_lin' in key:
            layer_groups['img_mod_lin'].append((key, tensor))
        elif 'txt_mod_lin' in key:
            layer_groups['txt_mod_lin'].append((key, tensor))
        elif 'img_mlp' in key:
            layer_groups['img_mlp'].append((key, tensor))
        elif 'txt_mlp' in key:
            layer_groups['txt_mlp'].append((key, tensor))
        elif 'modulation_lin' in key:
            layer_groups['modulation_lin'].append((key, tensor))
        elif 'linear1' in key:
            layer_groups['linear1'].append((key, tensor))
        elif 'linear2' in key:
            layer_groups['linear2'].append((key, tensor))
        elif 'final_layer' in key:
            layer_groups['final_layer'].append((key, tensor))
    
    print("\n" + "="*80)
    print("LAYER ANALYSIS BY TYPE")
    print("="*80)
    
    for layer_type, layers in layer_groups.items():
        if not layers:
            continue
            
        print(f"\n{layer_type.upper()}:")
        print("-" * 40)
        
        up_weights = []
        down_weights = []
        
        for key, tensor in layers:
            print(f"  {key}: {list(tensor.shape)}")
            
            if 'lora_up.weight' in key:
                up_weights.append(tensor)
                # Calculate weight statistics
                abs_mean = tensor.abs().mean().item()
                abs_std = tensor.abs().std().item()
                abs_max = tensor.abs().max().item()
                print(f"    Up weight stats: mean={abs_mean:.6f}, std={abs_std:.6f}, max={abs_max:.6f}")
                
                # Check dimensions for scaling hints
                if len(tensor.shape) >= 2:
                    out_dim, in_dim = tensor.shape[0], tensor.shape[1]
                    print(f"    Dimensions: {out_dim} -> {in_dim}")
                    
                    # Check if this suggests a scaling factor
                    if 'qkv' in key:
                        expected_base = out_dim / 3
                        print(f"    QKV scaling check: {out_dim} = 3 × {expected_base:.0f}")
                    elif 'mod_lin' in key and 'single' not in key:
                        expected_base = out_dim / 6
                        print(f"    Double mod scaling check: {out_dim} = 6 × {expected_base:.0f}")
                    elif 'modulation_lin' in key or ('mod_lin' in key and 'single' in key):
                        expected_base = out_dim / 3
                        print(f"    Single mod scaling check: {out_dim} = 3 × {expected_base:.0f}")
                        
            elif 'lora_down.weight' in key:
                down_weights.append(tensor)
                abs_mean = tensor.abs().mean().item()
                abs_std = tensor.abs().std().item()
                abs_max = tensor.abs().max().item()
                print(f"    Down weight stats: mean={abs_mean:.6f}, std={abs_std:.6f}, max={abs_max:.6f}")
        
        # Calculate relative magnitudes between up and down weights
        if up_weights and down_weights:
            up_mags = [w.abs().mean().item() for w in up_weights]
            down_mags = [w.abs().mean().item() for w in down_weights]
            
            avg_up = statistics.mean(up_mags)
            avg_down = statistics.mean(down_mags)
            ratio = avg_up / avg_down if avg_down > 0 else 0
            
            print(f"  Average up magnitude: {avg_up:.6f}")
            print(f"  Average down magnitude: {avg_down:.6f}")
            print(f"  Up/Down ratio: {ratio:.2f}")

if __name__ == "__main__":
    analyze_fal_lora("/Users/gt/Downloads/labubu_fal.safetensors")