#!/usr/bin/env python3
"""
Compare LoRA architectures between two safetensors files.
This helps identify differences in layer naming, structure, and training approaches.
"""

import argparse
import safetensors.torch
import torch
from pathlib import Path
from collections import defaultdict
import json


def extract_lora_info(state_dict):
    """Extract LoRA architecture information from state dict."""
    lora_info = {
        'layers': defaultdict(dict),
        'naming_style': 'unknown',
        'total_params': 0,
        'rank_info': {},
        'dtype_info': {},
    }
    
    # Detect naming style
    sample_keys = list(state_dict.keys())[:10]
    if any('lora_unet' in k for k in sample_keys):
        lora_info['naming_style'] = 'fal-kontext/kohya'
    elif any('transformer.transformer_blocks' in k for k in sample_keys):
        lora_info['naming_style'] = 'diffusers'
    elif any('transformer.lora_unet' in k for k in sample_keys):
        lora_info['naming_style'] = 'diffusers-converted'
    
    # Process each tensor
    for key, tensor in state_dict.items():
        # Skip non-LoRA keys
        if not any(suffix in key for suffix in ['.lora_A.weight', '.lora_B.weight', 
                                                 '.lora_down.weight', '.lora_up.weight',
                                                 '.alpha', '.lora_alpha']):
            continue
            
        # Extract base module name
        if '.lora_A.weight' in key:
            base_key = key.replace('.lora_A.weight', '')
            weight_type = 'A'
            lora_info['layers'][base_key]['A_shape'] = list(tensor.shape)
            lora_info['layers'][base_key]['rank'] = tensor.shape[0]
        elif '.lora_B.weight' in key:
            base_key = key.replace('.lora_B.weight', '')
            weight_type = 'B'
            lora_info['layers'][base_key]['B_shape'] = list(tensor.shape)
        elif '.lora_down.weight' in key:
            base_key = key.replace('.lora_down.weight', '')
            weight_type = 'down'
            lora_info['layers'][base_key]['down_shape'] = list(tensor.shape)
            lora_info['layers'][base_key]['rank'] = tensor.shape[0]
        elif '.lora_up.weight' in key:
            base_key = key.replace('.lora_up.weight', '')
            weight_type = 'up'
            lora_info['layers'][base_key]['up_shape'] = list(tensor.shape)
        elif '.alpha' in key or '.lora_alpha' in key:
            base_key = key.replace('.lora_alpha', '').replace('.alpha', '')
            lora_info['layers'][base_key]['alpha'] = tensor.item() if tensor.numel() == 1 else tensor.tolist()
            continue
        else:
            continue
            
        # Track dtype
        lora_info['dtype_info'][tensor.dtype] = lora_info['dtype_info'].get(tensor.dtype, 0) + 1
        
        # Track total parameters
        lora_info['total_params'] += tensor.numel()
        
    # Analyze rank distribution
    ranks = {}
    for layer, info in lora_info['layers'].items():
        if 'rank' in info:
            rank = info['rank']
            ranks[rank] = ranks.get(rank, 0) + 1
    lora_info['rank_info'] = ranks
    
    return lora_info


def categorize_layers(lora_info):
    """Categorize layers by their type and location."""
    categories = {
        'double_blocks': defaultdict(list),
        'single_blocks': defaultdict(list),
        'global': [],
    }
    
    for layer_name in lora_info['layers']:
        # Clean up layer name
        clean_name = layer_name.replace('transformer.', '').replace('lora_unet_', '')
        
        if 'double_blocks' in clean_name or 'transformer_blocks' in clean_name:
            # Extract block number and module type
            if 'double_blocks' in clean_name:
                parts = clean_name.split('_')
                block_idx = parts[2] if len(parts) > 2 else '0'
                module_type = '_'.join(parts[3:]) if len(parts) > 3 else 'unknown'
            else:
                parts = clean_name.split('.')
                block_idx = parts[1] if len(parts) > 1 else '0'
                module_type = '.'.join(parts[2:]) if len(parts) > 2 else 'unknown'
            
            categories['double_blocks'][module_type].append(int(block_idx))
            
        elif 'single_blocks' in clean_name or 'single_transformer_blocks' in clean_name:
            # Extract block number and module type
            if 'single_blocks' in clean_name:
                parts = clean_name.split('_')
                block_idx = parts[2] if len(parts) > 2 else '0'
                module_type = '_'.join(parts[3:]) if len(parts) > 3 else 'unknown'
            else:
                parts = clean_name.split('.')
                block_idx = parts[1] if len(parts) > 1 else '0'
                module_type = '.'.join(parts[2:]) if len(parts) > 2 else 'unknown'
                
            categories['single_blocks'][module_type].append(int(block_idx))
            
        else:
            categories['global'].append(clean_name)
    
    # Sort block indices
    for block_type in ['double_blocks', 'single_blocks']:
        for module_type in categories[block_type]:
            categories[block_type][module_type].sort()
    
    return categories


def compare_architectures(lora1_info, lora2_info):
    """Compare two LoRA architectures and identify differences."""
    comparison = {
        'naming_style_diff': lora1_info['naming_style'] != lora2_info['naming_style'],
        'total_params_diff': abs(lora1_info['total_params'] - lora2_info['total_params']),
        'missing_in_lora1': [],
        'missing_in_lora2': [],
        'rank_differences': [],
        'shape_differences': [],
    }
    
    # Get categorized layers
    cat1 = categorize_layers(lora1_info)
    cat2 = categorize_layers(lora2_info)
    
    # Compare module types in each category
    for category in ['double_blocks', 'single_blocks']:
        modules1 = set(cat1[category].keys())
        modules2 = set(cat2[category].keys())
        
        # Find missing modules
        for module in modules2 - modules1:
            comparison['missing_in_lora1'].append(f"{category}: {module}")
        for module in modules1 - modules2:
            comparison['missing_in_lora2'].append(f"{category}: {module}")
    
    # Compare global layers
    global1 = set(cat1['global'])
    global2 = set(cat2['global'])
    for layer in global2 - global1:
        comparison['missing_in_lora1'].append(f"global: {layer}")
    for layer in global1 - global2:
        comparison['missing_in_lora2'].append(f"global: {layer}")
    
    return comparison, cat1, cat2


def print_report(file1, file2, lora1_info, lora2_info, comparison, cat1, cat2):
    """Print a detailed comparison report."""
    print("=" * 80)
    print("LoRA ARCHITECTURE COMPARISON REPORT")
    print("=" * 80)
    
    print(f"\nFile 1: {file1}")
    print(f"  Naming style: {lora1_info['naming_style']}")
    print(f"  Total parameters: {lora1_info['total_params']:,}")
    print(f"  Unique layers: {len(lora1_info['layers'])}")
    print(f"  Rank distribution: {dict(lora1_info['rank_info'])}")
    print(f"  Data types: {dict(lora1_info['dtype_info'])}")
    
    print(f"\nFile 2: {file2}")
    print(f"  Naming style: {lora2_info['naming_style']}")
    print(f"  Total parameters: {lora2_info['total_params']:,}")
    print(f"  Unique layers: {len(lora2_info['layers'])}")
    print(f"  Rank distribution: {dict(lora2_info['rank_info'])}")
    print(f"  Data types: {dict(lora2_info['dtype_info'])}")
    
    print("\n" + "-" * 80)
    print("ARCHITECTURAL DIFFERENCES")
    print("-" * 80)
    
    if comparison['missing_in_lora1']:
        print(f"\nLayers present in {Path(file2).name} but MISSING in {Path(file1).name}:")
        for layer in sorted(comparison['missing_in_lora1']):
            print(f"  - {layer}")
    
    if comparison['missing_in_lora2']:
        print(f"\nLayers present in {Path(file1).name} but MISSING in {Path(file2).name}:")
        for layer in sorted(comparison['missing_in_lora2']):
            print(f"  - {layer}")
    
    print("\n" + "-" * 80)
    print("LAYER BREAKDOWN BY CATEGORY")
    print("-" * 80)
    
    for idx, (cat, name) in enumerate([(cat1, Path(file1).name), (cat2, Path(file2).name)]):
        print(f"\n{name}:")
        
        print("\n  Double Blocks (MMDiT):")
        for module_type, blocks in sorted(cat['double_blocks'].items()):
            print(f"    {module_type}: {len(blocks)} blocks (indices: {min(blocks)}-{max(blocks)})")
        
        print("\n  Single Blocks (DiT):")
        for module_type, blocks in sorted(cat['single_blocks'].items()):
            print(f"    {module_type}: {len(blocks)} blocks (indices: {min(blocks)}-{max(blocks)})")
        
        if cat['global']:
            print("\n  Global Layers:")
            for layer in sorted(cat['global']):
                print(f"    {layer}")
    
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)
    
    if comparison['missing_in_lora1']:
        print(f"\nTo match {Path(file2).name} architecture in SimpleTuner:")
        print("Add these modules to your --flux_lora_target configuration:")
        
        # Group recommendations by type
        missing_modules = defaultdict(set)
        for item in comparison['missing_in_lora1']:
            if 'double_blocks:' in item:
                module = item.split(': ')[1]
                missing_modules['double'].add(module)
            elif 'single_blocks:' in item:
                module = item.split(': ')[1]
                missing_modules['single'].add(module)
        
        if missing_modules['double']:
            print("\n  For double blocks, add:")
            for module in sorted(missing_modules['double']):
                if 'img_attn_proj' in module:
                    print("    - attn.to_out.0  # Image attention output projection")
                elif 'txt_attn_qkv' in module:
                    print("    - attn.to_added_qkv  # Text attention QKV (fused)")
                    print("    - attn.add_q_proj    # Or individual projections")
                    print("    - attn.add_k_proj")
                    print("    - attn.add_v_proj")
        
        if missing_modules['single']:
            print("\n  For single blocks, add:")
            for module in sorted(missing_modules['single']):
                if 'linear1' in module:
                    print("    - Consider using --flux_lora_target='fal-kontext-fused' for full compatibility")
                elif 'linear2' in module:
                    print("    - attn.to_out.0  # Single block attention output")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare LoRA architectures between two safetensors files"
    )
    parser.add_argument(
        "file1",
        type=str,
        help="Path to first LoRA file (e.g., SimpleTuner output)"
    )
    parser.add_argument(
        "file2", 
        type=str,
        help="Path to second LoRA file (e.g., FAL-kontext target)"
    )
    parser.add_argument(
        "--export-json",
        type=str,
        help="Export comparison results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Load both LoRA files
    print(f"Loading {args.file1}...")
    state_dict1 = safetensors.torch.load_file(args.file1)
    lora1_info = extract_lora_info(state_dict1)
    
    print(f"Loading {args.file2}...")
    state_dict2 = safetensors.torch.load_file(args.file2)
    lora2_info = extract_lora_info(state_dict2)
    
    # Compare architectures
    comparison, cat1, cat2 = compare_architectures(lora1_info, lora2_info)
    
    # Print report
    print_report(args.file1, args.file2, lora1_info, lora2_info, comparison, cat1, cat2)
    
    # Export to JSON if requested
    if args.export_json:
        export_data = {
            'file1': {
                'path': args.file1,
                'info': lora1_info,
                'categories': cat1
            },
            'file2': {
                'path': args.file2,
                'info': lora2_info,
                'categories': cat2
            },
            'comparison': comparison
        }
        
        # Convert defaultdicts to regular dicts for JSON serialization
        def convert_defaultdict(d):
            if isinstance(d, defaultdict):
                d = {k: convert_defaultdict(v) for k, v in d.items()}
            return d
        
        export_data = json.loads(json.dumps(export_data, default=convert_defaultdict))
        
        with open(args.export_json, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\nExported comparison data to {args.export_json}")


if __name__ == "__main__":
    main()