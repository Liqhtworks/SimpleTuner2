#!/usr/bin/env python3
"""
Convert SimpleTuner/diffusers LoRA files to FAL-kontext compatible format.

This script converts SimpleTuner trained LoRA files to the naming convention
and scaling factors expected by FAL-kontext.

Usage:
    python convert_lora_to_fal_kontext.py input_lora.safetensors output_lora.safetensors

The conversion performs:
1. Key name conversion: transformer_blocks.0.attn.to_qkv -> lora_unet_double_blocks_0_img_attn_qkv
2. Matrix name swap: lora_A/lora_B -> lora_down/lora_up
3. Inverse scaling factor application for proper FAL-kontext compatibility
"""

import argparse
import logging
import sys
from pathlib import Path

import safetensors.torch
import torch

# Embedded conversion function to avoid import dependencies
FAL_KONTEXT_SCALING_FACTORS = {
    # QKV projections output 3x the input dimension (Q, K, V)
    "attn.to_qkv": 3,
    "attn.add_qkv_proj": 3,
    # Double block modulation outputs 6x (gate_msa, shift_mlp, scale_mlp, gate_mlp, and 2 more)
    "norm1.linear": 6,
    "norm1_context.linear": 6,
    # Single block modulation outputs 3x (gate, shift, scale)
    "norm.linear": 3,
    # MLP layers that need 3x scaling
    "ff.net.0.proj": 3,
    "ff.net.2": 3,
    "ff_context.net.0.proj": 3,
    "ff_context.net.2": 3,
    # Regular layers have no scaling
    "default": 1,
}

def get_fal_kontext_scaling_factor(module_name):
    """Get the scaling factor for a specific module."""
    # Check each pattern in the scaling factors
    for pattern, factor in FAL_KONTEXT_SCALING_FACTORS.items():
        if pattern in module_name:
            return factor
    return FAL_KONTEXT_SCALING_FACTORS["default"]

def convert_simpletuner_to_fal_kontext(state_dict):
    """
    Convert SimpleTuner/diffusers LoRA format to FAL-kontext format.
    
    This function performs the reverse mapping of get_fal_kontext_mapped_key():
    1. Converts key names from diffusers format to FAL-kontext format
    2. Swaps lora_A/lora_B to lora_down/lora_up
    3. Applies inverse scaling factors
    
    Args:
        state_dict: Dictionary containing SimpleTuner LoRA weights
        
    Returns:
        Dictionary with FAL-kontext compatible weights
    """
    # Create direct mapping from diffusers modules to FAL-kontext names
    diffusers_to_fal_mapping = {
        # Double block mappings (MMDiT blocks)
        "attn.to_qkv": "img_attn_qkv",
        "attn.add_qkv_proj": "txt_attn_qkv", 
        "attn.to_out.0": "img_attn_proj",
        "attn.to_add_out": "txt_attn_proj",
        "norm1.linear": "img_mod_lin",
        "norm1_context.linear": "txt_mod_lin",
        "ff.net.0.proj": "img_mlp_0",
        "ff.net.2": "img_mlp_2",
        "ff_context.net.0.proj": "txt_mlp_0",
        "ff_context.net.2": "txt_mlp_2",
        # Single block mappings (DiT blocks) - these use different naming
        # For single blocks, we'll handle them specially
        "norm.linear": "modulation_lin",
    }
    
    fal_state_dict = {}
    
    for key, weight in state_dict.items():
        # Skip non-LoRA keys
        if not any(suffix in key for suffix in [".lora_A.weight", ".lora_B.weight", ".alpha", ".lora_alpha"]):
            continue
            
        # Extract base key and weight type
        if ".lora_A.weight" in key:
            base_key = key.replace(".lora_A.weight", "")
            weight_type = "lora_down"  # FAL-kontext uses lora_down for A matrices
        elif ".lora_B.weight" in key:
            base_key = key.replace(".lora_B.weight", "")
            weight_type = "lora_up"    # FAL-kontext uses lora_up for B matrices
        elif ".alpha" in key or ".lora_alpha" in key:
            base_key = key.replace(".lora_alpha", "").replace(".alpha", "")
            weight_type = "alpha"
        else:
            continue
            
        # Remove transformer prefix if present
        if base_key.startswith("transformer."):
            base_key = base_key[len("transformer."):]
            
        # Parse the structure to convert to FAL-kontext naming
        fal_key = None
        
        # Handle transformer blocks
        if base_key.startswith("transformer_blocks."):
            # Double blocks: transformer_blocks.X.module -> lora_unet_double_blocks_X_module
            parts = base_key.split(".", 2)  # ["transformer_blocks", "X", "module.path"]
            if len(parts) >= 3:
                block_num = parts[1]
                module_path = parts[2]
                
                # Map to FAL-kontext module name
                if module_path in diffusers_to_fal_mapping:
                    fal_module = diffusers_to_fal_mapping[module_path]
                    fal_key = f"lora_unet_double_blocks_{block_num}_{fal_module}"
                    
        elif base_key.startswith("single_transformer_blocks."):
            # Single blocks: single_transformer_blocks.X.module -> lora_unet_single_blocks_X_module
            parts = base_key.split(".", 2)  # ["single_transformer_blocks", "X", "module.path"]
            if len(parts) >= 3:
                block_num = parts[1]
                module_path = parts[2]
                
                # For single blocks, FAL-kontext uses different naming
                if module_path == "attn.to_qkv":
                    # Single block QKV is part of linear1 in FAL-kontext
                    fal_key = f"lora_unet_single_blocks_{block_num}_linear1"
                elif module_path == "attn.to_out.0":
                    fal_key = f"lora_unet_single_blocks_{block_num}_linear2"
                elif module_path in diffusers_to_fal_mapping:
                    fal_module = diffusers_to_fal_mapping[module_path]
                    fal_key = f"lora_unet_single_blocks_{block_num}_{fal_module}"
                    
        elif base_key == "proj_out":
            # Global projection layer
            fal_key = "lora_unet_final_layer_linear"
            
        # If we found a mapping, process the weight
        if fal_key:
            if weight_type in ["lora_down", "lora_up"]:
                final_key = f"{fal_key}.{weight_type}.weight"
                
                # Apply inverse scaling for lora_up (B) weights
                if weight_type == "lora_up":
                    # Get the original module name to determine scaling factor
                    original_module = base_key
                    if base_key.startswith("transformer_blocks.") or base_key.startswith("single_transformer_blocks."):
                        # Extract just the module part for scaling lookup
                        parts = base_key.split(".", 2)
                        if len(parts) >= 3:
                            original_module = parts[2]
                    
                    scaling_factor = get_fal_kontext_scaling_factor(original_module)
                    if scaling_factor > 1:
                        # Apply inverse scaling
                        weight = weight / scaling_factor
                        
                fal_state_dict[final_key] = weight
                
            elif weight_type == "alpha":
                final_key = f"{fal_key}.alpha"
                fal_state_dict[final_key] = weight
                
    return fal_state_dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SimpleTuner LoRA to FAL-kontext format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input_file", 
        type=str,
        help="Path to input SimpleTuner LoRA safetensors file"
    )
    parser.add_argument(
        "output_file", 
        type=str,
        help="Path to output FAL-kontext compatible safetensors file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output showing key mappings"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be converted without writing output file"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)
        
    if not input_path.suffix.lower() == '.safetensors':
        logger.error(f"Input file must be a .safetensors file: {input_path}")
        sys.exit(1)
    
    # Validate output path
    output_path = Path(args.output_file)
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not output_path.suffix.lower() == '.safetensors':
            logger.error(f"Output file must be a .safetensors file: {output_path}")
            sys.exit(1)
    
    try:
        # Load the input LoRA
        logger.info(f"Loading SimpleTuner LoRA from: {input_path}")
        state_dict = safetensors.torch.load_file(str(input_path))
        
        logger.info(f"Found {len(state_dict)} tensors in input file")
        
        # Check if it looks like a SimpleTuner LoRA
        simpletuner_keys = [k for k in state_dict.keys() if any(suffix in k for suffix in [".lora_A.weight", ".lora_B.weight"])]
        if not simpletuner_keys:
            logger.warning("Input file doesn't appear to contain SimpleTuner LoRA weights (no .lora_A.weight or .lora_B.weight keys found)")
            logger.info("Available keys:")
            for k in list(state_dict.keys())[:10]:  # Show first 10 keys
                logger.info(f"  {k}")
            if len(state_dict) > 10:
                logger.info(f"  ... and {len(state_dict) - 10} more")
        
        # Convert to FAL-kontext format
        logger.info("Converting to FAL-kontext format...")
        fal_state_dict = convert_simpletuner_to_fal_kontext(state_dict)
        
        if not fal_state_dict:
            logger.error("Conversion resulted in empty state dict. Check if input is a valid SimpleTuner LoRA.")
            sys.exit(1)
            
        logger.info(f"Conversion complete. Output contains {len(fal_state_dict)} tensors")
        
        if args.verbose or args.dry_run:
            logger.info("Key mappings:")
            # Show some example mappings
            original_keys = [k for k in state_dict.keys() if ".lora_A.weight" in k][:5]
            for orig_key in original_keys:
                # Find corresponding converted key
                base_key = orig_key.replace(".lora_A.weight", "")
                converted_keys = [k for k in fal_state_dict.keys() if base_key.split('.')[-1] in k and "lora_down" in k]
                if converted_keys:
                    logger.info(f"  {orig_key} -> {converted_keys[0]}")
                    
        if args.dry_run:
            logger.info("Dry run complete. No files were written.")
            return
            
        # Save the converted LoRA
        logger.info(f"Saving FAL-kontext LoRA to: {output_path}")
        safetensors.torch.save_file(fal_state_dict, str(output_path))
        
        logger.info("Conversion completed successfully!")
        logger.info(f"FAL-kontext LoRA saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()