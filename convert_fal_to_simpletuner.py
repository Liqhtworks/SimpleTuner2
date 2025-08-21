#!/usr/bin/env python3
"""
Convert FAL-style Kontext LoRA weights to SimpleTuner format.

Key differences:
1. FAL uses fused naming: img_attn_qkv, txt_attn_qkv (single layer)
   SimpleTuner uses: to_qkv, add_qkv_proj (for img/txt respectively when fused)
2. FAL scales FF layer B weights down by 1/3 during export
   SimpleTuner scales them up by 3 during load
3. Layer naming conventions differ between the two formats
"""

import argparse
import torch
import safetensors.torch
from pathlib import Path
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_layer_mapping() -> Dict[str, str]:
    """
    Create mapping from FAL layer names to SimpleTuner layer names.
    
    FAL structure:
    - lora_unet_double_blocks_{i}_{stream}_{component}.lora_{down/up}
    - lora_unet_single_blocks_{i}_{component}.lora_{down/up}
    
    SimpleTuner structure:
    - transformer.transformer_blocks.{i}.{component}.lora_{A/B}
    - transformer.single_transformer_blocks.{i}.{component}.lora_{A/B}
    """
    mapping = {}
    
    # Double blocks (0-18 in FAL correspond to transformer_blocks in SimpleTuner)
    for i in range(19):
        # Image stream attention layers
        mapping[f"lora_unet_double_blocks_{i}_img_attn_qkv"] = f"transformer.transformer_blocks.{i}.attn.to_qkv"
        mapping[f"lora_unet_double_blocks_{i}_img_attn_proj"] = f"transformer.transformer_blocks.{i}.attn.to_out.0"
        
        # Text stream attention layers
        mapping[f"lora_unet_double_blocks_{i}_txt_attn_qkv"] = f"transformer.transformer_blocks.{i}.attn.add_qkv_proj"
        mapping[f"lora_unet_double_blocks_{i}_txt_attn_proj"] = f"transformer.transformer_blocks.{i}.attn.to_add_out"
        
        # Image stream FF layers
        mapping[f"lora_unet_double_blocks_{i}_img_mlp_0"] = f"transformer.transformer_blocks.{i}.ff.net.0.proj"
        mapping[f"lora_unet_double_blocks_{i}_img_mlp_2"] = f"transformer.transformer_blocks.{i}.ff.net.2"
        
        # Text stream FF layers
        mapping[f"lora_unet_double_blocks_{i}_txt_mlp_0"] = f"transformer.transformer_blocks.{i}.ff_context.net.0.proj"
        mapping[f"lora_unet_double_blocks_{i}_txt_mlp_2"] = f"transformer.transformer_blocks.{i}.ff_context.net.2"
        
        # Modulation layers
        mapping[f"lora_unet_double_blocks_{i}_img_mod_lin"] = f"transformer.transformer_blocks.{i}.norm1.linear"
        mapping[f"lora_unet_double_blocks_{i}_txt_mod_lin"] = f"transformer.transformer_blocks.{i}.norm1_context.linear"
    
    # Single blocks (0-37 in FAL correspond to single_transformer_blocks in SimpleTuner)
    for i in range(38):
        # Single block MLP layers
        mapping[f"lora_unet_single_blocks_{i}_linear1"] = f"transformer.single_transformer_blocks.{i}.proj_mlp"
        mapping[f"lora_unet_single_blocks_{i}_linear2"] = f"transformer.single_transformer_blocks.{i}.proj_out"
        
        # Single block modulation
        mapping[f"lora_unet_single_blocks_{i}_modulation_lin"] = f"transformer.single_transformer_blocks.{i}.norm.linear"
    
    # Final layer
    mapping["lora_unet_final_layer_linear"] = "transformer.proj_out"
    
    return mapping


def convert_weights(fal_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert FAL LoRA weights to SimpleTuner format.
    
    Args:
        fal_weights: Dictionary of FAL format weights
        
    Returns:
        Dictionary of SimpleTuner format weights
    """
    layer_mapping = create_layer_mapping()
    simpletuner_weights = {}
    
    # Track which weights we've processed
    processed_keys = set()
    unmatched_keys = []
    
    for fal_key, weight in fal_weights.items():
        # Extract the base layer name and suffix
        if ".lora_down.weight" in fal_key:
            base_name = fal_key.replace(".lora_down.weight", "")
            suffix = ".lora_A.weight"
        elif ".lora_up.weight" in fal_key:
            base_name = fal_key.replace(".lora_up.weight", "")
            suffix = ".lora_B.weight"
        else:
            # Skip non-LoRA weights (e.g., alpha parameters if present)
            logger.debug(f"Skipping non-LoRA weight: {fal_key}")
            continue
        
        # Find the corresponding SimpleTuner layer name
        if base_name in layer_mapping:
            simpletuner_name = layer_mapping[base_name]
            new_key = simpletuner_name + suffix
            
            # Apply scaling for FF layers' B weights
            # FAL scales these down by 1/3, SimpleTuner expects them unscaled
            # (SimpleTuner will scale them up by 3 during load)
            if suffix == ".lora_B.weight" and ("ff.net." in simpletuner_name or "ff_context.net." in simpletuner_name):
                # FAL has already scaled down by 1/3, we need to restore the original scale
                weight = weight * 3.0
                logger.debug(f"Scaling up FF layer B weight: {new_key}")
            
            simpletuner_weights[new_key] = weight
            processed_keys.add(fal_key)
        else:
            unmatched_keys.append(fal_key)
    
    # Log statistics
    logger.info(f"Converted {len(processed_keys)} weights")
    if unmatched_keys:
        logger.warning(f"Unmatched keys ({len(unmatched_keys)}): {unmatched_keys[:5]}...")
    
    return simpletuner_weights


def add_adapter_config(weights: Dict[str, torch.Tensor], rank: int = None) -> Dict[str, torch.Tensor]:
    """
    Add LoRA alpha parameters if not present.
    SimpleTuner expects alpha parameters for each LoRA layer.
    """
    # Detect rank if not provided
    if rank is None:
        for key, tensor in weights.items():
            if "lora_A.weight" in key:
                rank = tensor.shape[0]
                logger.info(f"Detected LoRA rank: {rank}")
                break
    
    # Add alpha parameters (typically equal to rank)
    alpha_keys = set()
    for key in list(weights.keys()):
        if "lora_A.weight" in key:
            base_key = key.replace(".lora_A.weight", "")
            alpha_key = base_key + ".alpha"
            if alpha_key not in weights:
                weights[alpha_key] = torch.tensor(float(rank))
                alpha_keys.add(alpha_key)
    
    if alpha_keys:
        logger.info(f"Added {len(alpha_keys)} alpha parameters")
    
    return weights


def main():
    parser = argparse.ArgumentParser(description="Convert FAL Kontext LoRA to SimpleTuner format")
    parser.add_argument("input_path", type=str, help="Path to FAL LoRA safetensors file")
    parser.add_argument("output_path", type=str, help="Path for output SimpleTuner LoRA file")
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank (auto-detected if not provided)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load FAL weights
    logger.info(f"Loading FAL LoRA from: {args.input_path}")
    fal_weights = safetensors.torch.load_file(args.input_path)
    logger.info(f"Loaded {len(fal_weights)} tensors")
    
    # Convert to SimpleTuner format
    logger.info("Converting weights to SimpleTuner format...")
    simpletuner_weights = convert_weights(fal_weights)
    
    # Add alpha parameters
    simpletuner_weights = add_adapter_config(simpletuner_weights, args.rank)
    
    # Save converted weights
    logger.info(f"Saving SimpleTuner LoRA to: {args.output_path}")
    safetensors.torch.save_file(simpletuner_weights, args.output_path)
    logger.info(f"Successfully converted {len(simpletuner_weights)} tensors")
    
    # Print summary
    print("\nConversion Summary:")
    print(f"  Input: {args.input_path}")
    print(f"  Output: {args.output_path}")
    print(f"  Total weights converted: {len(simpletuner_weights)}")
    
    # List layer types
    layer_types = set()
    for key in simpletuner_weights.keys():
        if "lora_A" in key or "lora_B" in key:
            parts = key.split(".")
            if "transformer_blocks" in key:
                layer_types.add("transformer_blocks")
            elif "single_transformer_blocks" in key:
                layer_types.add("single_transformer_blocks")
            elif "proj_out" in key:
                layer_types.add("proj_out")
    print(f"  Layer types: {', '.join(sorted(layer_types))}")


if __name__ == "__main__":
    main()