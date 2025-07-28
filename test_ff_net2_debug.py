#!/usr/bin/env python3
"""
Test script to debug missing ff.net.2 and ff_context.net.2 LoRA weights.

This script will:
1. Load the Flux transformer model
2. Examine the structure of ff.net and ff_context modules  
3. Debug why ff.net.2 and ff_context.net.2 aren't getting LoRA adapters
4. Test the target module selection logic
"""

import torch
from diffusers import FluxTransformer2DModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def examine_flux_structure():
    """Load Flux transformer and examine the ff.net structure."""
    logger.info("Loading Flux transformer model...")
    
    # Load the transformer model
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )
    
    logger.info("Examining first transformer block structure...")
    
    # Get the first transformer block
    first_block = transformer.transformer_blocks[0]
    
    logger.info("=== First Block Structure ===")
    logger.info(f"Block type: {type(first_block)}")
    logger.info(f"Block attributes: {list(first_block.__dict__.keys())}")
    
    # Examine ff.net structure
    logger.info("\n=== FF.net Structure ===")
    ff_net = first_block.ff.net
    logger.info(f"ff.net type: {type(ff_net)}")
    logger.info(f"ff.net: {ff_net}")
    
    # Check if ff.net is a ModuleList or Sequential
    if hasattr(ff_net, '__len__'):
        logger.info(f"ff.net length: {len(ff_net)}")
        for i, layer in enumerate(ff_net):
            logger.info(f"  ff.net[{i}]: {type(layer)} - {layer}")
            if hasattr(layer, 'weight'):
                logger.info(f"    Weight shape: {layer.weight.shape}")
    
    # Examine ff_context.net structure  
    logger.info("\n=== FF_context.net Structure ===")
    ff_context_net = first_block.ff_context.net
    logger.info(f"ff_context.net type: {type(ff_context_net)}")
    logger.info(f"ff_context.net: {ff_context_net}")
    
    # Check if ff_context.net is a ModuleList or Sequential
    if hasattr(ff_context_net, '__len__'):
        logger.info(f"ff_context.net length: {len(ff_context_net)}")
        for i, layer in enumerate(ff_context_net):
            logger.info(f"  ff_context.net[{i}]: {type(layer)} - {layer}")
            if hasattr(layer, 'weight'):
                logger.info(f"    Weight shape: {layer.weight.shape}")
    
    return transformer

def test_target_module_selection():
    """Test the target module selection logic for fal-kontext fusion."""
    logger.info("\n=== Testing Target Module Selection ===")
    
    # Import the adapter logic
    import sys
    import os
    sys.path.append('/Users/gt/Sync/ai/tools/SimpleTuner2')
    
    try:
        from helpers.training.adapter import get_target_modules
        
        # Test with fal-kontext-fused configuration
        logger.info("Testing fal-kontext-fused target modules...")
        target_modules = get_target_modules("flux", "fal-kontext-fused")
        logger.info(f"Target modules: {target_modules}")
        
        # Check if ff.net.2 patterns are included
        ff_net_2_patterns = [mod for mod in target_modules if 'ff.net.2' in mod or 'ff_context.net.2' in mod]
        logger.info(f"FF.net.2 related patterns: {ff_net_2_patterns}")
        
        if not ff_net_2_patterns:
            logger.warning("No ff.net.2 or ff_context.net.2 patterns found in target modules!")
        
    except ImportError as e:
        logger.error(f"Could not import adapter module: {e}")
        return None
    
    return target_modules

def check_lora_fusion_mappings():
    """Check the LoRA fusion mappings for fal-kontext."""
    logger.info("\n=== Checking LoRA Fusion Mappings ===")
    
    try:
        from helpers.training.diffusers_overrides import apply_fal_kontext_fusion
        logger.info("Found apply_fal_kontext_fusion function")
        
        # Check if there are any mappings for img_mlp_2 and txt_mlp_2
        import inspect
        source = inspect.getsource(apply_fal_kontext_fusion)
        
        if 'img_mlp_2' in source:
            logger.info("Found img_mlp_2 mapping in fusion code")
        else:
            logger.warning("No img_mlp_2 mapping found in fusion code")
            
        if 'txt_mlp_2' in source:
            logger.info("Found txt_mlp_2 mapping in fusion code")  
        else:
            logger.warning("No txt_mlp_2 mapping found in fusion code")
            
    except ImportError as e:
        logger.error(f"Could not import diffusers_overrides: {e}")

def main():
    """Main test function."""
    logger.info("Starting FF.net.2 debug analysis...")
    
    # 1. Examine Flux structure
    transformer = examine_flux_structure()
    
    # 2. Test target module selection
    target_modules = test_target_module_selection()
    
    # 3. Check LoRA fusion mappings
    check_lora_fusion_mappings()
    
    # 4. Summary and recommendations
    logger.info("\n=== Summary ===")
    logger.info("Analysis complete. Check the output above for:")
    logger.info("1. Whether ff.net.2 and ff_context.net.2 exist as Linear modules")
    logger.info("2. Whether these modules are included in target_modules")
    logger.info("3. Whether fusion mappings exist for img_mlp_2/txt_mlp_2")

if __name__ == "__main__":
    main()