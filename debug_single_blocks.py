#!/usr/bin/env python3
"""Debug script to check single block LoRA module discovery"""

import torch
from diffusers import FluxTransformer2DModel
from peft import LoraConfig, get_peft_model
import sys

def inspect_single_block(model):
    """Inspect a single transformer block structure"""
    single_block = model.single_transformer_blocks[0]
    
    print("\n=== Single Block Module Structure ===")
    print(f"Block type: {type(single_block)}")
    
    # Check for expected modules
    modules_to_check = ['linear1', 'linear2', 'modulation_lin', 'norm', 'proj_out', 'attn']
    for module_name in modules_to_check:
        if hasattr(single_block, module_name):
            module = getattr(single_block, module_name)
            print(f"\n{module_name}: exists")
            if hasattr(module, 'weight'):
                print(f"  - weight shape: {module.weight.shape}")
            if module_name == 'norm' and hasattr(module, 'linear'):
                print(f"  - norm.linear weight shape: {module.linear.weight.shape}")
            if module_name == 'attn':
                for submodule in ['to_q', 'to_k', 'to_v', 'to_qkv', 'to_out']:
                    if hasattr(module, submodule):
                        print(f"  - has {submodule}")
        else:
            print(f"\n{module_name}: NOT FOUND")

def test_lora_discovery():
    """Test LoRA module discovery on single blocks"""
    print("Loading model...")
    model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    
    # First check: before fusion
    print("\n=== BEFORE FUSION ===")
    inspect_single_block(model)
    
    # Apply fusion if specified
    if "--fuse" in sys.argv:
        print("\n\n=== APPLYING FUSION ===")
        from helpers.training.diffusers_overrides import fuse_all_blocks_fal_kontext
        fusion_count = fuse_all_blocks_fal_kontext(model, permanent=True)
        print(f"Fused {fusion_count} blocks")
        
        print("\n=== AFTER FUSION ===")
        inspect_single_block(model)
    
    # Try to apply LoRA
    print("\n\n=== APPLYING LORA ===")
    target_modules = [
        "linear1", "linear2", "modulation_lin",
        "norm.linear", "proj_out", "attn.to_qkv"
    ]
    
    print(f"Target modules: {target_modules}")
    
    try:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.0,
        )
        
        model_with_lora = get_peft_model(model, lora_config)
        
        print("\n=== DISCOVERED LORA MODULES ===")
        lora_modules = []
        for name, module in model_with_lora.named_modules():
            if "lora" in name and "single_transformer_blocks" in name:
                lora_modules.append(name)
        
        lora_modules.sort()
        for module in lora_modules:
            print(f"  {module}")
            
        # Count by type
        linear1_count = sum(1 for m in lora_modules if "linear1" in m)
        linear2_count = sum(1 for m in lora_modules if "linear2" in m)
        modulation_count = sum(1 for m in lora_modules if "modulation_lin" in m or "norm.linear" in m)
        
        print(f"\nCounts:")
        print(f"  linear1: {linear1_count}")
        print(f"  linear2: {linear2_count}")
        print(f"  modulation_lin: {modulation_count}")
        
    except Exception as e:
        print(f"\nError applying LoRA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lora_discovery()