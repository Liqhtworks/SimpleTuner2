#!/usr/bin/env python3
"""Test what modules PEFT can find after fusion"""

import torch
from diffusers import FluxTransformer2DModel
from peft import LoraConfig, get_peft_model
from helpers.training.diffusers_overrides import fuse_all_blocks_fal_kontext

# Load model
model = FluxTransformer2DModel.from_pretrained(
    "/home/playerzer0x/ComfyUI/models/unet/FLUX.1-Kontext-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

# Apply fusion
fusion_count = fuse_all_blocks_fal_kontext(model, permanent=True)
print(f"Fused {fusion_count} blocks")

# Check what modules exist in first single block
block = model.single_transformer_blocks[0]
print("\nModules in first single block:")
for name, module in block.named_modules():
    if name:  # Skip the block itself
        print(f"  {name}: {type(module).__name__}")

# Try different target module combinations
target_configs = [
    ["linear1", "linear2", "modulation_lin"],
    ["linear1", "proj_out", "norm.linear"],
    ["linear1"],
]

for i, target_modules in enumerate(target_configs):
    print(f"\n\n=== Test {i+1}: Target modules = {target_modules} ===")
    
    try:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.0,
        )
        
        model_with_lora = get_peft_model(model, lora_config)
        
        # Count LoRA modules by type
        lora_counts = {}
        for name, _ in model_with_lora.named_modules():
            if "lora" in name and "single_transformer_blocks.0" in name:
                # Extract the module type
                for target in target_modules:
                    if target.replace(".", "_") in name:
                        module_type = target
                        lora_counts[module_type] = lora_counts.get(module_type, 0) + 1
                        break
        
        print(f"Found LoRA modules: {lora_counts}")
        
        # Show a few example module names
        print("Example LoRA module names:")
        count = 0
        for name, _ in model_with_lora.named_modules():
            if "lora" in name and "single_transformer_blocks.0" in name:
                print(f"  {name}")
                count += 1
                if count >= 5:
                    break
                    
    except Exception as e:
        print(f"Error: {e}")