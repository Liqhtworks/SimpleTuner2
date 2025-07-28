#!/usr/bin/env python3
"""
Test that our module mappings are consistent across all files.
"""

# Expected mappings based on our analysis
EXPECTED_DIFFUSERS_TO_FAL = {
    # Image path
    "attn.to_qkv": "img_attn_qkv",
    "attn.to_out.0": "img_attn_proj", 
    "norm1.linear": "img_mod_lin",
    "ff.net.0.proj": "img_mlp_0",
    "ff.net.2": "img_mlp_2",
    
    # Text path
    "attn.to_added_qkv": "txt_attn_qkv",  # Should be to_added_qkv, not add_qkv_proj
    "attn.to_add_out": "txt_attn_proj",
    "norm1_context.linear": "txt_mod_lin", 
    "ff_context.net.0.proj": "txt_mlp_0",
    "ff_context.net.2": "txt_mlp_2",
    
    # Single blocks
    "norm.linear": "modulation_lin",
}

EXPECTED_SCALING_FACTORS = {
    "attn.to_qkv": 3,
    "attn.to_added_qkv": 3,
    "norm1.linear": 6,
    "norm1_context.linear": 6, 
    "norm.linear": 3,
    "ff.net.0.proj": 3,
    "ff.net.2": 3,
    "ff_context.net.0.proj": 3,
    "ff_context.net.2": 3,
    "default": 1,
}

def check_adapter_mappings():
    """Check adapter.py mappings"""
    import sys
    sys.path.append('/Users/gt/Sync/ai/tools/SimpleTuner2')
    
    from helpers.training.adapter import FAL_KONTEXT_KEY_MAPPING, FAL_KONTEXT_SCALING_FACTORS
    
    print("=== Checking adapter.py mappings ===")
    
    # Check key mappings
    for fal_key, expected_diffusers in [
        ("img_attn_qkv", "attn.to_qkv"),
        ("txt_attn_qkv", "attn.to_added_qkv"),
        ("img_attn_proj", "attn.to_out.0"),
        ("txt_attn_proj", "attn.to_add_out"),
    ]:
        actual = FAL_KONTEXT_KEY_MAPPING.get(fal_key)
        if actual != expected_diffusers:
            print(f"❌ {fal_key}: expected {expected_diffusers}, got {actual}")
        else:
            print(f"✅ {fal_key}: {actual}")
    
    # Check scaling factors
    for module, expected_factor in [
        ("attn.to_qkv", 3),
        ("attn.to_added_qkv", 3),
        ("norm1.linear", 6),
        ("norm1_context.linear", 6),
    ]:
        actual = FAL_KONTEXT_SCALING_FACTORS.get(module)
        if actual != expected_factor:
            print(f"❌ Scaling {module}: expected {expected_factor}, got {actual}")
        else:
            print(f"✅ Scaling {module}: {actual}")

def check_convert_script_mappings():
    """Check convert script mappings"""
    import sys
    sys.path.append('/Users/gt/Sync/ai/tools/SimpleTuner2')
    
    from convert_lora_to_fal_kontext import convert_simpletuner_to_fal_kontext
    
    print("\\n=== Checking convert script ===")
    
    # Create a fake state dict to test conversion
    test_state_dict = {
        "transformer.transformer_blocks.0.attn.to_qkv.lora_A.weight": "fake_tensor_img_qkv_A",
        "transformer.transformer_blocks.0.attn.to_qkv.lora_B.weight": "fake_tensor_img_qkv_B", 
        "transformer.transformer_blocks.0.attn.to_added_qkv.lora_A.weight": "fake_tensor_txt_qkv_A",
        "transformer.transformer_blocks.0.attn.to_added_qkv.lora_B.weight": "fake_tensor_txt_qkv_B",
        "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight": "fake_tensor_img_proj_A",
        "transformer.transformer_blocks.0.attn.to_add_out.lora_A.weight": "fake_tensor_txt_proj_A",
    }
    
    converted = convert_simpletuner_to_fal_kontext(test_state_dict)
    
    expected_keys = [
        "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight",
        "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight",
        "lora_unet_double_blocks_0_txt_attn_qkv.lora_down.weight", 
        "lora_unet_double_blocks_0_txt_attn_qkv.lora_up.weight",
        "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight",
        "lora_unet_double_blocks_0_txt_attn_proj.lora_down.weight",
    ]
    
    for key in expected_keys:
        if key in converted:
            print(f"✅ Converted key: {key}")
        else:
            print(f"❌ Missing key: {key}")
    
    print(f"\\nTotal converted keys: {len(converted)}")
    if len(converted) < len(expected_keys):
        print("Available keys:")
        for k in sorted(converted.keys()):
            print(f"  {k}")

if __name__ == "__main__":
    check_adapter_mappings()
    check_convert_script_mappings()