#!/usr/bin/env python3
"""
Test script to verify double block fusion logic and conversion mappings.
"""

import logging
import sys
from pathlib import Path
import torch

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_structure_analysis():
    """Test the structural differences between SimpleTuner and FAL-kontext for double blocks."""
    print("Testing double block structural analysis...")
    
    # Based on lora_mapping_chart.md analysis
    simpletuner_layers = {
        # Image path
        "attn.to_qkv": {"shape": [9216, 16], "type": "lora_A/lora_B", "purpose": "Image attention QKV (3×3072)"},
        "attn.to_out.0": {"shape": [3072, 16], "type": "lora_A/lora_B", "purpose": "Image attention output"},
        "norm1.linear": {"shape": [18432, 16], "type": "lora_A/lora_B", "purpose": "Image modulation (6×3072)"},
        "ff.net.0.proj": {"shape": [12288, 16], "type": "lora_A/lora_B", "purpose": "Image MLP input (4×3072)"},
        "ff.net.2": {"shape": [3072, 16], "type": "lora_A/lora_B", "purpose": "Image MLP output"},
        
        # Text path
        "attn.to_added_qkv": {"shape": [9216, 16], "type": "lora_A/lora_B", "purpose": "Text attention QKV (3×3072)"},
        "attn.to_add_out": {"shape": [3072, 16], "type": "lora_A/lora_B", "purpose": "Text attention output"},
        "norm1_context.linear": {"shape": [18432, 16], "type": "lora_A/lora_B", "purpose": "Text modulation (6×3072)"},
        "ff_context.net.0.proj": {"shape": [12288, 16], "type": "lora_A/lora_B", "purpose": "Text MLP input (4×3072)"},
        "ff_context.net.2": {"shape": [3072, 16], "type": "lora_A/lora_B", "purpose": "Text MLP output"},
    }
    
    fal_kontext_layers = {
        # Image path
        "img_attn_qkv": {"shape": [9216, 16], "type": "lora_down/lora_up", "purpose": "Image attention QKV (3×3072)"},
        "img_attn_proj": {"shape": [3072, 16], "type": "lora_down/lora_up", "purpose": "Image attention output"},
        "img_mod_lin": {"shape": [18432, 16], "type": "lora_down/lora_up", "purpose": "Image modulation (6×3072)"},
        "img_mlp_0": {"shape": [12288, 16], "type": "lora_down/lora_up", "purpose": "Image MLP input (4×3072)"},
        "img_mlp_2": {"shape": [3072, 16], "type": "lora_down/lora_up", "purpose": "Image MLP output"},
        
        # Text path
        "txt_attn_qkv": {"shape": [9216, 16], "type": "lora_down/lora_up", "purpose": "Text attention QKV (3×3072)"},
        "txt_attn_proj": {"shape": [3072, 16], "type": "lora_down/lora_up", "purpose": "Text attention output"},
        "txt_mod_lin": {"shape": [18432, 16], "type": "lora_down/lora_up", "purpose": "Text modulation (6×3072)"},
        "txt_mlp_0": {"shape": [12288, 16], "type": "lora_down/lora_up", "purpose": "Text MLP input (4×3072)"},
        "txt_mlp_2": {"shape": [3072, 16], "type": "lora_down/lora_up", "purpose": "Text MLP output"},
    }
    
    print("✓ SimpleTuner double block layers:")
    for name, info in simpletuner_layers.items():
        print(f"  - {name}: {info['shape']} ({info['type']}) - {info['purpose']}")
    
    print("\n✓ FAL-kontext double block layers:")
    for name, info in fal_kontext_layers.items():
        print(f"  - {name}: {info['shape']} ({info['type']}) - {info['purpose']}")
    
    # Key mappings
    mappings = [
        ("SimpleTuner attn.to_qkv", "FAL img_attn_qkv"),
        ("SimpleTuner attn.to_out.0", "FAL img_attn_proj"),
        ("SimpleTuner attn.to_added_qkv", "FAL txt_attn_qkv"),
        ("SimpleTuner attn.to_add_out", "FAL txt_attn_proj"),
        ("SimpleTuner norm1.linear", "FAL img_mod_lin"),
        ("SimpleTuner norm1_context.linear", "FAL txt_mod_lin"),
        ("SimpleTuner ff.net.0.proj", "FAL img_mlp_0"),
        ("SimpleTuner ff.net.2", "FAL img_mlp_2"),
        ("SimpleTuner ff_context.net.0.proj", "FAL txt_mlp_0"),
        ("SimpleTuner ff_context.net.2", "FAL txt_mlp_2"),
    ]
    
    print("\n✓ Key mappings identified:")
    for mapping in mappings:
        print(f"  - {mapping[0]} -> {mapping[1]}")
    
    return True

def test_conversion_mapping():
    """Test that the conversion script mapping is correct for double blocks."""
    print("\nTesting double block conversion script mapping...")
    
    # Test mapping for different module paths
    test_cases = [
        # Image path
        ("transformer_blocks.0.attn.to_qkv", "lora_unet_double_blocks_0_img_attn_qkv"),
        ("transformer_blocks.0.attn.to_out.0", "lora_unet_double_blocks_0_img_attn_proj"),
        ("transformer_blocks.0.norm1.linear", "lora_unet_double_blocks_0_img_mod_lin"),
        ("transformer_blocks.0.ff.net.0.proj", "lora_unet_double_blocks_0_img_mlp_0"),
        ("transformer_blocks.0.ff.net.2", "lora_unet_double_blocks_0_img_mlp_2"),
        
        # Text path
        ("transformer_blocks.0.attn.to_added_qkv", "lora_unet_double_blocks_0_txt_attn_qkv"),
        ("transformer_blocks.0.attn.to_add_out", "lora_unet_double_blocks_0_txt_attn_proj"),
        ("transformer_blocks.0.norm1_context.linear", "lora_unet_double_blocks_0_txt_mod_lin"),
        ("transformer_blocks.0.ff_context.net.0.proj", "lora_unet_double_blocks_0_txt_mlp_0"),
        ("transformer_blocks.0.ff_context.net.2", "lora_unet_double_blocks_0_txt_mlp_2"),
    ]
    
    # Import the conversion function
    from convert_lora_to_fal_kontext import convert_simpletuner_to_fal_kontext
    
    # Create mock state dict
    state_dict = {}
    for diffusers_key, expected_fal_key in test_cases:
        # Add lora_A and lora_B weights with appropriate dimensions
        if "qkv" in diffusers_key or "mod_lin" in expected_fal_key:
            # QKV or modulation layers have larger output dimensions
            if "qkv" in diffusers_key:
                output_dim = 9216  # 3 * 3072
            else:  # modulation
                output_dim = 18432  # 6 * 3072
            state_dict[f"transformer.{diffusers_key}.lora_A.weight"] = torch.randn(16, 3072)
            state_dict[f"transformer.{diffusers_key}.lora_B.weight"] = torch.randn(output_dim, 16)
        elif "mlp_0" in expected_fal_key:
            # MLP input layers
            state_dict[f"transformer.{diffusers_key}.lora_A.weight"] = torch.randn(16, 3072)
            state_dict[f"transformer.{diffusers_key}.lora_B.weight"] = torch.randn(12288, 16)  # 4 * 3072
        elif "mlp_2" in expected_fal_key:
            # MLP output layers
            state_dict[f"transformer.{diffusers_key}.lora_A.weight"] = torch.randn(16, 12288)  # 4 * 3072
            state_dict[f"transformer.{diffusers_key}.lora_B.weight"] = torch.randn(3072, 16)
        else:
            # Standard attention output layers
            state_dict[f"transformer.{diffusers_key}.lora_A.weight"] = torch.randn(16, 3072)
            state_dict[f"transformer.{diffusers_key}.lora_B.weight"] = torch.randn(3072, 16)
    
    # Convert
    try:
        fal_dict = convert_simpletuner_to_fal_kontext(state_dict)
        print("✓ Conversion completed without errors")
        
        # Check mappings
        for diffusers_key, expected_fal_key in test_cases:
            found_down = f"{expected_fal_key}.lora_down.weight" in fal_dict
            found_up = f"{expected_fal_key}.lora_up.weight" in fal_dict
            
            if found_down and found_up:
                print(f"✓ {diffusers_key} -> {expected_fal_key}")
                
                # Verify dimensions match expected
                down_weight = fal_dict[f"{expected_fal_key}.lora_down.weight"]
                up_weight = fal_dict[f"{expected_fal_key}.lora_up.weight"]
                
                # Check that dimensions are correct (lora_A -> lora_down, lora_B -> lora_up)
                expected_down_shape = state_dict[f"transformer.{diffusers_key}.lora_A.weight"].shape
                expected_up_shape = state_dict[f"transformer.{diffusers_key}.lora_B.weight"].shape
                
                if list(down_weight.shape) == list(expected_down_shape) and list(up_weight.shape) == list(expected_up_shape):
                    print(f"  ✓ Dimensions correct: down {list(down_weight.shape)}, up {list(up_weight.shape)}")
                else:
                    print(f"  ✗ Dimension mismatch: expected down {list(expected_down_shape)}, up {list(expected_up_shape)}")
                    print(f"    Got down {list(down_weight.shape)}, up {list(up_weight.shape)}")
                    return False
            else:
                print(f"✗ {diffusers_key} -> {expected_fal_key} (missing keys)")
                return False
                
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All double block conversion mappings are correct!")
    return True

def test_scaling_factors():
    """Test that scaling factors match the expected values."""
    print("\nTesting scaling factor calculations...")
    
    base_dim = 3072
    scaling_tests = [
        ("QKV projection", 9216, 3, "3×3072 for query, key, value"),
        ("Double modulation", 18432, 6, "6×3072 for modulation parameters"),
        ("MLP expansion", 12288, 4, "4×3072 for feedforward expansion"),
        ("Base dimension", 3072, 1, "1×3072 for standard projections"),
    ]
    
    print("✓ Scaling factor verification:")
    for desc, dim, expected_factor, explanation in scaling_tests:
        actual_factor = dim // base_dim
        if actual_factor == expected_factor:
            print(f"  ✓ {desc}: {dim} = {expected_factor}×{base_dim} ({explanation})")
        else:
            print(f"  ✗ {desc}: {dim} ≠ {expected_factor}×{base_dim} (got {actual_factor}×)")
            return False
    
    return True

if __name__ == "__main__":
    print("=== Testing Double Block FAL-kontext Implementation ===\n")
    
    structure_ok = test_structure_analysis()
    conversion_ok = test_conversion_mapping()
    scaling_ok = test_scaling_factors()
    
    if structure_ok and conversion_ok and scaling_ok:
        print("\n🎉 All tests passed! The implementation correctly handles double block architecture differences.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)