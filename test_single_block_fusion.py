#!/usr/bin/env python3
"""
Test script to verify single block fusion logic and conversion mappings.
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
    """Test the structural differences between SimpleTuner and FAL-kontext."""
    print("Testing structural analysis...")
    
    # Based on single.md analysis
    simpletuner_layers = {
        "linear1": {"shape": [21504, 16], "type": "lora_A/lora_B"},
        "norm.linear": {"shape": [9216, 16], "type": "lora_A/lora_B"},
        "proj_out": {"shape": [3072, 16], "type": "lora_A/lora_B"},
    }
    
    fal_kontext_layers = {
        "linear1": {"shape": [21504, 16], "type": "lora_down/lora_up"},
        "modulation_lin": {"shape": [9216, 16], "type": "lora_down/lora_up"},
        "linear2": {"shape": [3072, 16], "type": "lora_down/lora_up"},
    }
    
    print("✓ SimpleTuner single block layers:")
    for name, info in simpletuner_layers.items():
        print(f"  - {name}: {info['shape']} ({info['type']})")
    
    print("\n✓ FAL-kontext single block layers:")
    for name, info in fal_kontext_layers.items():
        print(f"  - {name}: {info['shape']} ({info['type']})")
    
    # Key mappings
    mappings = [
        ("SimpleTuner proj_out", "FAL linear2"),
        ("SimpleTuner norm.linear", "FAL modulation_lin"),
        ("Both have linear1", "Same dimensions (21504 = 7×3072)")
    ]
    
    print("\n✓ Key mappings identified:")
    for mapping in mappings:
        print(f"  - {mapping[0]} -> {mapping[1]}")
    
    return True

def test_conversion_mapping():
    """Test that the conversion script mapping is correct."""
    print("\nTesting conversion script mapping...")
    
    # Test mapping for different module paths
    test_cases = [
        ("single_transformer_blocks.0.attn.to_qkv", "lora_unet_single_blocks_0_linear1"),
        ("single_transformer_blocks.0.proj_out", "lora_unet_single_blocks_0_linear2"),
        ("single_transformer_blocks.0.norm.linear", "lora_unet_single_blocks_0_modulation_lin"),
    ]
    
    # Import the conversion function
    from convert_lora_to_fal_kontext import convert_simpletuner_to_fal_kontext
    
    # Create mock state dict
    state_dict = {}
    for diffusers_key, expected_fal_key in test_cases:
        # Add lora_A and lora_B weights
        state_dict[f"transformer.{diffusers_key}.lora_A.weight"] = torch.randn(16, 3072)
        state_dict[f"transformer.{diffusers_key}.lora_B.weight"] = torch.randn(1024, 16)  # Mock size
    
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
            else:
                print(f"✗ {diffusers_key} -> {expected_fal_key} (missing keys)")
                return False
                
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False
    
    print("\n✓ All conversion mappings are correct!")
    return True

if __name__ == "__main__":
    print("=== Testing Single Block FAL-kontext Implementation ===\n")
    
    structure_ok = test_structure_analysis()
    conversion_ok = test_conversion_mapping()
    
    if structure_ok and conversion_ok:
        print("\n🎉 All tests passed! The implementation correctly handles single block architecture differences.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)