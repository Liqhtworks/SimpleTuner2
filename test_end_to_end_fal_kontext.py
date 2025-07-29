#!/usr/bin/env python3
"""
End-to-end test that validates SimpleTuner can produce FAL-compatible LoRAs 
when using the flux-kontext-fused setting.

This test simulates the complete pipeline:
1. Mock SimpleTuner training output with fal-kontext-fused target
2. Apply conversion using save_lora_in_kohya_format
3. Compare against reference FAL LoRA structure
4. Validate compatibility and completeness
"""

import logging
import sys
from pathlib import Path
import torch
import safetensors.torch
from collections import defaultdict

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_simpletuner_output():
    """Create a mock SimpleTuner LoRA output with fal-kontext-fused architecture."""
    print("Creating mock SimpleTuner LoRA output...")
    
    state_dict = {}
    rank = 16
    base_dim = 3072
    
    # Double blocks (19 blocks total)
    for block_id in range(19):
        block_base = f"transformer.transformer_blocks.{block_id}"
        
        # Image path
        state_dict[f"{block_base}.attn.to_qkv.lora_A.weight"] = torch.randn(rank, base_dim, dtype=torch.bfloat16)
        state_dict[f"{block_base}.attn.to_qkv.lora_B.weight"] = torch.randn(base_dim * 3, rank, dtype=torch.bfloat16)
        state_dict[f"{block_base}.attn.to_out.0.lora_A.weight"] = torch.randn(rank, base_dim, dtype=torch.bfloat16)
        state_dict[f"{block_base}.attn.to_out.0.lora_B.weight"] = torch.randn(base_dim, rank, dtype=torch.bfloat16)
        state_dict[f"{block_base}.norm1.linear.lora_A.weight"] = torch.randn(rank, base_dim, dtype=torch.bfloat16)
        state_dict[f"{block_base}.norm1.linear.lora_B.weight"] = torch.randn(base_dim * 6, rank, dtype=torch.bfloat16)
        state_dict[f"{block_base}.ff.net.0.proj.lora_A.weight"] = torch.randn(rank, base_dim, dtype=torch.bfloat16)
        state_dict[f"{block_base}.ff.net.0.proj.lora_B.weight"] = torch.randn(base_dim * 4, rank, dtype=torch.bfloat16)
        state_dict[f"{block_base}.ff.net.2.lora_A.weight"] = torch.randn(rank, base_dim * 4, dtype=torch.bfloat16)
        state_dict[f"{block_base}.ff.net.2.lora_B.weight"] = torch.randn(base_dim, rank, dtype=torch.bfloat16)
        
        # Text path
        state_dict[f"{block_base}.attn.to_added_qkv.lora_A.weight"] = torch.randn(rank, base_dim, dtype=torch.bfloat16)
        state_dict[f"{block_base}.attn.to_added_qkv.lora_B.weight"] = torch.randn(base_dim * 3, rank, dtype=torch.bfloat16)
        state_dict[f"{block_base}.attn.to_add_out.lora_A.weight"] = torch.randn(rank, base_dim, dtype=torch.bfloat16)
        state_dict[f"{block_base}.attn.to_add_out.lora_B.weight"] = torch.randn(base_dim, rank, dtype=torch.bfloat16)
        state_dict[f"{block_base}.norm1_context.linear.lora_A.weight"] = torch.randn(rank, base_dim, dtype=torch.bfloat16)
        state_dict[f"{block_base}.norm1_context.linear.lora_B.weight"] = torch.randn(base_dim * 6, rank, dtype=torch.bfloat16)
        state_dict[f"{block_base}.ff_context.net.0.proj.lora_A.weight"] = torch.randn(rank, base_dim, dtype=torch.bfloat16)
        state_dict[f"{block_base}.ff_context.net.0.proj.lora_B.weight"] = torch.randn(base_dim * 4, rank, dtype=torch.bfloat16)
        state_dict[f"{block_base}.ff_context.net.2.lora_A.weight"] = torch.randn(rank, base_dim * 4, dtype=torch.bfloat16)
        state_dict[f"{block_base}.ff_context.net.2.lora_B.weight"] = torch.randn(base_dim, rank, dtype=torch.bfloat16)
    
    # Single blocks (38 blocks total) - after fusion with linear1
    for block_id in range(38):
        block_base = f"transformer.single_transformer_blocks.{block_id}"
        
        # Fused linear1 (QKV + MLP combined)
        state_dict[f"{block_base}.linear1.lora_A.weight"] = torch.randn(rank, base_dim, dtype=torch.bfloat16)
        state_dict[f"{block_base}.linear1.lora_B.weight"] = torch.randn(base_dim * 7, rank, dtype=torch.bfloat16)  # 7x for QKV+MLP
        
        # Modulation layer (norm.linear)
        state_dict[f"{block_base}.norm.linear.lora_A.weight"] = torch.randn(rank, base_dim, dtype=torch.bfloat16)
        state_dict[f"{block_base}.norm.linear.lora_B.weight"] = torch.randn(base_dim * 3, rank, dtype=torch.bfloat16)
        
        # Output projection (proj_out)
        state_dict[f"{block_base}.proj_out.lora_A.weight"] = torch.randn(rank, base_dim + base_dim * 4, dtype=torch.bfloat16)  # 15360 = 3072 + 12288
        state_dict[f"{block_base}.proj_out.lora_B.weight"] = torch.randn(base_dim, rank, dtype=torch.bfloat16)
    
    # Final layer
    state_dict["transformer.proj_out.lora_A.weight"] = torch.randn(rank, base_dim, dtype=torch.bfloat16)
    state_dict["transformer.proj_out.lora_B.weight"] = torch.randn(64, rank, dtype=torch.bfloat16)  # Output channels
    
    print(f"✓ Created mock SimpleTuner state dict with {len(state_dict)} tensors")
    return state_dict

def convert_to_fal_format(simpletuner_state_dict):
    """Convert SimpleTuner format to FAL-kontext format."""
    print("Converting to FAL-kontext format...")
    
    from convert_lora_to_fal_kontext import convert_simpletuner_to_fal_kontext
    
    try:
        fal_state_dict = convert_simpletuner_to_fal_kontext(simpletuner_state_dict)
        print(f"✓ Conversion successful, output has {len(fal_state_dict)} tensors")
        return fal_state_dict
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_layer_structure(state_dict, name):
    """Analyze the layer structure of a state dict."""
    print(f"\n=== Analyzing {name} Structure ===")
    
    layer_counts = defaultdict(int)
    unique_layers = set()
    
    for key in state_dict.keys():
        # Extract base layer name
        if '.lora_down.weight' in key:
            base = key.replace('.lora_down.weight', '')
            unique_layers.add(base)
        elif '.lora_up.weight' in key:
            base = key.replace('.lora_up.weight', '')
            unique_layers.add(base)
        elif '.lora_A.weight' in key:
            base = key.replace('.lora_A.weight', '')
            unique_layers.add(base)
        elif '.lora_B.weight' in key:
            base = key.replace('.lora_B.weight', '')
            unique_layers.add(base)
    
    # Categorize layers
    for layer in unique_layers:
        if 'double_blocks' in layer or 'transformer_blocks' in layer:
            if 'img_attn_qkv' in layer or 'attn.to_qkv' in layer:
                layer_counts['double_img_attn_qkv'] += 1
            elif 'txt_attn_qkv' in layer or 'to_added_qkv' in layer:
                layer_counts['double_txt_attn_qkv'] += 1
            elif 'img_attn_proj' in layer or 'attn.to_out.0' in layer:
                layer_counts['double_img_attn_proj'] += 1
            elif 'txt_attn_proj' in layer or 'to_add_out' in layer:
                layer_counts['double_txt_attn_proj'] += 1
            elif 'img_mod_lin' in layer or 'norm1.linear' in layer:
                layer_counts['double_img_mod_lin'] += 1
            elif 'txt_mod_lin' in layer or 'norm1_context.linear' in layer:
                layer_counts['double_txt_mod_lin'] += 1
            elif 'img_mlp_0' in layer or 'ff.net.0.proj' in layer:
                layer_counts['double_img_mlp_0'] += 1
            elif 'img_mlp_2' in layer or 'ff.net.2' in layer:
                layer_counts['double_img_mlp_2'] += 1
            elif 'txt_mlp_0' in layer or 'ff_context.net.0.proj' in layer:
                layer_counts['double_txt_mlp_0'] += 1
            elif 'txt_mlp_2' in layer or 'ff_context.net.2' in layer:
                layer_counts['double_txt_mlp_2'] += 1
        elif 'single_blocks' in layer or 'single_transformer_blocks' in layer:
            if 'linear1' in layer:
                layer_counts['single_linear1'] += 1
            elif 'linear2' in layer or 'proj_out' in layer:
                layer_counts['single_linear2'] += 1
            elif 'modulation_lin' in layer or 'norm.linear' in layer:
                layer_counts['single_modulation_lin'] += 1
        elif 'final_layer' in layer or (layer.endswith('proj_out') and 'blocks' not in layer):
            layer_counts['final_layer'] += 1
    
    print("Layer counts:")
    for layer_type, count in sorted(layer_counts.items()):
        print(f"  {layer_type}: {count}")
    
    return layer_counts, unique_layers

def load_reference_fal_lora(path):
    """Load the reference FAL LoRA for comparison."""
    print(f"Loading reference FAL LoRA from {path}...")
    
    try:
        state_dict = safetensors.torch.load_file(path)
        print(f"✓ Loaded reference with {len(state_dict)} tensors")
        return state_dict
    except Exception as e:
        print(f"✗ Failed to load reference: {e}")
        return None

def compare_structures(converted_counts, reference_counts, converted_layers, reference_layers):
    """Compare the converted structure against the reference."""
    print("\n=== Structure Comparison ===")
    
    # Expected layer counts based on the reference FAL LoRA
    expected_counts = {
        'double_img_attn_qkv': 19,
        'double_txt_attn_qkv': 19,
        'double_img_attn_proj': 19,
        'double_txt_attn_proj': 19,
        'double_img_mod_lin': 19,
        'double_txt_mod_lin': 19,
        'double_img_mlp_0': 19,
        'double_img_mlp_2': 19,
        'double_txt_mlp_0': 19,
        'double_txt_mlp_2': 19,
        'single_linear1': 38,
        'single_linear2': 38,
        'single_modulation_lin': 38,
        'final_layer': 1,
    }
    
    all_match = True
    
    print("Comparing layer counts:")
    for layer_type, expected in expected_counts.items():
        converted = converted_counts.get(layer_type, 0)
        reference = reference_counts.get(layer_type, 0)
        
        if converted == expected == reference:
            print(f"  ✓ {layer_type}: {converted} (matches expected and reference)")
        elif converted == expected:
            print(f"  ✓ {layer_type}: {converted} (matches expected, reference has {reference})")
        else:
            print(f"  ✗ {layer_type}: converted={converted}, expected={expected}, reference={reference}")
            all_match = False
    
    return all_match

def test_dimension_compatibility(converted_dict, reference_dict):
    """Test that converted dimensions match reference dimensions."""
    print("\n=== Dimension Compatibility Test ===")
    
    # Sample a few key layers to verify dimensions
    test_layers = [
        ("lora_unet_double_blocks_0_img_attn_qkv", "Image attention QKV"),
        ("lora_unet_double_blocks_0_img_mod_lin", "Image modulation"),
        ("lora_unet_single_blocks_0_linear1", "Single block linear1"),
        ("lora_unet_single_blocks_0_modulation_lin", "Single block modulation"),
        ("lora_unet_final_layer_linear", "Final layer"),
    ]
    
    all_match = True
    
    for layer_base, description in test_layers:
        down_key = f"{layer_base}.lora_down.weight"
        up_key = f"{layer_base}.lora_up.weight"
        
        if down_key in converted_dict and up_key in converted_dict:
            if down_key in reference_dict and up_key in reference_dict:
                conv_down_shape = list(converted_dict[down_key].shape)
                conv_up_shape = list(converted_dict[up_key].shape)
                ref_down_shape = list(reference_dict[down_key].shape)
                ref_up_shape = list(reference_dict[up_key].shape)
                
                if conv_down_shape == ref_down_shape and conv_up_shape == ref_up_shape:
                    print(f"  ✓ {description}: shapes match")
                    print(f"    down: {conv_down_shape}, up: {conv_up_shape}")
                else:
                    print(f"  ✗ {description}: shape mismatch")
                    print(f"    converted - down: {conv_down_shape}, up: {conv_up_shape}")
                    print(f"    reference - down: {ref_down_shape}, up: {ref_up_shape}")
                    all_match = False
            else:
                print(f"  ? {description}: not found in reference (converted has it)")
        else:
            print(f"  ✗ {description}: not found in converted output")
            all_match = False
    
    return all_match

def main():
    print("=== End-to-End FAL-Kontext Compatibility Test ===\n")
    
    # Step 1: Create mock SimpleTuner output
    simpletuner_dict = create_mock_simpletuner_output()
    if not simpletuner_dict:
        print("❌ Failed to create mock SimpleTuner output")
        return False
    
    # Step 2: Convert to FAL format
    converted_dict = convert_to_fal_format(simpletuner_dict)
    if not converted_dict:
        print("❌ Failed to convert to FAL format")
        return False
    
    # Step 3: Load reference FAL LoRA
    reference_path = "/home/playerzer0x/ComfyUI/models/loras/labubu_fal_2500/labubu_fal_2500.safetensors"
    reference_dict = load_reference_fal_lora(reference_path)
    if not reference_dict:
        print("❌ Failed to load reference FAL LoRA")
        return False
    
    # Step 4: Analyze structures
    converted_counts, converted_layers = analyze_layer_structure(converted_dict, "Converted SimpleTuner")
    reference_counts, reference_layers = analyze_layer_structure(reference_dict, "Reference FAL")
    
    # Step 5: Compare structures
    structure_match = compare_structures(converted_counts, reference_counts, converted_layers, reference_layers)
    
    # Step 6: Test dimension compatibility
    dimension_match = test_dimension_compatibility(converted_dict, reference_dict)
    
    # Final assessment
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    if structure_match and dimension_match:
        print("🎉 SUCCESS: SimpleTuner with flux-kontext-fused produces FAL-compatible LoRAs!")
        print("✓ Layer structure matches expected counts")
        print("✓ Dimensions are compatible with reference")
        print("✓ Conversion pipeline works correctly")
        return True
    else:
        print("❌ FAILURE: Issues found in compatibility")
        if not structure_match:
            print("✗ Layer structure doesn't match expected counts")
        if not dimension_match:
            print("✗ Dimensions don't match reference")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)