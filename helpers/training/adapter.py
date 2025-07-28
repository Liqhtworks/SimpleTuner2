import peft
import torch
import safetensors.torch


def determine_adapter_target_modules(args, unet, transformer):
    if unet is not None:
        return ["to_k", "to_q", "to_v", "to_out.0"]
    elif transformer is not None:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        if args.model_family.lower() == "flux" and args.flux_lora_target == "all":
            # target_modules = mmdit layers here
            target_modules = [
                "to_k",
                "to_q",
                "to_v",
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
            ]
        elif args.flux_lora_target == "context":
            # i think these are the text input layers.
            target_modules = [
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_add_out",
            ]
        elif args.flux_lora_target == "context+ffs":
            # i think these are the text input layers.
            target_modules = [
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_add_out",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ]
        elif args.flux_lora_target == "all+ffs":
            target_modules = [
                "to_k",
                "to_q",
                "to_v",
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
                "proj_mlp",
                "proj_out",
            ]
        elif args.flux_lora_target == "ai-toolkit":
            # from ostris' ai-toolkit, possibly required to continue finetuning one.
            target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
                "norm.linear",
                "norm1.linear",
                "norm1_context.linear",
                "proj_mlp",
                "proj_out",
            ]
        elif args.flux_lora_target == "fal":
            # from fal-ai, possibly required to continue finetuning one.
            target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ]
        elif args.flux_lora_target == "fal-kontext":
            # fal-ai kontext variant with fused QKV projections and modulation layers
            # This targets the same layers as the analyzed LoRA structure:
            # - Fused QKV projections for image/text attention
            # - Modulation linear layers for double/single blocks
            # - Attention output projections
            # - Final projection layer
            target_modules = [
                # For training new models, we need to support both fused and unfused variants
                # Double blocks (MMDiT blocks):
                "attn.to_q",  # Will be fused into to_qkv if fuse_qkv_projections is enabled
                "attn.to_k",
                "attn.to_v",
                "attn.to_qkv",  # Fused QKV projection for image attention
                "attn.add_q_proj",  # Will be fused into add_qkv_proj if enabled
                "attn.add_k_proj",
                "attn.add_v_proj", 
                "attn.add_qkv_proj",  # Fused QKV projection for text attention
                "attn.to_out.0",  # Image attention output projection
                "attn.to_add_out",  # Text attention output projection
                "norm1.linear",  # Image modulation linear (outputs 6x hidden for gate_msa, shift_mlp, scale_mlp, gate_mlp)
                "norm1_context.linear",  # Text modulation linear (outputs 6x hidden)
                # Single blocks (DiT blocks):
                "attn.to_qkv",  # Fused QKV for single blocks (if supported)
                "attn.to_q",  # Unfused variants for compatibility
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",  # Attention output
                "norm.linear",  # Modulation linear (outputs 3x hidden for gate, shift, scale)
                # Global:
                "proj_out",  # Final output projection
            ]
        elif args.flux_lora_target == "fal-kontext-fused":
            # Full FAL kontext fusion with QKV+MLP combined in single blocks
            # and FAL-style aliases for double blocks
            target_modules = [
                # Single blocks - FAL's linear1 (QKV + MLP fused)
                "linear1",  # 7x output (3x QKV + 4x MLP)
                "attn.to_out.0",  # Attention output (if exists)
                "norm.linear",  # Modulation (3x)
                
                # Double blocks - actual module paths (not aliases)
                # Image path
                "attn.to_qkv",  # Image QKV (fused) - FAL's img_attn_qkv
                "attn.to_out.0",  # Image attention output - FAL's img_attn_proj
                "norm1.linear",  # Image modulation (6x) - FAL's img_mod_lin
                "ff.net.0.proj",  # Image FF first layer - FAL's img_mlp_0
                "ff.net.2",  # Image FF second layer - FAL's img_mlp_2
                
                # Text path
                "attn.to_added_qkv",  # Text QKV (fused) - FAL's txt_attn_qkv
                "attn.to_add_out",  # Text attention output - FAL's txt_attn_proj  
                "norm1_context.linear",  # Text modulation (6x) - FAL's txt_mod_lin
                "ff_context.net.0.proj",  # Text FF first layer - FAL's txt_mlp_0
                "ff_context.net.2",  # Text FF second layer - FAL's txt_mlp_2
                
                # Global
                "proj_out",  # Final output projection
            ]
        elif args.flux_lora_target == "daisy":
            # from fal-ai, possibly required to continue finetuning one.
            target_modules = [
                "single_transformer_blocks.9.attn.to_q",
                "single_transformer_blocks.9.attn.to_k",
                "single_transformer_blocks.9.attn.to_v",
                "single_transformer_blocks.12.attn.to_q",
                "single_transformer_blocks.12.attn.to_k",
                "single_transformer_blocks.12.attn.to_v",
                "single_transformer_blocks.16.attn.to_q",
                "single_transformer_blocks.16.attn.to_k",
                "single_transformer_blocks.16.attn.to_v",
                "single_transformer_blocks.20.attn.to_q",
                "single_transformer_blocks.20.attn.to_k",
                "single_transformer_blocks.20.attn.to_v",
                "single_transformer_blocks.25.attn.to_q",
                "single_transformer_blocks.25.attn.to_k",
                "single_transformer_blocks.25.attn.to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ]
        elif args.flux_lora_target == "daisy-tiny":
            # from fal-ai, possibly required to continue finetuning one.
            target_modules = [
                "single_transformer_blocks.9.attn.to_q",
                "single_transformer_blocks.9.attn.to_k",
                "single_transformer_blocks.9.attn.to_v",
                "single_transformer_blocks.20.attn.to_q",
                "single_transformer_blocks.20.attn.to_k",
                "single_transformer_blocks.20.attn.to_v",
                "single_transformer_blocks.25.attn.to_q",
                "single_transformer_blocks.25.attn.to_k",
                "single_transformer_blocks.25.attn.to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ]      
        elif args.flux_lora_target == "tiny":
            # From TheLastBen
            # https://www.reddit.com/r/StableDiffusion/comments/1f523bd/good_flux_loras_can_be_less_than_45mb_128_dim/
            target_modules = [
                "single_transformer_blocks.7.proj_out",
                "single_transformer_blocks.20.proj_out",
            ]
        elif args.flux_lora_target == "nano":
            # From TheLastBen
            # https://www.reddit.com/r/StableDiffusion/comments/1f523bd/good_flux_loras_can_be_less_than_45mb_128_dim/
            target_modules = [
                "single_transformer_blocks.7.proj_out",
            ]

        return target_modules


# Mapping from fal-kontext LoRA naming to SimpleTuner/diffusers naming
FAL_KONTEXT_KEY_MAPPING = {
    # Double blocks (MMDiT) mappings
    "img_attn_qkv": "attn.to_qkv",  # Fused QKV for image attention
    "txt_attn_qkv": "attn.add_qkv_proj",  # Fused QKV for text attention
    "img_attn_proj": "attn.to_out.0",  # Image attention output
    "txt_attn_proj": "attn.to_add_out",  # Text attention output  
    "img_mod_lin": "norm1.linear",  # Image modulation
    "txt_mod_lin": "norm1_context.linear",  # Text modulation
    "img_mlp_0": "ff.net.0.proj",  # Image MLP first layer
    "img_mlp_2": "ff.net.2",  # Image MLP second layer
    "txt_mlp_0": "ff_context.net.0.proj",  # Text MLP first layer
    "txt_mlp_2": "ff_context.net.2",  # Text MLP second layer
    # Single blocks (DiT) mappings
    "linear1": "attn.to_qkv",  # Fused QKV (if the LoRA uses this naming)
    "linear2": "attn.to_out.0",  # Attention output
    "modulation_lin": "norm.linear",  # Single block modulation
    # Global mappings
    "final_layer_linear": "proj_out",  # Final output projection
}

# Scaling factors for specific layer types based on output dimensions
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


def get_fal_kontext_mapped_key(lora_key):
    """Map fal-kontext LoRA keys to SimpleTuner module names."""
    # Remove .lora_A.weight or .lora_B.weight suffix
    base_key = lora_key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
    
    # Extract the components
    parts = base_key.split(".")
    
    # Handle transformer prefix if present
    if parts[0] == "transformer":
        parts = parts[1:]
    
    # Handle lora_unet prefix if present (from the example)
    if parts[0] == "lora_unet":
        parts = parts[1:]
    
    # Map based on the structure
    if len(parts) >= 3:
        block_type = parts[0]  # e.g., "double_blocks", "single_blocks"
        block_num = parts[1]   # e.g., "0", "1", etc.
        layer_name = "_".join(parts[2:])  # e.g., "img_attn_qkv"
        
        # Look up the mapping
        if layer_name in FAL_KONTEXT_KEY_MAPPING:
            mapped_name = FAL_KONTEXT_KEY_MAPPING[layer_name]
            # Reconstruct with proper prefix
            if block_type == "double_blocks":
                return f"transformer.transformer_blocks.{block_num}.{mapped_name}"
            elif block_type == "single_blocks":
                return f"transformer.single_transformer_blocks.{block_num}.{mapped_name}"
    
    # Handle global layers
    if "final_layer" in base_key:
        return "transformer.proj_out"
    
    # Return original if no mapping found
    return base_key


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


@torch.no_grad()
def load_lora_weights(dictionary, filename, loraKey="default", use_dora=False):
    additional_keys = set()
    state_dict = safetensors.torch.load_file(filename)
    
    # Check if this is a fal-kontext style LoRA
    is_fal_kontext = any(
        "img_attn_qkv" in k or "txt_attn_qkv" in k or 
        "img_mlp" in k or "txt_mlp" in k or 
        "modulation_lin" in k or "mod_lin" in k 
        for k in state_dict.keys()
    )
    
    for prefix, model in dictionary.items():
        lora_layers = {
            (prefix + "." + x): y
            for (x, y) in model.named_modules()
            if isinstance(y, peft.tuners.lora.layer.Linear)
        }
    missing_keys = set(
        [x + ".lora_A.weight" for x in lora_layers.keys()]
        + [x + ".lora_B.weight" for x in lora_layers.keys()]
        + (
            [x + ".lora_magnitude_vector.weight" for x in lora_layers.keys()]
            if use_dora
            else []
        )
    )
    
    # b_up_factor for certain B weights (e.g. in ff and ff_context blocks)
    b_up_factor = 3

    for k, v in state_dict.items():
        # Map the key if it's a fal-kontext LoRA
        if is_fal_kontext:
            mapped_k = get_fal_kontext_mapped_key(k)
        else:
            mapped_k = k
            
        if "lora_A" in k:
            kk = mapped_k.replace(".lora_A.weight", "")
            if kk in lora_layers:
                lora_layers[kk].lora_A[loraKey].weight.copy_(v)
                if k in missing_keys:
                    missing_keys.remove(k)
            else:
                additional_keys.add(k)
        elif "lora_B" in k:
            kk = mapped_k.replace(".lora_B.weight", "")
            if kk in lora_layers:
                # Get appropriate scaling factor
                if is_fal_kontext:
                    scaling_factor = get_fal_kontext_scaling_factor(kk)
                    scaled_v = v * scaling_factor
                elif ("ff.net." in kk) or ("ff_context.net." in kk):
                    # Original FF scaling logic
                    scaled_v = v * b_up_factor
                else:
                    scaled_v = v
                    
                lora_layers[kk].lora_B[loraKey].weight.copy_(scaled_v)
                if k in missing_keys:
                    missing_keys.remove(k)
            else:
                additional_keys.add(k)
        elif ".alpha" in k or ".lora_alpha" in k:
            kk = mapped_k.replace(".lora_alpha", "").replace(".alpha", "")
            if kk in lora_layers:
                lora_layers[kk].lora_alpha[loraKey] = v
        elif ".lora_magnitude_vector" in k:
            kk = mapped_k.replace(".lora_magnitude_vector.weight", "")
            if kk in lora_layers:
                lora_layers[kk].lora_magnitude_vector[loraKey].weight.copy_(v)
                if k in missing_keys:
                    missing_keys.remove(k)
            else:
                additional_keys.add(k)
                
    return (additional_keys, missing_keys)
