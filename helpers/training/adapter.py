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
                "attn.to_out.0",  # Attention output
                "norm.linear",  # Modulation (3x)
                
                # Double blocks - FAL-style aliases
                "img_attn_qkv",  # Image QKV (fused)
                "txt_attn_qkv",  # Text QKV (fused)
                "img_attn_proj",  # Image attention output
                "txt_attn_proj",  # Text attention output
                "img_mod_lin",  # Image modulation (6x)
                "txt_mod_lin",  # Text modulation (6x)
                "img_mlp_0",  # Image FF first layer
                "img_mlp_2",  # Image FF second layer
                "txt_mlp_0",  # Text FF first layer
                "txt_mlp_2",  # Text FF second layer
                
                # Global
                "proj_out",  # Final output projection
                
                # Fallback for partially fused models
                "attn.to_qkv",  # Standard fused QKV
                "attn.add_qkv_proj",  # Standard fused text QKV
                "attn.to_out.0",
                "attn.to_add_out",
                "norm1.linear",
                "norm1_context.linear",
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
