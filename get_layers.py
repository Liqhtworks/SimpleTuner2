import safetensors.torch
state_dict = safetensors.torch.load_file("/Users/gt/Downloads/labubu_fal.safetensors")
unique_layers = set()
for k in state_dict.keys():
    # Remove .lora_A.weight or .lora_B.weight
    base = k.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
    unique_layers.add(base)
print(sorted(unique_layers))
