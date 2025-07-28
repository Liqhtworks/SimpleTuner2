# LoRA Architecture Mapping Chart

## SimpleTuner ↔ FAL-Kontext Layer Mapping

Based on analysis of double.md and single.md files.

### Double Blocks (MMDiT)

| SimpleTuner Layer | FAL-Kontext Layer | lora_A → lora_down | lora_B → lora_up | Purpose |
|-------------------|-------------------|-------------------|------------------|---------|
| `attn.to_qkv` | `img_attn_qkv` | [16, 3072] | [9216, 16] | Image attention QKV (3×3072) |
| `attn.to_out.0` | `img_attn_proj` | [16, 3072] | [3072, 16] | Image attention output |
| `attn.to_added_qkv` | `txt_attn_qkv` | [16, 3072] | [9216, 16] | Text attention QKV (3×3072) |
| `attn.to_add_out` | `txt_attn_proj` | [16, 3072] | [3072, 16] | Text attention output |
| `norm1.linear` | `img_mod_lin` | [16, 3072] | [18432, 16] | Image modulation (6×3072) |
| `norm1_context.linear` | `txt_mod_lin` | [16, 3072] | [18432, 16] | Text modulation (6×3072) |
| `ff.net.0.proj` | `img_mlp_0` | [16, 3072] | [12288, 16] | Image MLP input (4×3072) |
| `ff.net.2` | `img_mlp_2` | [16, 12288] | [3072, 16] | Image MLP output |
| `ff_context.net.0.proj` | `txt_mlp_0` | [16, 3072] | [12288, 16] | Text MLP input (4×3072) |
| `ff_context.net.2` | `txt_mlp_2` | [16, 12288] | [3072, 16] | Text MLP output |

### Single Blocks (DiT)

| SimpleTuner Layer | FAL-Kontext Layer | lora_A → lora_down | lora_B → lora_up | Purpose |
|-------------------|-------------------|-------------------|------------------|---------|
| `linear1` | `linear1` | [16, 3072] | [21504, 16] | Fused QKV+MLP (7×3072) |
| `norm.linear` | `modulation_lin` | [16, 3072] | [9216, 16] | Single modulation (3×3072) |
| `proj_out` | `linear2` | [16, 15360] | [3072, 16] | Output projection |

### Global Layers

| SimpleTuner Layer | FAL-Kontext Layer | Purpose |
|-------------------|-------------------|---------|
| `proj_out` | `final_layer_linear` | Final model output |

## Key Mapping Rules

### Weight Type Conversion
```
SimpleTuner → FAL-Kontext
lora_A      → lora_down    (rank bottleneck - smaller dimension)
lora_B      → lora_up      (expansion - larger dimension)
.alpha      → .alpha       (scaling factor - unchanged)
```

### Dimensional Analysis
- **Rank**: Always 16 in these examples
- **Base dimension**: 3072 (model hidden size)
- **Scaling factors**:
  - QKV: 3× → 9216
  - Double modulation: 6× → 18432  
  - Single modulation: 3× → 9216
  - MLP expansion: 4× → 12288
  - Single linear1: 7× → 21504

### Data Type Conversion
- **SimpleTuner**: BF16 (bfloat16)
- **FAL-Kontext**: F32 (float32)

## Verification Status

✅ **All mappings verified** - Dimensions match perfectly between SimpleTuner and FAL-Kontext formats
✅ **A/B → down/up mapping correct** - Follows standard LoRA convention
✅ **Complete architecture coverage** - All required layers present
✅ **Scaling factors accurate** - Mathematical relationships preserved