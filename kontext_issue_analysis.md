# Kontext LoRA Training Issue Analysis

## Summary
The Kontext LoRA training started experiencing issues after commit `e2898c94372c83785f108941de412c0ac21be63b` (dated Sun Jun 29 19:31:23 2025). This analysis documents the key changes that likely caused the regression.

## Problematic Commit Details
- **Hash**: e2898c94372c83785f108941de412c0ac21be63b  
- **Author**: Yael Walker
- **Date**: Sun Jun 29 19:31:23 2025 +0200
- **Message**: "initial changes (to be tested tomorrow)"

## Key Changes Identified

### 1. **Critical Change: `build_kontext_inputs` Function**
The most significant change is in `helpers/models/flux/__init__.py`:

#### Before (Working Version):
```python
def build_kontext_inputs(
    cond_latents: torch.Tensor,  # Single tensor
    dtype: torch.dtype,
    device: torch.device,
    latent_channels: int,
):
    B, C, H, W = cond_latents.shape
    packed_cond = pack_latents(cond_latents, B, C, H, W).to(device=device, dtype=dtype)
    
    # Simple ID generation
    idx_y = torch.arange(H // 2, device=device)
    idx_x = torch.arange(W // 2, device=device)
    ids = torch.stack(torch.meshgrid(idx_y, idx_x, indexing="ij"), dim=-1)
    ones = torch.ones_like(ids[..., :1])
    ids = torch.cat([ones, ids], dim=-1).view(1, -1, 3).expand(B, -1, -1).to(dtype)
    
    return packed_cond, ids
```

#### After (Problematic Version):
```python
def build_kontext_inputs(
    cond_latents: list[torch.Tensor],  # Now expects a LIST
    dtype: torch.dtype,
    device: torch.device,
    # latent_channels parameter removed
):
    packed_cond = []
    packed_ids = []
    
    # Complex coordinate offsetting algorithm
    x0 = 0
    y0 = 0
    for latent in cond_latents:
        B, C, H, W = latent.shape
        packed_cond.append(pack_latents(latent, B, C, H, W).to(device=device, dtype=dtype))
        
        # New offsetting logic
        x = 0
        y = 0
        if H + y0 > W + x0:
            x = x0
        else:
            y = y0
        
        # Modified ID generation with offsets
        idx_y = torch.arange(H // 2, device=device) + y//2
        idx_x = torch.arange(W // 2, device=device) + x//2
        ids = torch.stack(torch.meshgrid(idx_y, idx_x, indexing="ij"), dim=-1)
        ones = torch.ones_like(ids[..., :1])
        ids = torch.cat([ones, ids], dim=-1).view(1, -1, 3).expand(B, -1, -1).to(dtype)
        
        x0 = max(x0, W + x)
        y0 = max(y0, H + y)
    
    packed_cond = torch.cat(packed_cond, dim=1)
    packed_ids = torch.cat(packed_ids, dim=1)
    
    return packed_cond, packed_ids
```

### 2. **Conditioning Mappings Changes**

The commit also introduced changes to how conditioning datasets are mapped:

- Changed from `StateTracker.get_conditioning_mappings().items()` to just `.values()` in some places
- Modified the conditioning mapping structure to support multiple conditioning datasets per training dataset
- Changed `set_conditioning_dataset` to `set_conditioning_datasets` (plural)

### 3. **Collate Function Changes**

In `helpers/training/collate.py`:
- Added support for conditioning inputs to be lists
- Modified how conditioning latents and pixel values are prepared
- Added new list wrapping/unwrapping logic

### 4. **Common Model Changes**

In `helpers/models/common.py`:
- Added new `prepare_batch_conditions` method that unwraps lists:
```python
def prepare_batch_conditions(self, batch: dict, state: dict) -> dict:
    # Expects lists but most models want single tensors
    if batch.get("conditioning_pixel_values") is not None:
        batch["conditioning_pixel_values"] = batch["conditioning_pixel_values"][0]
    if batch.get("conditioning_latents") is not None:
        batch["conditioning_latents"] = batch["conditioning_latents"][0]
    return batch
```

## Root Cause Analysis

The main issue appears to be a **type mismatch** and **incompatible ID generation**:

1. **Type Mismatch**: The new code expects conditioning latents as a list of tensors, but the calling code in `flux/model.py` still passes a single tensor
2. **ID Generation**: The new coordinate offsetting algorithm is designed for multi-image scenarios but breaks the simple single-image case
3. **Missing Validation**: The code comment mentions "to be tested tomorrow" - this appears to be untested code that was committed

## Impact

The changes were likely intended to support multi-image conditioning (as mentioned in the comments about ComfyUI compatibility), but they break the existing single-image Kontext training workflow by:
1. Changing the expected input types without updating all callers
2. Introducing complex coordinate offsetting that may produce incorrect IDs for single images
3. Breaking the established ID generation pattern that Kontext models expect

## Recommended Fix

To fix the issue while maintaining backward compatibility:

1. **Option A (Quick Fix)**: Revert the `build_kontext_inputs` function to the previous version
2. **Option B (Proper Fix)**: Add logic to detect single vs. multi-image cases and handle both:
   - If input is a single tensor, use the old logic
   - If input is a list, use the new multi-image logic
   - Ensure all calling code properly specifies which case it needs

## Testing Branch

The code has been rolled back to commit `cdb04973` (the commit before the problematic one) on branch `test-pre-problematic-commit` for testing whether this resolves the Kontext training issues.

## Next Steps

1. Test Kontext LoRA training on the rolled-back version to confirm this is the issue
2. If confirmed, either:
   - Cherry-pick other beneficial changes while keeping the old `build_kontext_inputs`
   - Implement a proper fix that handles both single and multi-image cases
3. Add tests to prevent similar regressions in the future