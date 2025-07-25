import torch
import torch.nn as nn
from diffusers.models.attention import Attention
import logging

logger = logging.getLogger(__name__)

# Configuration flag - set this based on your needs
PERMANENT_FUSION = True  # Set to False if you need unfuse capability


@torch.no_grad()
def fuse_projections_smart(self, fuse=True, permanent=None):
    """
    Fuse QKV projections with option for permanent (delete originals) or reversible fusion.

    Args:
        fuse: Whether to fuse (always True for compatibility)
        permanent: Override for PERMANENT_FUSION setting. If None, uses global setting.
    """
    if self.fused_projections:
        return  # Already fused

    # Determine if this should be permanent
    is_permanent = PERMANENT_FUSION if permanent is None else permanent

    device = self.to_q.weight.data.device
    dtype = self.to_q.weight.data.dtype

    if not self.is_cross_attention:
        # Fuse Q, K, V for self-attention
        concatenated_weights = torch.cat(
            [self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data]
        )
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]

        # Create fused layer
        self.to_qkv = nn.Linear(
            in_features, out_features, bias=self.use_bias, device=device, dtype=dtype
        )
        self.to_qkv.weight.copy_(concatenated_weights)

        if self.use_bias:
            concatenated_bias = torch.cat(
                [self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data]
            )
            self.to_qkv.bias.copy_(concatenated_bias)

        if is_permanent:
            # DELETE the original layers
            del self.to_q
            del self.to_k
            del self.to_v

            # Remove from _modules to ensure they're not accessible
            if "to_q" in self._modules:
                del self._modules["to_q"]
            if "to_k" in self._modules:
                del self._modules["to_k"]
            if "to_v" in self._modules:
                del self._modules["to_v"]

    else:
        # For cross-attention, keep to_q separate, only fuse k,v
        concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]

        self.to_kv = nn.Linear(
            in_features, out_features, bias=self.use_bias, device=device, dtype=dtype
        )
        self.to_kv.weight.copy_(concatenated_weights)

        if self.use_bias:
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            self.to_kv.bias.copy_(concatenated_bias)

        if is_permanent:
            # DELETE the original k,v layers
            del self.to_k
            del self.to_v

            if "to_k" in self._modules:
                del self._modules["to_k"]
            if "to_v" in self._modules:
                del self._modules["to_v"]

    # Handle added projections for SD3 and others
    if (
        getattr(self, "add_q_proj", None) is not None
        and getattr(self, "add_k_proj", None) is not None
        and getattr(self, "add_v_proj", None) is not None
    ):
        concatenated_weights = torch.cat(
            [
                self.add_q_proj.weight.data,
                self.add_k_proj.weight.data,
                self.add_v_proj.weight.data,
            ]
        )
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]

        self.to_added_qkv = nn.Linear(
            in_features,
            out_features,
            bias=self.added_proj_bias,
            device=device,
            dtype=dtype,
        )
        self.to_added_qkv.weight.copy_(concatenated_weights)

        if self.added_proj_bias:
            concatenated_bias = torch.cat(
                [
                    self.add_q_proj.bias.data,
                    self.add_k_proj.bias.data,
                    self.add_v_proj.bias.data,
                ]
            )
            self.to_added_qkv.bias.copy_(concatenated_bias)

        if is_permanent:
            # DELETE the original added projection layers
            del self.add_q_proj
            del self.add_k_proj
            del self.add_v_proj

            if "add_q_proj" in self._modules:
                del self._modules["add_q_proj"]
            if "add_k_proj" in self._modules:
                del self._modules["add_k_proj"]
            if "add_v_proj" in self._modules:
                del self._modules["add_v_proj"]

    self.fused_projections = True
    fusion_type = "permanent" if is_permanent else "reversible"
    logger.debug(f"Fused projections for {self.__class__.__name__} ({fusion_type})")


@torch.no_grad()
def unfuse_projections_smart(self):
    """
    Unfuse the QKV projections back to their individual components.
    Will warn and return if fusion was permanent.
    """
    if not self.fused_projections:
        logger.debug("Projections are not fused, nothing to unfuse")
        return

    # Check if layers were deleted (permanent fusion)
    if not hasattr(self, "to_q") and hasattr(self, "to_qkv"):
        logger.warning(
            "Cannot unfuse projections - original layers were deleted during permanent fusion! "
            "Set PERMANENT_FUSION=False or use fuse_projections(permanent=False) for reversible fusion."
        )
        return

    logger.debug(f"Unfusing projections for {self.__class__.__name__}")

    # Handle self-attention unfusing
    if hasattr(self, "to_qkv"):
        # Get device and dtype from fused layer
        device = self.to_qkv.weight.device
        dtype = self.to_qkv.weight.dtype

        # Get the concatenated weights and bias
        concatenated_weights = self.to_qkv.weight.data

        # Calculate dimensions
        total_dim = concatenated_weights.shape[0]
        q_dim = self.inner_dim
        k_dim = self.inner_kv_dim
        v_dim = self.inner_kv_dim

        # Verify dimensions
        assert (
            total_dim == q_dim + k_dim + v_dim
        ), f"Dimension mismatch: {total_dim} != {q_dim} + {k_dim} + {v_dim}"

        # Split the weights
        q_weight = concatenated_weights[:q_dim]
        k_weight = concatenated_weights[q_dim : q_dim + k_dim]
        v_weight = concatenated_weights[q_dim + k_dim :]

        # Create individual linear layers
        self.to_q = nn.Linear(
            self.query_dim, q_dim, bias=self.use_bias, device=device, dtype=dtype
        )
        self.to_k = nn.Linear(
            self.cross_attention_dim,
            k_dim,
            bias=self.use_bias,
            device=device,
            dtype=dtype,
        )
        self.to_v = nn.Linear(
            self.cross_attention_dim,
            v_dim,
            bias=self.use_bias,
            device=device,
            dtype=dtype,
        )

        # Copy weights
        self.to_q.weight.data.copy_(q_weight)
        self.to_k.weight.data.copy_(k_weight)
        self.to_v.weight.data.copy_(v_weight)

        # Handle biases if they exist
        if (
            self.use_bias
            and hasattr(self.to_qkv, "bias")
            and self.to_qkv.bias is not None
        ):
            concatenated_bias = self.to_qkv.bias.data
            q_bias = concatenated_bias[:q_dim]
            k_bias = concatenated_bias[q_dim : q_dim + k_dim]
            v_bias = concatenated_bias[q_dim + k_dim :]

            self.to_q.bias.data.copy_(q_bias)
            self.to_k.bias.data.copy_(k_bias)
            self.to_v.bias.data.copy_(v_bias)

        # Remove the fused layer
        del self.to_qkv
        if "to_qkv" in self._modules:
            del self._modules["to_qkv"]

        logger.debug("Unfused to_qkv -> to_q, to_k, to_v")

    # Handle cross-attention unfusing (fused K,V only)
    elif hasattr(self, "to_kv"):
        # Get device and dtype
        device = self.to_kv.weight.device
        dtype = self.to_kv.weight.dtype

        # Get concatenated weights
        concatenated_weights = self.to_kv.weight.data

        # Calculate dimensions
        total_dim = concatenated_weights.shape[0]
        k_dim = self.inner_kv_dim
        v_dim = self.inner_kv_dim

        assert (
            total_dim == k_dim + v_dim
        ), f"Dimension mismatch for KV: {total_dim} != {k_dim} + {v_dim}"

        # Split weights
        k_weight = concatenated_weights[:k_dim]
        v_weight = concatenated_weights[k_dim:]

        # Create individual layers
        self.to_k = nn.Linear(
            self.cross_attention_dim,
            k_dim,
            bias=self.use_bias,
            device=device,
            dtype=dtype,
        )
        self.to_v = nn.Linear(
            self.cross_attention_dim,
            v_dim,
            bias=self.use_bias,
            device=device,
            dtype=dtype,
        )

        # Copy weights
        self.to_k.weight.data.copy_(k_weight)
        self.to_v.weight.data.copy_(v_weight)

        # Handle biases
        if (
            self.use_bias
            and hasattr(self.to_kv, "bias")
            and self.to_kv.bias is not None
        ):
            concatenated_bias = self.to_kv.bias.data
            k_bias = concatenated_bias[:k_dim]
            v_bias = concatenated_bias[k_dim:]

            self.to_k.bias.data.copy_(k_bias)
            self.to_v.bias.data.copy_(v_bias)

        # Remove fused layer
        del self.to_kv
        if "to_kv" in self._modules:
            del self._modules["to_kv"]

        logger.debug("Unfused to_kv -> to_k, to_v")

    # Handle added projections (SD3/Flux style)
    if hasattr(self, "to_added_qkv"):
        # Get device and dtype
        device = self.to_added_qkv.weight.device
        dtype = self.to_added_qkv.weight.dtype

        # Get concatenated weights
        concatenated_weights = self.to_added_qkv.weight.data

        # Calculate dimensions
        total_dim = concatenated_weights.shape[0]
        q_dim = self.inner_dim
        k_dim = self.inner_kv_dim
        v_dim = self.inner_kv_dim

        assert (
            total_dim == q_dim + k_dim + v_dim
        ), f"Dimension mismatch for added QKV: {total_dim} != {q_dim} + {k_dim} + {v_dim}"

        # Split weights
        add_q_weight = concatenated_weights[:q_dim]
        add_k_weight = concatenated_weights[q_dim : q_dim + k_dim]
        add_v_weight = concatenated_weights[q_dim + k_dim :]

        # Create individual layers
        self.add_q_proj = nn.Linear(
            self.added_kv_proj_dim,
            q_dim,
            bias=self.added_proj_bias,
            device=device,
            dtype=dtype,
        )
        self.add_k_proj = nn.Linear(
            self.added_kv_proj_dim,
            k_dim,
            bias=self.added_proj_bias,
            device=device,
            dtype=dtype,
        )
        self.add_v_proj = nn.Linear(
            self.added_kv_proj_dim,
            v_dim,
            bias=self.added_proj_bias,
            device=device,
            dtype=dtype,
        )

        # Copy weights
        self.add_q_proj.weight.data.copy_(add_q_weight)
        self.add_k_proj.weight.data.copy_(add_k_weight)
        self.add_v_proj.weight.data.copy_(add_v_weight)

        # Handle biases
        if (
            self.added_proj_bias
            and hasattr(self.to_added_qkv, "bias")
            and self.to_added_qkv.bias is not None
        ):
            concatenated_bias = self.to_added_qkv.bias.data
            add_q_bias = concatenated_bias[:q_dim]
            add_k_bias = concatenated_bias[q_dim : q_dim + k_dim]
            add_v_bias = concatenated_bias[q_dim + k_dim :]

            self.add_q_proj.bias.data.copy_(add_q_bias)
            self.add_k_proj.bias.data.copy_(add_k_bias)
            self.add_v_proj.bias.data.copy_(add_v_bias)

        # Remove fused layer
        del self.to_added_qkv
        if "to_added_qkv" in self._modules:
            del self._modules["to_added_qkv"]

        logger.debug("Unfused to_added_qkv -> add_q_proj, add_k_proj, add_v_proj")

    # Mark as unfused
    self.fused_projections = False
    logger.debug("Unfusing complete")


@torch.no_grad()
def fuse_single_block_qkv_mlp(block, permanent=True):
    """
    Fuse QKV and MLP projections in FluxSingleTransformerBlock to match FAL's linear1.
    This creates a single projection outputting 7x dimensions (3x for QKV + 4x for MLP).
    """
    # Import from the correct location
    try:
        from helpers.models.flux.transformer import FluxSingleTransformerBlock
    except ImportError:
        from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
    
    if not isinstance(block, FluxSingleTransformerBlock):
        logger.warning(f"Block is not FluxSingleTransformerBlock, skipping fusion: {type(block)}")
        return
        
    if hasattr(block, 'linear1'):
        logger.debug("Block already has linear1, skipping fusion")
        return
    
    # Get attention module
    attn = block.attn
    device = attn.to_q.weight.data.device
    dtype = attn.to_q.weight.data.dtype
    hidden_dim = attn.to_q.weight.data.shape[1]  # 3072
    
    # Check if already fused QKV
    if hasattr(attn, 'to_qkv'):
        qkv_weight = attn.to_qkv.weight.data  # Shape: [9216, 3072]
    else:
        # Concatenate individual Q, K, V
        qkv_weight = torch.cat([
            attn.to_q.weight.data, 
            attn.to_k.weight.data, 
            attn.to_v.weight.data
        ])  # Shape: [9216, 3072]
    
    mlp_weight = block.proj_mlp.weight.data  # Shape: [12288, 3072]
    
    # Combine into single projection (QKV + MLP)
    concatenated_weights = torch.cat([qkv_weight, mlp_weight], dim=0)  # Shape: [21504, 3072]
    
    # Create fused layer (FAL's "linear1")
    block.linear1 = nn.Linear(
        hidden_dim,  # 3072
        concatenated_weights.shape[0],  # 21504 (7x)
        bias=True,
        device=device,
        dtype=dtype
    )
    block.linear1.weight.copy_(concatenated_weights)
    
    # Handle biases
    if hasattr(attn, 'to_qkv') and hasattr(attn.to_qkv, 'bias') and attn.to_qkv.bias is not None:
        qkv_bias = attn.to_qkv.bias.data
    elif attn.use_bias:
        qkv_bias = torch.cat([
            attn.to_q.bias.data,
            attn.to_k.bias.data,
            attn.to_v.bias.data
        ])
    else:
        qkv_bias = torch.zeros(9216, device=device, dtype=dtype)
    
    mlp_bias = block.proj_mlp.bias.data if hasattr(block.proj_mlp, 'bias') and block.proj_mlp.bias is not None else torch.zeros(12288, device=device, dtype=dtype)
    concatenated_bias = torch.cat([qkv_bias, mlp_bias])
    block.linear1.bias.copy_(concatenated_bias)
    
    if permanent:
        # Store original modules for potential unfusing
        if not hasattr(block, '_original_modules'):
            block._original_modules = {}
        
        # Store references before deletion
        block._original_modules['attn_to_q'] = attn.to_q if hasattr(attn, 'to_q') else None
        block._original_modules['attn_to_k'] = attn.to_k if hasattr(attn, 'to_k') else None
        block._original_modules['attn_to_v'] = attn.to_v if hasattr(attn, 'to_v') else None
        block._original_modules['attn_to_qkv'] = attn.to_qkv if hasattr(attn, 'to_qkv') else None
        block._original_modules['proj_mlp'] = block.proj_mlp
        
        # Delete original layers
        if hasattr(attn, 'to_qkv'):
            del attn.to_qkv
            if 'to_qkv' in attn._modules:
                del attn._modules['to_qkv']
        else:
            for attr in ['to_q', 'to_k', 'to_v']:
                if hasattr(attn, attr):
                    delattr(attn, attr)
                if attr in attn._modules:
                    del attn._modules[attr]
                    
        del block.proj_mlp
        if 'proj_mlp' in block._modules:
            del block._modules['proj_mlp']
    
    # Mark as fused
    block.fused_qkv_mlp = True
    logger.debug(f"Fused single block QKV+MLP into linear1 with output dim {concatenated_weights.shape[0]}")


@torch.no_grad()
def fuse_double_block_components(block, permanent=True):
    """
    Fuse components in FluxTransformerBlock to match FAL's double_blocks structure.
    This includes separate fusion for image and text paths.
    """
    # Import from the correct location
    try:
        from helpers.models.flux.transformer import FluxTransformerBlock
    except ImportError:
        from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
    
    if not isinstance(block, FluxTransformerBlock):
        logger.warning(f"Block is not FluxTransformerBlock, skipping fusion: {type(block)}")
        return
        
    attn = block.attn
    device = attn.to_q.weight.data.device
    dtype = attn.to_q.weight.data.dtype
    hidden_dim = attn.query_dim  # 3072
    
    # Store original modules
    if not hasattr(block, '_original_modules'):
        block._original_modules = {}
    
    # Fuse image path QKV
    if hasattr(attn, 'to_qkv'):
        # Already fused
        block.img_attn_qkv = attn.to_qkv
    else:
        # Fuse Q, K, V for image
        img_qkv_weights = torch.cat([
            attn.to_q.weight.data,
            attn.to_k.weight.data,
            attn.to_v.weight.data
        ])  # [9216, 3072]
        
        block.img_attn_qkv = nn.Linear(hidden_dim, 9216, bias=attn.use_bias, device=device, dtype=dtype)
        block.img_attn_qkv.weight.copy_(img_qkv_weights)
        
        if attn.use_bias:
            img_qkv_bias = torch.cat([
                attn.to_q.bias.data,
                attn.to_k.bias.data,
                attn.to_v.bias.data
            ])
            block.img_attn_qkv.bias.copy_(img_qkv_bias)
    
    # Fuse text path QKV
    if hasattr(attn, 'to_added_qkv'):
        # Already fused
        block.txt_attn_qkv = attn.to_added_qkv
    else:
        # Fuse add_q_proj, add_k_proj, add_v_proj for text
        txt_qkv_weights = torch.cat([
            attn.add_q_proj.weight.data,
            attn.add_k_proj.weight.data,
            attn.add_v_proj.weight.data
        ])  # [9216, 3072]
        
        block.txt_attn_qkv = nn.Linear(hidden_dim, 9216, bias=attn.added_proj_bias, device=device, dtype=dtype)
        block.txt_attn_qkv.weight.copy_(txt_qkv_weights)
        
        if attn.added_proj_bias:
            txt_qkv_bias = torch.cat([
                attn.add_q_proj.bias.data,
                attn.add_k_proj.bias.data,
                attn.add_v_proj.bias.data
            ])
            block.txt_attn_qkv.bias.copy_(txt_qkv_bias)
    
    # Keep modulation and FF layers as-is but add FAL-style aliases
    block.img_mod_lin = block.norm1.linear  # Image modulation (6x output)
    block.txt_mod_lin = block.norm1_context.linear  # Text modulation (6x output)
    
    # Add aliases for FF layers
    if hasattr(block.ff, 'net') and len(block.ff.net) >= 3:
        block.img_mlp_0 = block.ff.net[0]  # First layer (Linear proj)
        block.img_mlp_2 = block.ff.net[2]  # Second layer after activation
    
    if hasattr(block.ff_context, 'net') and len(block.ff_context.net) >= 3:
        block.txt_mlp_0 = block.ff_context.net[0]
        block.txt_mlp_2 = block.ff_context.net[2]
    
    # Keep attention output projections with aliases
    block.img_attn_proj = attn.to_out[0] if hasattr(attn.to_out, '__getitem__') else attn.to_out
    block.txt_attn_proj = attn.to_add_out
    
    if permanent:
        # Store originals for unfusing
        block._original_modules.update({
            'attn_to_q': getattr(attn, 'to_q', None),
            'attn_to_k': getattr(attn, 'to_k', None),
            'attn_to_v': getattr(attn, 'to_v', None),
            'attn_to_qkv': getattr(attn, 'to_qkv', None),
            'attn_add_q_proj': getattr(attn, 'add_q_proj', None),
            'attn_add_k_proj': getattr(attn, 'add_k_proj', None),
            'attn_add_v_proj': getattr(attn, 'add_v_proj', None),
            'attn_to_added_qkv': getattr(attn, 'to_added_qkv', None),
        })
        
        # Clean up original unfused layers if they exist
        for attr in ['to_q', 'to_k', 'to_v']:
            if hasattr(attn, attr) and hasattr(block, 'img_attn_qkv'):
                delattr(attn, attr)
                if attr in attn._modules:
                    del attn._modules[attr]
                    
        for attr in ['add_q_proj', 'add_k_proj', 'add_v_proj']:
            if hasattr(attn, attr) and hasattr(block, 'txt_attn_qkv'):
                delattr(attn, attr)
                if attr in attn._modules:
                    del attn._modules[attr]
    
    # Mark as FAL-style fused
    block.fal_kontext_fused = True
    logger.debug(f"Fused double block components to match FAL kontext structure")


def fuse_all_blocks_fal_kontext(model, permanent=True):
    """
    Apply FAL kontext-style fusion to all transformer blocks in the model.
    """
    fusion_count = {'single': 0, 'double': 0}
    
    # Fuse single transformer blocks
    if hasattr(model, 'single_transformer_blocks'):
        for i, block in enumerate(model.single_transformer_blocks):
            try:
                fuse_single_block_qkv_mlp(block, permanent=permanent)
                fusion_count['single'] += 1
            except Exception as e:
                logger.error(f"Failed to fuse single block {i}: {e}")
    
    # Fuse double transformer blocks (MMDiT)
    if hasattr(model, 'transformer_blocks'):
        for i, block in enumerate(model.transformer_blocks):
            try:
                fuse_double_block_components(block, permanent=permanent)
                fusion_count['double'] += 1
            except Exception as e:
                logger.error(f"Failed to fuse double block {i}: {e}")
    
    logger.info(f"FAL kontext fusion complete: {fusion_count['single']} single blocks, {fusion_count['double']} double blocks")
    return fusion_count


# Custom forward methods for fused blocks
def fused_single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    attention_mask=None,
):
    """
    Forward method for FluxSingleTransformerBlock with fused QKV+MLP (linear1).
    """
    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    
    # Use fused linear1 if available
    if hasattr(self, 'linear1'):
        # Apply fused projection
        fused_output = self.linear1(norm_hidden_states)  # [batch, seq, 21504]
        
        # Split into QKV and MLP components
        qkv_output, mlp_output = fused_output.split([9216, 12288], dim=-1)
        
        # Process QKV through attention
        batch_size, seq_len, _ = norm_hidden_states.shape
        inner_dim = 3072  # Hidden dimension
        head_dim = inner_dim // self.attn.heads
        
        # Reshape QKV
        qkv = qkv_output.view(batch_size, seq_len, 3, self.attn.heads, head_dim)
        q, k, v = qkv.unbind(2)  # Split into Q, K, V
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply norms if present
        if hasattr(self.attn, 'norm_q') and self.attn.norm_q is not None:
            q = self.attn.norm_q(q)
        if hasattr(self.attn, 'norm_k') and self.attn.norm_k is not None:
            k = self.attn.norm_k(k)
        
        # Apply rotary embeddings if present
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)
        
        # Attention computation
        if attention_mask is not None:
            from diffusers.models.transformers.transformer_flux import expand_flux_attention_mask
            attention_mask = expand_flux_attention_mask(hidden_states, attention_mask)
        
        # Use Flash Attention or standard attention
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            # Fallback to standard attention
            scale = 1.0 / (head_dim ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask
            probs = torch.nn.functional.softmax(scores, dim=-1)
            attn_output = torch.matmul(probs, v)
        
        # Reshape attention output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, inner_dim)
        
        # Apply attention output projection if present
        if hasattr(self.attn, 'to_out'):
            if hasattr(self.attn.to_out, '__getitem__'):
                attn_output = self.attn.to_out[0](attn_output)
            else:
                attn_output = self.attn.to_out(attn_output)
        
        # Apply MLP activation
        mlp_hidden_states = self.act_mlp(mlp_output)
    else:
        # Fallback to original implementation
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        
        if attention_mask is not None:
            from diffusers.models.transformers.transformer_flux import expand_flux_attention_mask
            attention_mask = expand_flux_attention_mask(hidden_states, attention_mask)
        
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )
    
    # Combine attention and MLP outputs
    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    gate = gate.unsqueeze(1)
    hidden_states = gate * self.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    
    return hidden_states


def apply_fal_kontext_forward_overrides(model):
    """
    Apply custom forward methods to support FAL kontext fusion.
    """
    try:
        from helpers.models.flux.transformer import FluxSingleTransformerBlock
    except ImportError:
        from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
    import types
    
    # Override forward for single blocks with fused layers
    if hasattr(model, 'single_transformer_blocks'):
        for block in model.single_transformer_blocks:
            if hasattr(block, 'linear1'):
                # Bind the custom forward method
                block.forward = types.MethodType(fused_single_block_forward, block)
                logger.debug("Applied fused forward to single transformer block")
    
    # Double blocks keep their original forward but use the aliased layers
    # The FAL-style aliases (img_attn_qkv, txt_attn_qkv, etc.) will be picked up
    # by the LoRA targeting system
    
    logger.info("Applied FAL kontext forward overrides")


def patch_attention_flexible():
    """Apply flexible fusion/unfusion patches to Attention class"""
    # Store originals
    Attention._original_fuse_projections = Attention.fuse_projections
    Attention._original_unfuse_projections = getattr(
        Attention, "unfuse_projections", None
    )

    # Apply our versions
    Attention.fuse_projections = fuse_projections_smart
    Attention.unfuse_projections = unfuse_projections_smart

    logger.info(
        f"Patched Attention with flexible fusion (permanent={PERMANENT_FUSION})"
    )


# Convenience functions for different use cases
def enable_permanent_fusion():
    """Enable permanent fusion mode globally"""
    global PERMANENT_FUSION
    PERMANENT_FUSION = True
    logger.info("Enabled permanent QKV fusion mode")


def enable_reversible_fusion():
    """Enable reversible fusion mode globally"""
    global PERMANENT_FUSION
    PERMANENT_FUSION = False
    logger.info("Enabled reversible QKV fusion mode")


patch_attention_flexible()
