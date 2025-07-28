#!/usr/bin/env python3
"""
Debug script to understand Flux transformer FeedForward structure
without loading the full model (which requires GPU).
"""

import inspect
from diffusers.models.transformers.flux_transformer_2d import FluxTransformerBlock

def analyze_flux_feedforward():
    """Analyze the FeedForward structure in Flux transformer."""
    
    print("=== Analyzing FluxTransformerBlock structure ===")
    
    # Get the source code of FluxTransformerBlock
    try:
        source = inspect.getsource(FluxTransformerBlock)
        
        # Look for FeedForward related code
        lines = source.split('\n')
        ff_lines = [line for line in lines if 'ff' in line.lower() or 'feedforward' in line.lower()]
        
        print("Lines containing 'ff' or 'feedforward':")
        for line in ff_lines:
            print(f"  {line}")
            
    except Exception as e:
        print(f"Could not get source: {e}")
    
    # Try to analyze the class structure
    print("\n=== FluxTransformerBlock attributes ===")
    
    # Look at class annotations/attributes
    if hasattr(FluxTransformerBlock, '__annotations__'):
        print("Annotations:", FluxTransformerBlock.__annotations__)
    
    # Look at __init__ signature
    try:
        init_sig = inspect.signature(FluxTransformerBlock.__init__)
        print("__init__ signature:", init_sig)
    except Exception as e:
        print(f"Could not get __init__ signature: {e}")


def analyze_feedforward_module():
    """Try to understand the FeedForward module structure."""
    
    try:
        # Import the specific FeedForward class if it exists
        from diffusers.models.transformers.flux_transformer_2d import FeedForward
        
        print("\n=== FeedForward module found ===")
        
        # Get source if possible
        try:
            ff_source = inspect.getsource(FeedForward)
            print("FeedForward source:")
            print(ff_source)
        except Exception as e:
            print(f"Could not get FeedForward source: {e}")
            
    except ImportError:
        print("\n=== No FeedForward class found in flux_transformer_2d ===")
        
        # Try other locations
        try:
            from diffusers.models.attention import FeedForward
            print("Found FeedForward in attention module")
            
            ff_source = inspect.getsource(FeedForward)
            print("FeedForward source:")
            print(ff_source[:1000] + "..." if len(ff_source) > 1000 else ff_source)
            
        except ImportError:
            print("Could not find FeedForward class")


if __name__ == "__main__":
    analyze_flux_feedforward()
    analyze_feedforward_module()