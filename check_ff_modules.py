#!/usr/bin/env python3
"""
Simple script to check FeedForward module structure from diffusers.
"""

def check_feedforward_structure():
    """Check the structure of diffusers FeedForward modules."""
    
    print("=== Checking diffusers FeedForward structure ===")
    
    try:
        # Import without loading models
        from diffusers.models.attention import FeedForward
        import torch.nn as nn
        
        print(f"FeedForward class: {FeedForward}")
        
        # Try to create a minimal instance to see structure
        # Use minimal parameters that won't require GPU
        try:
            ff = FeedForward(
                dim=64,  # Small dimension
                dim_out=64,
                mult=2,
                dropout=0.0,
                activation_fn="gelu",
                final_dropout=False,
            )
            
            print(f"\nFeedForward instance: {ff}")
            print(f"FeedForward modules:")
            for name, module in ff.named_modules():
                if name:  # Skip the root module
                    print(f"  {name}: {type(module).__name__} - {module}")
                    
        except Exception as e:
            print(f"Could not create FeedForward instance: {e}")
            
        # Check if there's a 'net' attribute structure
        print(f"\nFeedForward.__dict__.keys(): {list(FeedForward.__dict__.keys())}")
        
    except ImportError as e:
        print(f"Could not import FeedForward: {e}")
        
    # Check what's in the current working directory for reference
    try:
        import diffusers
        print(f"\nDiffusers version: {diffusers.__version__}")
    except:
        pass


def check_flux_module_naming():
    """Check how Flux modules are typically named."""
    
    print("\n=== Checking typical Flux module naming patterns ===")
    
    # Based on the files we've seen, let's check what structure is expected
    expected_patterns = [
        "ff.net.0.proj",  # What we're targeting
        "ff.net.2",       # What's missing
        "ff_context.net.0.proj",  # Text version
        "ff_context.net.2",       # Text version missing
    ]
    
    print("Expected patterns from configuration:")
    for pattern in expected_patterns:
        print(f"  {pattern}")
        
    print("\nBased on the double.md evidence:")
    print("  - ff.net.0.proj: ✅ Has LoRA weights")
    print("  - ff.net.2: ❌ Missing LoRA weights (section exists but empty)")
    print("  - ff_context.net.0.proj: ✅ Has LoRA weights") 
    print("  - ff_context.net.2: ❌ Missing LoRA weights (section exists but empty)")


if __name__ == "__main__":
    check_feedforward_structure()
    check_flux_module_naming()