#!/usr/bin/env python3
"""
Download DINOv3 models to local cache for offline SAM3D usage.
"""
import os
import torch

# Set cache directory
cache_dir = os.path.join(os.getcwd(), "checkpoints", ".cache")
os.environ["TORCH_HOME"] = cache_dir
print(f"Setting TORCH_HOME to: {cache_dir}")

# Download DINOv3 models that SAM3D might use
models_to_download = [
    "dinov2_vitg14",  # Vision Transformer Giant
    "dinov2_vitl14",  # Vision Transformer Large
    "dinov2_vitb14",  # Vision Transformer Base
]

for model_name in models_to_download:
    print(f"\n{'='*60}")
    print(f"Downloading {model_name}...")
    print(f"{'='*60}")
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            source="github",
        )
        print(f"✓ Successfully downloaded {model_name}")
        del model  # Free memory
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")

print(f"\n{'='*60}")
print("Cache directory contents:")
print(f"{'='*60}")
hub_dir = os.path.join(cache_dir, "hub")
if os.path.exists(hub_dir):
    for item in os.listdir(hub_dir):
        item_path = os.path.join(hub_dir, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
        else:
            size_mb = os.path.getsize(item_path) / (1024 * 1024)
            print(f"  📄 {item} ({size_mb:.1f} MB)")
else:
    print("  ⚠️  Hub directory not found")

print(f"\n✓ Done! Models cached to: {hub_dir}")
