#!/usr/bin/env python3
"""
Efficient script to download DeepSeek-R1-Distill-Qwen-1.5B from Hugging Face.

Usage:
    python scripts/download_deepseek_model.py [--output-dir DIR] [--token TOKEN]

Features:
    - Clean directory structure (no nested cache folders)
    - Progress tracking with tqdm
    - Resume capability for interrupted downloads
    - Proper error handling
    - Optional authentication
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface-hub not installed. Install with:")
    print("  pip install huggingface-hub")
    sys.exit(1)


def download_model(
    repo_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    output_dir: str = None,
    token: str = None
):
    """
    Download a Hugging Face model to a clean local directory.
    
    Args:
        repo_id: Hugging Face model repository ID
        output_dir: Local directory to save model files
        token: Hugging Face API token (optional, uses HF_TOKEN env var if not provided)
    """
    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading: {repo_id}")
    print(f"📂 Output directory: {output_dir.absolute()}")
    print(f"⚡ Using local_dir for clean structure (no cache hashes)")
    
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    try:
        # Download with local_dir for clean structure
        # resume_download=True allows resuming interrupted downloads
        # local_dir_use_symlinks=False copies files directly (cleaner)
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
            repo_type="model"
        )
        
        print(f"\n✅ Download complete!")
        print(f"📁 Model files location: {local_path}")
        
        # List downloaded files
        files = list(Path(local_path).rglob("*"))
        files = [f for f in files if f.is_file()]
        print(f"📊 Downloaded {len(files)} files")
        
        # Show key files
        key_files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
        print("\n🔑 Key files:")
        for key_file in key_files:
            matching = [f for f in files if f.name == key_file]
            if matching:
                print(f"  ✓ {key_file}")
            else:
                print(f"  ✗ {key_file} (not found)")
        
        return str(local_path)
        
    except Exception as e:
        print(f"\n❌ Error during download: {e}", file=sys.stderr)
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Verify the repository exists: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        print("  3. If authentication is required, run: huggingface-cli login")
        print("     Or set HF_TOKEN environment variable")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download DeepSeek-R1-Distill-Qwen-1.5B model from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model files (default: ./models/DeepSeek-R1-Distill-Qwen-1.5B)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (default: uses HF_TOKEN or HUGGINGFACE_HUB_TOKEN env var)"
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Hugging Face repository ID (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)"
    )
    
    args = parser.parse_args()
    
    download_model(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        token=args.token
    )


if __name__ == "__main__":
    main()
