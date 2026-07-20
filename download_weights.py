#!/usr/bin/env python3
"""
Download TRELLIS.2 model weights and dependencies.

Total download: ~20 GB
Cache directory: ~/.cache/trellis2/ (configurable via TRELLIS2_CACHE env var)

Gated models (need HuggingFace access):
  - DinoV3 (facebook/dinov3-vitl16-pretrain-lvd1689m) → downloaded from ModelScope (no auth)
  - RMBG-2.0 (briaai/RMBG-2.0) → needs HF_TOKEN
    1. Visit https://huggingface.co/briaai/RMBG-2.0 → "Agree and access repository"
    2. Create token at https://huggingface.co/settings/tokens
    3. export HF_TOKEN=your_token
"""
import os, sys, json, shutil

CACHE = os.environ.get("TRELLIS2_CACHE", os.path.expanduser("~/.cache/trellis2"))
PIPELINE_DIR = os.path.join(CACHE, "pipeline")
IMAGE_DIR = os.path.join(CACHE, "image")
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


def download_file(url, dest, desc=""):
    """Download with progress bar."""
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    print(f"  Downloading {desc}...")
    print(f"    {url}")

    def report(count, block_size, total_size):
        pct = min(100, int(count * block_size * 100 / max(total_size, 1)))
        mb = min(total_size, count * block_size) / 1024**2
        total_mb = total_size / 1024**2
        print(f"\r    {pct:3d}%  {mb:.0f}/{total_mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()
    return dest


def download_trellis2_4b():
    """Download TRELLIS.2-4B model (from HuggingFace)."""
    print("\n[1/4] TRELLIS.2-4B (~16 GB)")

    from huggingface_hub import snapshot_download
    repo = "JeffreyXiang/TRELLIS.2-4B"

    try:
        path = snapshot_download(
            repo,
            local_dir=os.path.join(PIPELINE_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"  OK: {path}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Try manual download:")
        print(f"    pip install huggingface_hub")
        print(f"    export HF_ENDPOINT=https://hf-mirror.com")
        print(f"    huggingface-cli download {repo} --local-dir {PIPELINE_DIR}")
        raise


def download_image_large():
    """Download TRELLIS-image-large (~3.1 GB)."""
    print("\n[2/4] TRELLIS-image-large (~3.1 GB)")

    from huggingface_hub import snapshot_download
    repo = "JeffreyXiang/TRELLIS-image-large"

    try:
        path = snapshot_download(
            repo,
            local_dir=os.path.join(IMAGE_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"  OK: {path}")
    except Exception as e:
        print(f"  ERROR: {e}")
        raise


def download_dinov3():
    """Download DinoV3 from ModelScope (no auth needed)."""
    print("\n[3/4] DinoV3 ViT-L/16 (~1.2 GB)")

    from modelscope import snapshot_download
    try:
        path = snapshot_download(
            "tastelikefeet/dinov3-vitl16-pretrain-lvd1689m",
            cache_dir=os.path.join(CACHE, "modelscope"),
        )
        print(f"  OK: {path}")
        return path
    except ImportError:
        print("  modelscope not installed, trying pip...")
        os.system(f"{sys.executable} -m pip install modelscope -q")
        from modelscope import snapshot_download
        path = snapshot_download(
            "tastelikefeet/dinov3-vitl16-pretrain-lvd1689m",
            cache_dir=os.path.join(CACHE, "modelscope"),
        )
        print(f"  OK: {path}")
        return path


def download_rmbg2():
    """Download RMBG-2.0 (needs HF token for gated access)."""
    print("\n[4/4] RMBG-2.0 (~1 GB)")

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("  WARNING: HF_TOKEN not set. RMBG-2.0 requires HuggingFace access.")
        print("  1. Visit https://huggingface.co/briaai/RMBG-2.0")
        print("  2. Click 'Agree and access repository'")
        print("  3. Create token at https://huggingface.co/settings/tokens")
        print("  4. Run: export HF_TOKEN=your_token")
        print("  Skipping RMBG-2.0 download for now...")
        return None

    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(
            "briaai/RMBG-2.0",
            local_dir=os.path.join(CACHE, "rmbg2"),
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True,
        )
        print(f"  OK: {path}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  RMBG-2.0 download failed. Background removal will be unavailable.")


def create_pipeline_config(dinov3_path):
    """Create pipeline.json pointing to downloaded models."""
    print("\nCreating pipeline config...")

    config = {
        "name": "Trellis2ImageTo3DPipeline",
        "args": {
            "models": {
                "sparse_structure_decoder": os.path.join(IMAGE_DIR, "ckpts", "ss_dec_conv3d_16l8_fp16"),
                "sparse_structure_flow_model": "ckpts/ss_flow_img_dit_1_3B_64_bf16",
                "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
                "shape_slat_flow_model_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
                "shape_slat_flow_model_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
                "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
                "tex_slat_flow_model_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
                "tex_slat_flow_model_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
            },
            "sparse_structure_sampler": {
                "name": "FlowEulerGuidanceIntervalSampler",
                "args": {"sigma_min": 1e-05},
                "params": {
                    "steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7,
                    "guidance_interval": [0.6, 1.0], "rescale_t": 5.0,
                },
            },
            "shape_slat_sampler": {
                "name": "FlowEulerGuidanceIntervalSampler",
                "args": {"sigma_min": 1e-05},
                "params": {
                    "steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5,
                    "guidance_interval": [0.6, 1.0], "rescale_t": 3.0,
                },
            },
            "tex_slat_sampler": {
                "name": "FlowEulerGuidanceIntervalSampler",
                "args": {"sigma_min": 1e-05},
                "params": {
                    "steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0,
                    "guidance_interval": [0.6, 0.9], "rescale_t": 3.0,
                },
            },
            "image_cond_model": {
                "name": "DinoV3FeatureExtractor",
                "args": {"model_name": dinov3_path},
            },
            "rembg_model": {
                "name": "BiRefNet",
                "args": {"model_name": "briaai/RMBG-2.0"},
            },
            "default_pipeline_type": "512",
        },
    }

    # Include normalization stats (default values for TRELLIS.2-4B)
    config["args"]["shape_slat_normalization"] = {
        "mean": [0.781, 0.018, -0.495, -0.558, 1.061, 0.093, 1.518, -0.933,
                 -0.733, 2.604, -0.118, -2.144, 0.495, -2.180, -2.131, -0.997,
                 0.261, -2.217, 1.260, -0.150, 3.791, 1.481, -1.046, -1.524,
                 -0.060, 2.221, 1.621, 0.877, 0.567, -3.176, -3.187, 1.579],
        "std": [5.972, 4.707, 5.445, 5.210, 5.320, 4.547, 5.021, 5.444,
                5.227, 5.683, 4.831, 5.286, 5.652, 5.368, 5.525, 4.731,
                4.805, 5.124, 5.531, 5.619, 5.104, 5.418, 5.270, 5.547,
                5.635, 5.235, 6.110, 5.511, 6.237, 4.879, 5.347, 5.406],
    }
    config["args"]["tex_slat_normalization"] = {
        "mean": [3.502, 2.212, 2.226, 0.251, -0.026, -0.687, 0.440, -0.928,
                 0.029, -0.340, -0.870, 1.038, -0.972, 0.126, -1.129, 0.455,
                 -1.210, 2.069, 0.545, 2.569, -0.323, 2.293, -1.926, -1.218,
                 1.214, 0.972, -0.024, 0.107, 2.022, 0.251, -0.662, -0.769],
        "std": [2.666, 2.744, 2.765, 2.595, 3.037, 2.291, 2.145, 2.912,
                2.969, 2.502, 2.155, 3.163, 2.621, 2.382, 3.187, 3.022,
                2.296, 3.235, 3.233, 2.260, 2.875, 2.811, 3.293, 2.675,
                2.681, 2.372, 2.452, 2.354, 2.995, 2.380, 2.786, 2.775],
    }

    config_path = os.path.join(PIPELINE_DIR, "pipeline.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_path}")


# ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TRELLIS.2 Weight Downloader")
    print(f"  Cache: {CACHE}")
    print("=" * 60)

    os.makedirs(PIPELINE_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 1-2: Main model weights
    download_trellis2_4b()
    download_image_large()

    # 3: DinoV3
    dinov3_path = download_dinov3()

    # 4: RMBG-2.0
    download_rmbg2()

    # Create config
    create_pipeline_config(dinov3_path)

    print(f"\n{'=' * 60}")
    print("  Download complete!")
    print(f"  Cache: {CACHE}")
    print(f"  Run: python generate.py <image> -o output.glb")
    print("=" * 60)


if __name__ == "__main__":
    main()
