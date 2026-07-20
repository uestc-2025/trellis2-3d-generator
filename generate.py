#!/usr/bin/env python3
"""
TRELLIS.2 Image-to-3D Generator
Usage: python generate.py <image> [-o output.glb] [-r 512|1024] [--steps 12]
"""
import argparse
import os, sys, time, gc, json
import numpy as np
import torch
from PIL import Image

# ── Path setup ────────────────────────────────────────────
# TRELLIS.2 must be installed. Check common locations.
TRELLIS_PATHS = [
    os.path.expanduser("~/TRELLIS.2"),
    os.path.join(os.path.dirname(__file__), "TRELLIS.2"),
    "/root/TRELLIS.2",
]
TRELLIS_ROOT = None
for p in TRELLIS_PATHS:
    if os.path.exists(os.path.join(p, "trellis2", "__init__.py")):
        TRELLIS_ROOT = p
        break

if TRELLIS_ROOT is None:
    print("ERROR: TRELLIS.2 not found. Clone it first:")
    print("  git clone https://github.com/microsoft/TRELLIS.2.git")
    print("  cd TRELLIS.2 && pip install -e .")
    sys.exit(1)

sys.path.insert(0, TRELLIS_ROOT)

# ── Cache paths ────────────────────────────────────────────
CACHE_BASE = os.environ.get("TRELLIS2_CACHE", os.path.expanduser("~/.cache/trellis2"))
PIPELINE_CACHE = os.path.join(CACHE_BASE, "pipeline")
IMAGE_CACHE = os.path.join(CACHE_BASE, "image")


def check_cache():
    """Verify model caches exist."""
    pipeline_json = os.path.join(PIPELINE_CACHE, "pipeline.json")
    if not os.path.exists(pipeline_json):
        print("ERROR: Model weights not found. Run download_weights.py first.")
        print(f"  Expected: {pipeline_json}")
        sys.exit(1)
    return PIPELINE_CACHE


def load_envmap(assets_dir):
    """Load an HDR environment map."""
    import OpenEXR, Imath
    from trellis2.renderers import EnvMap

    # Look for any .exr file
    hdri_dir = os.path.join(assets_dir, "hdri")
    exr_files = []
    if os.path.exists(hdri_dir):
        exr_files = [f for f in os.listdir(hdri_dir) if f.endswith('.exr')]

    if not exr_files:
        print("WARNING: No HDR envmap found. Using default lighting.")
        # Create a simple flat envmap
        hdr = np.ones((512, 1024, 3), dtype=np.float32) * 0.5
    else:
        path = os.path.join(hdri_dir, exr_files[0])
        f = OpenEXR.InputFile(path)
        dw = f.header()['dataWindow']
        w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        rgb = [np.frombuffer(f.channel(c, pt), dtype=np.float32).reshape(h, w) for c in 'RGB']
        hdr = np.stack(rgb, axis=-1)

    return EnvMap(torch.tensor(hdr, dtype=torch.float32, device='cuda'))


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 Image-to-3D")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--output", default="output.glb", help="Output GLB path")
    parser.add_argument("-r", "--resolution", type=int, default=512, choices=[512, 1024],
                        help="Pipeline resolution (512 = lower VRAM)")
    parser.add_argument("--steps", type=int, default=12, help="Sampling steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pbr", action="store_true", help="Disable PBR material export")
    args = parser.parse_args()

    # Check input
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    # Check cache
    cache = check_cache()

    # Environment
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.mem_get_info()
    print(f"VRAM: free={mem[0]/1e9:.1f}GB / total={mem[1]/1e9:.1f}GB")
    print(f"Input: {args.image}")
    print(f"Pipeline: {args.resolution}, steps={args.steps}")
    print()

    # ── Load pipeline ──────────────────────────────────
    print("[1/4] Loading pipeline...")
    t0 = time.time()
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(cache)
    pipeline.low_vram = True
    pipeline.default_pipeline_type = str(args.resolution)
    pipeline.cuda()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Load envmap ────────────────────────────────────
    print("[2/4] Loading environment map...")
    assets = os.path.join(TRELLIS_ROOT, "assets")
    envmap = load_envmap(assets)

    # ── Load image ─────────────────────────────────────
    print("[3/4] Processing image...")
    image = Image.open(args.image).convert('RGB')
    print(f"  Size: {image.size}")

    # ── Inference ──────────────────────────────────────
    print("[4/4] Running inference...")
    t0 = time.time()
    torch.cuda.empty_cache()
    gc.collect()

    pipeline_type = str(args.resolution)
    mesh = pipeline.run(
        image,
        pipeline_type=pipeline_type,
        num_samples=1,
        seed=args.seed,
        sparse_structure_sampler_params={'steps': args.steps},
        shape_slat_sampler_params={'steps': args.steps},
        tex_slat_sampler_params={'steps': args.steps},
    )[0]

    dt = time.time() - t0
    v = mesh.vertices.cpu().numpy()
    print(f"  Done in {dt:.1f}s")
    print(f"  Vertices: {v.shape[0]:,}  Faces: {mesh.faces.shape[0]:,}")
    print(f"  PBR channels: {mesh.attrs.shape[1]}")

    # ── Export ─────────────────────────────────────────
    print(f"\nExporting to {args.output}...")
    t0 = time.time()

    mesh.simplify(5000000)
    torch.cuda.empty_cache()

    import o_voxel
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=500000,
        texture_size=2048,
        remesh=True, remesh_band=1, remesh_project=0,
        verbose=False,
    )
    glb.export(args.output, extension_webp=True)
    sz = os.path.getsize(args.output) / 1024**2
    print(f"  Done in {time.time()-t0:.1f}s → {args.output} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
