"""
TRELLIS.2 Post-Processing Pipeline
===================================
Two modules:
  Module A: Geometry smoothing (remove surface noise like water droplets)
  Module B: Color calibration (match GLB texture to input photo foreground)

Usage:
  python pipeline_post.py <input_glb> [--original <photo>] [--output <out.glb>]
                            [--smooth] [--calibrate] [--all]

Examples:
  # Geometry only
  python pipeline_post.py output_apple.glb --smooth --output apple_smooth.glb

  # Color only (needs original photo for reference)
  python pipeline_post.py output_apple.glb --calibrate --original photo.jpg --output apple_color.glb

  # Both
  python pipeline_post.py output_apple.glb --all --original photo.jpg --output apple_final.glb
"""
import sys, os, struct, json, io
import numpy as np
from PIL import Image
import trimesh

# ============================================================
# Module A: Geometry Smoothing
# ============================================================

def geometry_smooth(mesh, lamb=0.5, mu=-0.53, iterations=5):
    """
    Taubin smoothing: uniform low-pass filter that preserves volume better than
    pure Laplacian. Effectively removes high-frequency surface noise (droplets,
    small bumps) while keeping the overall shape.

    Taubin filter = Laplacian forward step (shrink) + reverse step (un-shrink).
    lambda > 0, mu < -lambda for band-pass effect.

    Args:
        mesh: trimesh.Trimesh
        lamb: forward step scale (0-1)
        mu: reverse step scale (negative, |mu| > lamb)
        iterations: number of Taubin passes
    """
    print(f"\n[A] Geometry Smoothing: lambda={lamb}, mu={mu}, iters={iterations}")
    print(f"    Vertices: {len(mesh.vertices):,}")

    for it in range(iterations):
        vertices = mesh.vertices.copy()
        adjacency = mesh.vertex_neighbors
        new_vertices = vertices.copy()

        # Forward: Laplacian smoothing (shrink)
        for i in range(len(vertices)):
            neighbors = adjacency[i]
            if len(neighbors) < 2:
                continue
            centroid = vertices[neighbors].mean(axis=0)
            new_vertices[i] = vertices[i] + lamb * (centroid - vertices[i])

        # Reverse: Laplacian with negative scale (un-shrink)
        vertices2 = new_vertices.copy()
        for i in range(len(vertices2)):
            neighbors = adjacency[i]
            if len(neighbors) < 2:
                continue
            centroid = new_vertices[neighbors].mean(axis=0)
            vertices2[i] = new_vertices[i] + mu * (centroid - new_vertices[i])

        mesh.vertices = vertices2
        print(f"    iter {it+1}: done")

    return mesh


# ============================================================
# Module B: Color Calibration
# ============================================================

def extract_foreground_stats(image_path):
    """
    Extract foreground color statistics from the original photo using rembg.
    Falls back to full-image stats if rembg is not available.
    """
    from PIL import Image
    img = Image.open(image_path).convert('RGB')

    try:
        from rembg import remove
        print("  [B] Using rembg for foreground extraction...")
        fg = remove(img)
        fg_arr = np.array(fg)
        # Pixels with alpha > 128 are foreground
        alpha = fg_arr[:, :, 3] if fg_arr.shape[2] == 4 else np.ones(fg_arr.shape[:2]) * 255
        mask = alpha > 128
        if mask.sum() < 100:
            print("  [B] WARNING: rembg found almost no foreground, using full image")
            mask = np.ones(fg_arr.shape[:2], dtype=bool)
    except ImportError:
        print("  [B] rembg not installed, using full image stats")
        img_arr = np.array(img)
        mask = np.ones(img_arr.shape[:2], dtype=bool)

    img_arr = np.array(img)
    fg_pixels = img_arr[mask]
    stats = {
        'mean': fg_pixels.mean(axis=0),
        'std': fg_pixels.std(axis=0),
        'min': fg_pixels.min(axis=0),
        'max': fg_pixels.max(axis=0),
        'p10': np.percentile(fg_pixels, 10, axis=0),
        'p90': np.percentile(fg_pixels, 90, axis=0),
    }
    print(f"  [B] Foreground: {mask.sum():,} pixels, mean RGB [{stats['mean'][0]:.0f},{stats['mean'][1]:.0f},{stats['mean'][2]:.0f}]")
    return stats


def calibrate_colors(glb_path, fg_stats, strength=0.7):
    """
    Match GLB texture to foreground color statistics using histogram matching.

    Args:
        glb_path: path to input GLB
        fg_stats: foreground stats dict from extract_foreground_stats()
        strength: blend factor (0=original, 1=full match)
    """
    print(f"\n[B] Color Calibration: strength={strength}")

    scene = trimesh.load(glb_path)

    for name, geom in scene.geometry.items():
        if not hasattr(geom, 'visual'):
            continue
        mat = getattr(geom.visual, 'material', None)
        if mat is None or not hasattr(mat, 'baseColorTexture') or mat.baseColorTexture is None:
            continue

        tex_arr = np.array(mat.baseColorTexture.convert('RGB')).astype(np.float32)
        print(f"  [B] Original texture RGB mean: [{tex_arr[:,:,0].mean():.0f},{tex_arr[:,:,1].mean():.0f},{tex_arr[:,:,2].mean():.0f}]")

        # Per-channel histogram matching
        result = tex_arr.copy()
        for c in range(3):
            ch = tex_arr[:, :, c]
            t_mean, t_std = ch.mean(), max(ch.std(), 1.0)
            f_mean, f_std = fg_stats['mean'][c], max(fg_stats['std'][c], 1.0)

            # Match: scale to target std, shift to target mean
            matched = (ch - t_mean) * (f_std / t_std) + f_mean

            # Blend with original
            result[:, :, c] = ch * (1 - strength) + matched * strength

        # Clip and apply slight saturation boost
        result = np.clip(result, 0, 255).astype(np.uint8)
        enhanced = Image.fromarray(result)
        from PIL import ImageEnhance
        enhanced = ImageEnhance.Color(enhanced).enhance(1.15)
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)
        result = np.array(enhanced)

        print(f"  [B] Calibrated RGB mean: [{result[:,:,0].mean():.0f},{result[:,:,1].mean():.0f},{result[:,:,2].mean():.0f}]")

        # Apply back
        if mat.baseColorTexture.mode == 'RGBA':
            alpha = np.array(mat.baseColorTexture)[:, :, 3]
            result = np.concatenate([result, alpha[:, :, None]], axis=-1)

        mat.baseColorTexture = Image.fromarray(result)

        # Fix PBR factors for better color rendering
        mat.metallicFactor = 0.0
        mat.roughnessFactor = 0.5

    return scene


# ============================================================
# Main Pipeline
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='TRELLIS.2 Post-Processing Pipeline')
    parser.add_argument('input_glb', help='Input GLB file')
    parser.add_argument('--original', '-o', help='Original photo for color calibration')
    parser.add_argument('--output', '-O', default=None, help='Output GLB file')
    parser.add_argument('--smooth', action='store_true', help='Apply geometry smoothing')
    parser.add_argument('--calibrate', action='store_true', help='Apply color calibration')
    parser.add_argument('--all', '-a', action='store_true', help='Apply both smoothing and calibration')
    parser.add_argument('--smooth-strength', type=float, default=0.5, help='Taubin lambda (0-1)')
    parser.add_argument('--smooth-iters', type=int, default=5, help='Taubin iterations')
    parser.add_argument('--color-strength', type=float, default=0.7)
    args = parser.parse_args()

    if not args.smooth and not args.calibrate and not args.all:
        parser.print_help()
        sys.exit(1)

    if (args.calibrate or args.all) and not args.original:
        print("ERROR: --original <photo> is required for color calibration")
        sys.exit(1)

    if args.output is None:
        base = args.input_glb.replace('.glb', '')
        args.output = f"{base}_post.glb"

    print(f"Input:  {args.input_glb}")
    print(f"Output: {args.output}")

    if args.all:
        args.smooth = True
        args.calibrate = True

    # Load mesh
    print("\nLoading GLB...")
    scene = trimesh.load(args.input_glb)

    # Get the mesh from scene
    mesh = None
    mesh_name = None
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            mesh = geom
            mesh_name = name
            break

    if mesh is None:
        print("ERROR: No mesh found in GLB")
        sys.exit(1)

    # Module A: Geometry Smoothing
    if args.smooth:
        mesh = geometry_smooth(mesh,
                               lamb=args.smooth_strength,
                               mu=-(args.smooth_strength + 0.03),
                               iterations=args.smooth_iters)
        scene.geometry[mesh_name] = mesh

    # Module B: Color Calibration
    if args.calibrate:
        fg_stats = extract_foreground_stats(args.original)
        # Write temp file for trimesh to load
        tmp_path = args.input_glb + '.tmp.glb'
        scene.export(tmp_path)
        scene = calibrate_colors(tmp_path, fg_stats,
                                 strength=args.color_strength)
        os.remove(tmp_path)

    # Export
    print(f"\nExporting to {args.output}...")
    scene.export(args.output)
    sz = os.path.getsize(args.output) / 1024**2
    print(f"Done: {args.output} ({sz:.1f} MB)")


if __name__ == '__main__':
    main()
