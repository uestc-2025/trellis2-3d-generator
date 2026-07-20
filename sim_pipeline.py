#!/usr/bin/env python3
"""
Path 1: Offline Physics Simulation Pipeline
GLB → Normalize → Convex Hull → PyBullet → Final Pose GLB

Usage:
  python sim_pipeline.py output_apple.glb                    # single drop
  python sim_pipeline.py A.glb B.glb C.glb                    # multi-object
  python sim_pipeline.py output_apple.glb --count 5 --spread   # 5 copies
"""
import os, sys, time
import numpy as np
import trimesh


def prep_mesh(glb_path, target_size=0.5):
    """Load, center, scale to real-world meters."""
    scene = trimesh.load(glb_path)
    mesh = list(scene.geometry.values())[0]
    mesh.vertices -= mesh.vertices.mean(axis=0)
    scale = target_size / (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
    mesh.vertices *= scale
    mesh.vertices[:, 1] -= mesh.vertices[:, 1].min()
    return mesh


def sim_drop(meshes, labels, output="sim_result.glb", duration=3.0):
    """Run PyBullet drop simulation, export final pose GLB."""
    import pybullet as p
    import pybullet_data

    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    bodies = []
    for i, (mesh, label) in enumerate(zip(meshes, labels)):
        ext = mesh.bounding_box.extents / 2
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext.tolist())
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=ext.tolist(),
                                      rgbaColor=[0.85, 0.15, 0.1, 1.0])
        x = (i - (len(meshes)-1)/2) * 0.8
        z = 1.5 + i * 0.6
        bid = p.createMultiBody(baseMass=0.15, baseCollisionShapeIndex=col_id,
                                 baseVisualShapeIndex=vis_id, basePosition=[x, 0, z])
        bodies.append((bid, mesh, label))
        print(f"  [{label}] drop from ({x:.1f}, 0, {z:.1f})")

    fps = 240
    for step in range(int(duration * fps)):
        p.stepSimulation()
        if step % (fps//2) == 0:
            for bid, _, label in bodies:
                pos, _ = p.getBasePositionAndOrientation(bid)
            if len(bodies) == 1:
                print(f"  t={step/fps:.2f}s  z={pos[2]:.3f}")

    # Collect final poses
    final_scene = trimesh.Scene()
    for bid, mesh, label in bodies:
        pos, orn = p.getBasePositionAndOrientation(bid)
        T = np.eye(4)
        T[:3, :3] = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        T[:3, 3] = pos
        fm = mesh.copy()
        fm.apply_transform(T)
        final_scene.add_geometry(fm, node_name=label)
        print(f"  [{label}] final: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    p.disconnect()
    final_scene.export(output)
    print(f"\n  Saved: {output} ({os.path.getsize(output)/1024**2:.1f} MB)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline Physics Simulation Pipeline")
    parser.add_argument("glb_files", nargs="+")
    parser.add_argument("--count", type=int, default=0, help="Duplicate first GLB N times")
    parser.add_argument("--output", default="sim_result.glb")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--size", type=float, default=0.5)
    args = parser.parse_args()

    glbs = args.glb_files
    if args.count > 1:
        glbs = [args.glb_files[0]] * args.count

    print(f"Simulating {len(glbs)} object(s)...\n")
    meshes = [prep_mesh(g, args.size) for g in glbs]
    labels = [os.path.splitext(os.path.basename(g))[0] for g in glbs]
    sim_drop(meshes, labels, args.output, args.duration)


if __name__ == "__main__":
    main()
