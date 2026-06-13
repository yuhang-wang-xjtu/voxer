"""
Streaming voxelization pipeline: download -> process -> delete.

Problem: Raw 3D models from Objaverse are 50-200MB each.
5000 models = 250GB+ storage, unsustainable on Colab/Google Drive.

Solution: Process one model at a time.
    1. Download glb -> temporary storage
    2. Render 2D reference views (for analysis/pseudo-labeling)
    3. Voxelize to 64^3 RGBA (with background filtering)
    4. Save voxels -> persistent storage (1 MB each)
    5. Delete raw glb -> free space
    6. Move to next model

Storage comparison:
    Raw:   5000 models x 50 MB = 250 GB
    Voxels: 5000 models x 1 MB = 5 GB
    Views:  5000 models x 6 x 40 KB = 1.2 GB
    Total: ~6 GB (40x reduction)

For 2D analysis: Pre-rendered views are saved alongside voxels,
so CLIP scoring, style analysis, etc. can still be done offline.
"""

import os
import gc
import json
import time
import shutil
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class VoxelizeStats:
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    total_time: float = 0.0
    per_model_times: List[float] = field(default_factory=list)
    failed_uids: List[str] = field(default_factory=list)


def render_reference_views(
    mesh,
    output_dir: str,
    uid: str,
    num_views: int = 6,
) -> List[str]:
    """Render 2D reference views of the original mesh as small PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    try:
        points, _ = mesh.sample(50000, return_index=True)
    except Exception:
        points = mesh.vertices

    if len(points) == 0:
        return paths

    view_names = ["front", "back", "left", "right", "top", "bottom"]
    for i, name in enumerate(view_names[:num_views]):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        if name == "front":
            x, y = points[:, 0], points[:, 2]
        elif name == "back":
            x, y = -points[:, 0], points[:, 2]
        elif name == "left":
            x, y = -points[:, 1], points[:, 2]
        elif name == "right":
            x, y = points[:, 1], points[:, 2]
        elif name == "top":
            x, y = points[:, 0], points[:, 1]
        else:
            x, y = points[:, 0], -points[:, 1]

        center_x, center_y = x.mean(), y.mean()
        x, y = x - center_x, y - center_y

        ax.scatter(x, y, s=0.3, c="black", alpha=0.3, edgecolors="none")
        ax.set_aspect("equal")
        ax.set_xlim(x.min() - 0.1, x.max() + 0.1)
        ax.set_ylim(y.min() - 0.1, y.max() + 0.1)
        ax.axis("off")

        path = os.path.join(output_dir, f"{uid}_view_{name}.png")
        fig.savefig(path, dpi=64, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        paths.append(path)

    return paths


def download_and_voxelize(
    uid: str,
    annotations: Dict,
    output_dir: str,
    voxel_resolution: int = 64,
    render_views: bool = True,
    num_views: int = 6,
    filter_background: bool = True,
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Download one model, voxelize it, save results, delete original.

    Args:
        uid: Objaverse model UID
        annotations: objaverse annotations dict (uid -> metadata)
        output_dir: Where to save voxels and views
        voxel_resolution: Output voxel grid resolution
        render_views: Whether to render and save 2D reference views
        num_views: Number of reference views to render
        filter_background: Use inspect.py heuristics to filter background parts
        cache_dir: Optional directory to cache downloaded glb files

    Returns:
        Path to saved voxel .npy file, or None if failed
    """
    import objaverse
    import trimesh
    from voxer.inspect import (
        load_glb_parts, voxelize_filtered,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Download
    try:
        download_dir = cache_dir or os.path.join(output_dir, "_tmp_downloads")
        os.makedirs(download_dir, exist_ok=True)

        downloaded = objaverse.load_objects(
            uids=[uid],
            download_processes=1,
        )
        glb_path = downloaded.get(uid)
        if glb_path is None or not os.path.exists(glb_path):
            return None
    except Exception:
        return None

    # Step 2: Render 2D reference views
    view_paths = []
    if render_views:
        try:
            views_dir = os.path.join(output_dir, "..", "views")
            views_dir = os.path.abspath(views_dir)
            os.makedirs(views_dir, exist_ok=True)
            scene = trimesh.load(glb_path, force="scene")
            if isinstance(scene, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
            else:
                mesh = scene
            view_paths = render_reference_views(mesh, views_dir, uid, num_views)
        except Exception:
            pass

    # Step 3: Voxelize (with or without background filtering)
    try:
        if filter_background:
            voxel = voxelize_filtered(glb_path, resolution=voxel_resolution)
        else:
            parts = load_glb_parts(glb_path)
            if parts is None:
                return None
            voxel = voxelize_filtered(glb_path, resolution=voxel_resolution)
    except Exception:
        voxel = None

    # Step 4: Save
    if voxel is not None and voxel.sum() > 0:
        voxel_path = os.path.join(output_dir, f"{uid}.npy")
        np.save(voxel_path, voxel)

        meta = {
            "uid": uid,
            "name": str(annotations.get("name", "")),
            "description": str(annotations.get("description", "")),
            "tags": [str(t.get("name", "")) for t in annotations.get("tags", [])],
            "categories": [str(c.get("name", "")) for c in annotations.get("categories", [])],
            "views": [os.path.basename(v) for v in view_paths],
            "voxel_resolution": voxel_resolution,
            "non_empty_ratio": float((voxel[..., 3] > 0).mean()),
        }
        meta_path = os.path.join(output_dir, f"{uid}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    else:
        voxel_path = None

    # Step 5: Delete raw glb
    try:
        if os.path.exists(glb_path):
            os.remove(glb_path)
    except Exception:
        pass

    gc.collect()
    return voxel_path


def stream_voxelize_batch(
    uids: List[str],
    output_dir: str,
    annotations: Optional[Dict] = None,
    voxel_resolution: int = 64,
    render_views: bool = True,
    num_views: int = 6,
    filter_background: bool = True,
    cache_dir: Optional[str] = None,
    resume: bool = True,
    cleanup_on_finish: bool = True,
) -> VoxelizeStats:
    """
    Stream-process a batch of models: download -> voxelize -> delete.

    Args:
        uids: List of Objaverse model UIDs
        output_dir: Directory to save results
        annotations: Pre-loaded annotations dict (lazy loaded if None)
        voxel_resolution: Output voxel resolution (default 64)
        render_views: Render 2D reference views before voxelization
        num_views: Number of reference views per model
        filter_background: Apply background part filtering
        cache_dir: Temporary directory for downloads
        resume: Skip models that already have voxels saved
        cleanup_on_finish: Delete cache_dir when done

    Returns:
        VoxelizeStats with processing statistics
    """
    import objaverse

    stats = VoxelizeStats(total=len(uids))
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    cache_dir = cache_dir or os.path.join(output_dir, "_download_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Load annotations if needed
    if annotations is None:
        print("Loading annotations...")
        annotations = objaverse.load_annotations()

    # Resume: check which uids already processed
    if resume:
        existing = set()
        for f in os.listdir(output_dir):
            if f.endswith(".npy"):
                existing.add(f.replace(".npy", ""))
        remaining = [u for u in uids if u not in existing]
        stats.skipped = len(uids) - len(remaining)
        if stats.skipped > 0:
            print(f"Resuming: {stats.skipped} already processed, "
                  f"{len(remaining)} remaining")
        uids = remaining

    print(f"\n{'='*60}")
    print(f"Streaming Voxelization Pipeline")
    print(f"  Models: {len(uids)}")
    print(f"  Resolution: {voxel_resolution}^3")
    print(f"  Views: {num_views} ({'enabled' if render_views else 'disabled'})")
    print(f"  Background filter: {'enabled' if filter_background else 'disabled'}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    for i, uid in enumerate(uids):
        t0 = time.time()

        if i % 50 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / max(i, 1)) * (len(uids) - i) if i > 0 else 0
            print(f"[{i}/{len(uids)}] elapsed={elapsed:.0f}s, "
                  f"ETA={eta:.0f}s | ok={stats.success}, fail={stats.failed}")

        anno = annotations.get(uid, {})

        result = download_and_voxelize(
            uid=uid,
            annotations=anno,
            output_dir=output_dir,
            voxel_resolution=voxel_resolution,
            render_views=render_views,
            num_views=num_views,
            filter_background=filter_background,
            cache_dir=cache_dir,
        )

        t1 = time.time()
        stats.per_model_times.append(t1 - t0)

        if result is not None:
            stats.success += 1
        else:
            stats.failed += 1
            stats.failed_uids.append(uid)

    stats.total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Streaming Voxelization Complete")
    print(f"  Total:   {stats.total}")
    print(f"  Success: {stats.success} ({stats.success/max(stats.total,1):.1%})")
    print(f"  Failed:  {stats.failed}")
    print(f"  Skipped: {stats.skipped}")
    print(f"  Time:    {stats.total_time:.0f}s "
          f"({stats.total_time/max(stats.success,1):.1f}s/model)")
    print(f"{'='*60}")

    if cleanup_on_finish and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Save stats
    stats_path = os.path.join(output_dir, "voxelize_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "total": stats.total,
            "success": stats.success,
            "failed": stats.failed,
            "skipped": stats.skipped,
            "total_time": stats.total_time,
            "avg_time_per_model": stats.total_time / max(stats.success, 1),
            "failed_uids": stats.failed_uids,
        }, f, indent=2)

    # Build manifest
    manifest = []
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(".json") and not f.startswith("voxelize"):
            meta_path = os.path.join(output_dir, f)
            with open(meta_path, "r", encoding="utf-8") as mf:
                manifest.append(json.load(mf))

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Manifest: {len(manifest)} models -> {manifest_path}")
    return stats


def load_voxels_from_stream(
    voxel_dir: str,
    max_models: Optional[int] = None,
    shuffle: bool = True,
) -> np.ndarray:
    """Load all voxelized models from a stream output directory."""
    voxel_files = sorted([f for f in os.listdir(voxel_dir) if f.endswith(".npy")])

    if shuffle:
        np.random.shuffle(voxel_files)

    if max_models:
        voxel_files = voxel_files[:max_models]

    voxels = []
    for f in voxel_files:
        path = os.path.join(voxel_dir, f)
        v = np.load(path)
        voxels.append(v)

    return np.array(voxels, dtype=np.uint8)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream voxelization for Objaverse models"
    )
    parser.add_argument("--num-models", "-n", type=int, default=100)
    parser.add_argument("--output", "-o", default="./stream_output")
    parser.add_argument("--resolution", "-r", type=int, default=64)
    parser.add_argument("--no-views", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--categories", "-c", nargs="+",
                        default=["chair", "table", "sofa", "lamp"])

    args = parser.parse_args()

    import objaverse

    print("Loading annotations...")
    annotations = objaverse.load_annotations()

    uids = []
    for uid, anno in annotations.items():
        name = str(anno.get("name", "")).lower()
        tags = " ".join([
            str(t.get("name", "")).lower() for t in anno.get("tags", [])
        ])
        desc = str(anno.get("description", ""))
        text = name + " " + tags
        if any(kw in text for kw in args.categories) and len(desc) > 20:
            uids.append(uid)
        if len(uids) >= args.num_models:
            break

    print(f"Found {len(uids)} models matching: {args.categories}")

    stats = stream_voxelize_batch(
        uids=uids,
        output_dir=args.output,
        annotations=annotations,
        voxel_resolution=args.resolution,
        render_views=not args.no_views,
        filter_background=not args.no_filter,
        resume=not args.no_resume,
    )
