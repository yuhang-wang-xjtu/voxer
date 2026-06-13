"""
Model inspection and part extraction tool for Objaverse glb files.

Problems to solve:
1. Some Sketchfab models include background/scene geometry (floor, walls,
   props) that get voxelized along with the main object.
2. Need a way to visually inspect models and identify problematic ones.

Features:
- Load glb files and list all sub-meshes
- Visualize each sub-mesh and the combined model (3-view projections)
- Heuristic detection of background/scene geometry
- Batch inspection with HTML summary report
- Filter out background parts and re-voxelize cleanly
"""

import os
import json
import math
import warnings
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class MeshInfo:
    name: str
    index: int
    num_vertices: int
    num_faces: int
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    volume: float
    extent: float
    flatness: float
    is_ground_plane: bool = False
    is_background: bool = False
    flag_reasons: List[str] = field(default_factory=list)


def load_glb_parts(file_path: str, sample_points: int = 50000) -> Optional[Tuple[List["trimesh.Trimesh"], "trimesh.Trimesh"]]:
    """Load a glb file and return list of sub-meshes plus combined mesh."""
    import trimesh

    try:
        scene = trimesh.load(file_path, force="scene")
    except Exception as e:
        try:
            scene = trimesh.load(file_path, force="mesh")
            if scene is None:
                return None
            return [scene], scene
        except Exception:
            return None

    if not isinstance(scene, trimesh.Scene):
        if scene is not None:
            return [scene], scene
        return None

    geometry = list(scene.geometry.values())
    if len(geometry) == 0:
        return None

    combined = trimesh.util.concatenate(geometry)
    return geometry, combined


def analyze_mesh(mesh: "trimesh.Trimesh", idx: int) -> MeshInfo:
    """Analyze a single mesh and return metadata."""
    vertices = mesh.vertices
    name = mesh.metadata.get("name", f"mesh_{idx}") if hasattr(mesh, "metadata") else f"mesh_{idx}"

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    centroid = vertices.mean(axis=0)
    extent = np.linalg.norm(bbox_max - bbox_min)

    sizes = bbox_max - bbox_min
    volume = sizes[0] * sizes[1] * sizes[2]

    sorted_sizes = sorted(sizes)
    flatness = sorted_sizes[2] / (sorted_sizes[0] + 1e-8)

    return MeshInfo(
        name=name,
        index=idx,
        num_vertices=len(vertices),
        num_faces=len(mesh.faces) if mesh.faces is not None else 0,
        centroid=centroid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        volume=volume,
        extent=extent,
        flatness=flatness,
    )


def detect_background_parts(
    meshes: List[MeshInfo],
    total_volume: Optional[float] = None,
    plane_flatness_threshold: float = 50.0,
    volume_ratio_threshold: float = 0.7,
    centroid_distance_std: float = 2.5,
) -> List[MeshInfo]:
    """
    Flag meshes that are likely background/scene elements.

    Heuristics:
    1. Very flat relative to other dimensions (ground plane / wall)
    2. Disproportionately large volume compared to other parts
    3. Centroid far from the cluster of other centroids
    """
    if len(meshes) <= 1:
        return meshes

    volumes = [m.volume for m in meshes]
    centroids = np.array([m.centroid for m in meshes])
    median_centroid = np.median(centroids, axis=0)
    centroid_distances = np.linalg.norm(centroids - median_centroid, axis=1)

    mean_dist = np.mean(centroid_distances) if len(centroid_distances) > 0 else 0
    std_dist = np.std(centroid_distances) if len(centroid_distances) > 0 else 1

    total_vol = total_volume or max(volumes)

    for i, m in enumerate(meshes):
        reasons = []

        if m.flatness > plane_flatness_threshold:
            m.is_ground_plane = True
            reasons.append(f"flatness={m.flatness:.1f} (very flat, possible plane)")

        if volumes[i] > total_vol * volume_ratio_threshold and len(meshes) > 1:
            reasons.append(f"volume dominates ({volumes[i]/total_vol:.1%} of total)")

        if centroid_distances[i] > mean_dist + centroid_distance_std * std_dist and len(meshes) > 1:
            reasons.append(f"centroid offset {centroid_distances[i]:.2f}, far from cluster")

        if reasons:
            m.is_background = True
            m.flag_reasons = reasons

    return meshes


def render_mesh_views(
    mesh: "trimesh.Trimesh",
    sample_points: int = 20000,
    title: str = "",
    color: Optional[str] = None,
) -> "plt.Figure":
    """Render 3 orthogonal views of a mesh as scatter plots."""
    import matplotlib.pyplot as plt

    try:
        points, _ = mesh.sample(sample_points, return_index=True)
    except Exception:
        vertices = mesh.vertices
        if len(vertices) > sample_points:
            idx = np.random.choice(len(vertices), sample_points, replace=False)
            points = vertices[idx]
        else:
            points = vertices

    if len(points) == 0:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax in axes:
            ax.text(0.5, 0.5, "Empty", ha="center", va="center")
            ax.set_aspect("equal")
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    views = [
        ("Top (XY)", 0, 1),
        ("Front (XZ)", 0, 2),
        ("Side (YZ)", 1, 2),
    ]

    for ax, (label, dim1, dim2) in zip(axes, views):
        ax.scatter(
            points[:, dim1], points[:, dim2],
            s=0.5, alpha=0.5, c=color or "steelblue", edgecolors="none"
        )
        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")
        ax.axis("equal")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    return fig


def render_part_overview(
    meshes: List["trimesh.Trimesh"],
    infos: List[MeshInfo],
    sample_points: int = 5000,
    title: str = "",
) -> "plt.Figure":
    """Render all sub-meshes in a single 3-view plot, colored by part."""
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(meshes), 1)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    views = [
        ("Top (XY)", 0, 1),
        ("Front (XZ)", 0, 2),
        ("Side (YZ)", 1, 2),
    ]

    for ax, (label, dim1, dim2) in zip(axes, views):
        for i, (mesh, info) in enumerate(zip(meshes, infos)):
            try:
                pts, _ = mesh.sample(sample_points, return_index=True)
            except Exception:
                verts = mesh.vertices
                if len(verts) > sample_points:
                    idx = np.random.choice(len(verts), sample_points, replace=False)
                    pts = verts[idx]
                else:
                    pts = verts

            label_text = f"{info.name[:20]}"
            if info.is_background:
                label_text += " [!]"

            ax.scatter(
                pts[:, dim1], pts[:, dim2],
                s=0.3, alpha=0.4,
                c=[colors[i]], edgecolors="none",
                label=label_text,
            )
        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")

    axes[0].legend(loc="upper right", fontsize=6, ncol=2,
                   bbox_to_anchor=(1.0, -0.05))
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    return fig


def inspect_model(
    file_path: str,
    output_dir: Optional[str] = None,
    sample_points: int = 20000,
    show_plots: bool = False,
) -> Dict:
    """
    Load a glb file, extract and analyze all sub-meshes,
    flag background parts, and render visualizations.

    Returns:
        dict with inspection results and paths to saved images
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    result = {
        "file": file_path,
        "uid": os.path.splitext(os.path.basename(file_path))[0],
        "num_parts": 0,
        "parts": [],
        "flagged_count": 0,
        "flagged_indices": [],
        "images": {},
    }

    parts = load_glb_parts(file_path, sample_points)
    if parts is None:
        result["error"] = "Failed to load model"
        return result

    meshes, combined = parts
    result["num_parts"] = len(meshes)

    infos = [analyze_mesh(m, i) for i, m in enumerate(meshes)]
    infos = detect_background_parts(infos)

    volumes = [m.volume for m in infos]
    for i, info in enumerate(infos):
        result["parts"].append({
            "name": info.name,
            "index": info.index,
            "vertices": info.num_vertices,
            "faces": info.num_faces,
            "centroid": info.centroid.tolist(),
            "volume": float(info.volume),
            "flatness": float(info.flatness),
            "is_background": info.is_background,
            "is_ground_plane": info.is_ground_plane,
            "reasons": info.flag_reasons,
        })
        if info.is_background:
            result["flagged_count"] += 1
            result["flagged_indices"].append(i)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        uid = result["uid"]

        if len(meshes) > 1:
            fig = render_part_overview(
                meshes, infos, sample_points // 4,
                title=f"{uid} — {len(meshes)} parts ({result['flagged_count']} flagged)"
            )
            path = os.path.join(output_dir, f"{uid}_overview.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            result["images"]["overview"] = path

        clean_indices = [i for i in range(len(meshes)) if not infos[i].is_background]
        if len(clean_indices) > 0 and len(clean_indices) < len(meshes):
            clean_meshes = [meshes[i] for i in clean_indices]
            clean_infos = [infos[i] for i in clean_indices]
            clean_combined = trimesh.util.concatenate(clean_meshes)
            fig = render_mesh_views(
                clean_combined, sample_points,
                title=f"{uid} — Clean (filtered)",
                color="forestgreen"
            )
            path = os.path.join(output_dir, f"{uid}_clean.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            result["images"]["clean"] = path

        fig = render_mesh_views(
            combined, sample_points,
            title=f"{uid} — Full model",
            color="steelblue"
        )
        path = os.path.join(output_dir, f"{uid}_full.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        result["images"]["full"] = path

    if show_plots:
        plt.show()

    return result


def batch_inspect(
    model_dir: str,
    output_dir: str = "./inspection",
    pattern: str = "*.glb",
    max_models: Optional[int] = None,
) -> Dict:
    """
    Scan a directory of glb files and inspect each one.
    Generates an HTML summary page.

    Returns:
        dict with summary statistics and path to HTML report
    """
    import glob

    files = sorted(glob.glob(os.path.join(model_dir, pattern)))
    if max_models:
        files = files[:max_models]

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    stats = {
        "total": len(files),
        "ok": 0,
        "flagged": 0,
        "failed": 0,
        "total_parts": 0,
        "total_flagged_parts": 0,
    }

    print(f"Inspecting {len(files)} models...")
    for i, f in enumerate(files):
        if i % 20 == 0:
            print(f"  [{i}/{len(files)}]")

        result = inspect_model(f, output_dir, show_plots=False)
        all_results.append(result)

        if "error" in result:
            stats["failed"] += 1
        elif result["flagged_count"] > 0:
            stats["flagged"] += 1
            stats["total_flagged_parts"] += result["flagged_count"]
        else:
            stats["ok"] += 1
        stats["total_parts"] += result["num_parts"]

    print(f"\nDone. Results: {stats['ok']} clean, {stats['flagged']} flagged, {stats['failed']} failed")

    html_path = _generate_html_report(all_results, stats, output_dir)
    stats["html_report"] = html_path

    json_path = os.path.join(output_dir, "inspection_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    return stats


def _generate_html_report(
    results: List[Dict],
    stats: Dict,
    output_dir: str,
) -> str:
    """Generate an HTML summary page for visual inspection."""
    html_path = os.path.join(output_dir, "report.html")

    # Count per file for flagged models
    flagged_models = [r for r in results if r.get("flagged_count", 0) > 0]

    rows = []
    for r in results:
        uid = r["uid"]
        error = r.get("error", "")
        num_parts = r["num_parts"]
        flagged = r["flagged_count"]

        img_full = os.path.basename(r.get("images", {}).get("full", ""))
        img_overview = os.path.basename(r.get("images", {}).get("overview", ""))
        img_clean = os.path.basename(r.get("images", {}).get("clean", ""))

        status_class = "error" if error else ("flagged" if flagged > 0 else "ok")
        status_text = error or (f"{flagged} flagged" if flagged else "clean")

        parts_html = ""
        for p in r.get("parts", []):
            bg_class = "bg" if p["is_background"] else ""
            reasons = "<br>".join(p.get("reasons", []))
            parts_html += f"""
            <tr class="{bg_class}">
                <td>{p['name'][:30]}</td>
                <td>{p['vertices']:,}</td>
                <td>{p['volume']:.2e}</td>
                <td>{p['flatness']:.1f}</td>
                <td>{'flag' if p['is_background'] else '—'}</td>
                <td class="reasons">{reasons}</td>
            </tr>"""

        img_block = ""
        for img_type, img_path in r.get("images", {}).items():
            img_file = os.path.basename(img_path)
            img_block += f'<img src="{img_file}" class="thumb" onclick="zoom(this)" title="{uid} - {img_type}">'

        rows.append(f"""
        <div class="model-card {status_class}">
            <div class="model-header">
                <span class="uid">{uid[:40]}...</span>
                <span class="status">{status_text}</span>
                <span class="count">{num_parts} parts</span>
            </div>
            <div class="images">{img_block}</div>
            <details>
                <summary>Part details</summary>
                <table class="parts-table">
                    <tr><th>Name</th><th>Vertices</th><th>Volume</th><th>Flatness</th><th>Status</th><th>Reasons</th></tr>
                    {parts_html}
                </table>
            </details>
        </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voxer Model Inspection Report</title>
<style>
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
    h1 {{ color: #e94560; }}
    .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
    .stat {{ background: #16213e; padding: 15px 25px; border-radius: 8px; text-align: center; }}
    .stat .value {{ font-size: 2em; font-weight: bold; }}
    .stat.ok .value {{ color: #4ecca3; }}
    .stat.flagged .value {{ color: #f0a500; }}
    .stat.error .value {{ color: #e94560; }}
    .model-card {{ background: #16213e; margin: 15px 0; padding: 15px; border-radius: 8px; border-left: 4px solid #444; }}
    .model-card.ok {{ border-left-color: #4ecca3; }}
    .model-card.flagged {{ border-left-color: #f0a500; }}
    .model-card.error {{ border-left-color: #e94560; }}
    .model-header {{ display: flex; justify-content: space-between; margin-bottom: 10px; }}
    .uid {{ font-family: monospace; font-size: 0.85em; }}
    .images {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }}
    .thumb {{ max-height: 200px; border-radius: 4px; cursor: pointer; transition: transform 0.2s; }}
    .thumb:hover {{ transform: scale(1.02); }}
    .parts-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.85em; }}
    .parts-table th, .parts-table td {{ padding: 6px 10px; border-bottom: 1px solid #333; text-align: left; }}
    .parts-table .bg {{ background: rgba(240, 165, 0, 0.15); }}
    .reasons {{ color: #f0a500; font-size: 0.8em; }}
    details {{ margin: 10px 0; }}
    summary {{ cursor: pointer; color: #aaa; font-size: 0.9em; }}
    #zoomed {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 1000; text-align: center; }}
    #zoomed img {{ max-width: 90%; max-height: 90%; margin-top: 2%; }}
</style>
<script>
function zoom(el) {{
    var d = document.getElementById('zoomed');
    d.innerHTML = '<img src="' + el.src + '" onclick="this.parentElement.style.display=\\'none\\'">';
    d.style.display = 'block';
}}
</script>
</head>
<body>
<h1>Voxer Model Inspection Report</h1>
<div class="summary">
    <div class="stat"><div>Total</div><div class="value">{stats['total']}</div></div>
    <div class="stat ok"><div>Clean</div><div class="value">{stats['ok']}</div></div>
    <div class="stat flagged"><div>Flagged</div><div class="value">{stats['flagged']}</div></div>
    <div class="stat error"><div>Failed</div><div class="value">{stats['failed']}</div></div>
    <div class="stat"><div>Total Parts</div><div class="value">{stats['total_parts']}</div></div>
    <div class="stat flagged"><div>Flagged Parts</div><div class="value">{stats['total_flagged_parts']}</div></div>
</div>
<div id="zoomed"></div>
{''.join(rows)}
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path


def voxelize_filtered(
    file_path: str,
    resolution: int = 64,
    sample_points: int = 200000,
) -> Optional[np.ndarray]:
    """
    Voxelize a model while filtering out detected background parts.

    Returns:
        (resolution, resolution, resolution, 4) RGBA voxel array, or None
    """
    import trimesh

    parts = load_glb_parts(file_path, sample_points)
    if parts is None:
        return None

    meshes, _ = parts
    infos = [analyze_mesh(m, i) for i, m in enumerate(meshes)]
    infos = detect_background_parts(infos)

    clean_indices = [i for i in range(len(meshes)) if not infos[i].is_background]
    if len(clean_indices) == 0:
        return None

    if len(clean_indices) == 1:
        mesh = meshes[clean_indices[0]]
    else:
        mesh = trimesh.util.concatenate([meshes[i] for i in clean_indices])

    try:
        points, face_idx = trimesh.sample.sample_surface(mesh, sample_points)
    except Exception:
        return None

    colors = _sample_colors(mesh, points, face_idx)

    mins, maxs = points.min(axis=0), points.max(axis=0)
    scale = maxs - mins
    if np.any(scale < 1e-8):
        return None

    points_norm = (points - mins) / (scale + 1e-8)
    coords = (points_norm * (resolution - 1)).astype(int)
    coords = np.clip(coords, 0, resolution - 1)

    voxel = np.zeros((resolution, resolution, resolution, 4), dtype=np.uint8)
    for c, color in zip(coords, colors):
        voxel[c[0], c[1], c[2], :3] = color
        voxel[c[0], c[1], c[2], 3] = 255

    return voxel


def _sample_colors(mesh, points, face_idx) -> np.ndarray:
    """Sample colors from mesh at given points."""
    try:
        if (
            hasattr(mesh.visual, "material")
            and hasattr(mesh.visual.material, "baseColorFactor")
        ):
            color = np.array(mesh.visual.material.baseColorFactor[:3]) * 255
            return np.tile(color.astype(np.uint8), (len(points), 1))
    except Exception:
        pass

    try:
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            from trimesh.visual.color import vertex_to_face_color
            face_colors = vertex_to_face_color(mesh.visual.vertex_colors, mesh.faces)
            return face_colors[face_idx][:, :3].astype(np.uint8)
    except Exception:
        pass

    return np.tile(np.array([128, 128, 128], dtype=np.uint8), (len(points), 1))


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Inspect 3D models and detect background parts")
    parser.add_argument("path", help="Path to a glb file or directory of glb files")
    parser.add_argument("--output", "-o", default="./inspection",
                        help="Output directory for images and report")
    parser.add_argument("--max", "-n", type=int, default=None,
                        help="Max number of models to inspect")
    parser.add_argument("--single", "-s", action="store_true",
                        help="Inspect a single file (not batch)")
    args = parser.parse_args()

    if args.single or os.path.isfile(args.path):
        result = inspect_model(args.path, args.output, show_plots=False)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        if result.get("flagged_count", 0) > 0:
            print(f"\nFlagged {result['flagged_count']} background parts!")
            for p in result["parts"]:
                if p["is_background"]:
                    print(f"  - {p['name']}: {p['reasons']}")
    else:
        stats = batch_inspect(args.path, args.output, max_models=args.max)
        print(f"\nReport: {stats.get('html_report', 'N/A')}")
