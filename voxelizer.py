#!/usr/bin/env python3
"""
voxelizer.py — CAD-to-Voxel Ground Truth Generator
====================================================

Loads CAD models (.step / .stp / .stl), aligns them with the exact
same spatial framing as **render_engine.py**, and voxelises into
boolean 128×128×128 grids conforming to ``voxel_grid_spec.json``.

Pipeline per tool
-----------------
1.  Load mesh (cadquery for STEP, VTK for STL).
2.  Centre XY on rotation axis, flip Z (tip → min-Z).
    *Identical to* ``render_engine.preprocess_mesh``.
3.  Translate so the drill tip sits at the 80 % mark (from the top)
    of the global bounding volume ([-10, 10] mm on each axis).
4.  Point-in-mesh test on N³ voxel centres (VTK ray-casting via
    ``vtkSelectEnclosedPoints``).
5.  Save boolean ``.npz`` (voxel_grid, volume_bounds, grid_shape).

The CAD model is typically taller than the 20 mm global cube — the
point-in-mesh test naturally crops the upper shank.  This is desired.

Multi-tool batches use ``concurrent.futures.ProcessPoolExecutor``
for CPU-parallel processing.

Coordinate mapping  (mesh → voxel grid)
---------------------------------------
    mesh X   →  voxel X  (lateral, left–right)
    mesh Z   →  voxel Y  (axial, along tool shaft — up in image)
   −mesh Y   →  voxel Z  (depth, front–back; matches Visual Hull convention)

Dependencies
------------
    pip install pyvista numpy vtk
    # For STEP files:
    conda install -c conda-forge cadquery

Usage
-----
    # Single tool
    python voxelizer.py -m drills/tool.STEP -s spec.json -o output_voxels/

    # Batch (all STEP/STL in a folder)
    python voxelizer.py --batch drills/ -s spec.json -o output_voxels/ -j 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyvista as pv
import vtk

try:
    from vtkmodules.util.numpy_support import vtk_to_numpy
except ImportError:
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore[import]


# ═════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════

TESS_LINEAR:  float = 0.005       # STEP → STL linear tolerance (mm)
TESS_ANGULAR: float = 0.1         # STEP → STL angular tolerance (rad)
TIP_FROM_TOP: float = 0.80        # Matches render_engine.TIP_FROM_TOP
CAD_EXTENSIONS       = {".step", ".stp", ".stl"}

# Camera constants (must match render_engine.py exactly)
_FOCAL_LENGTH_MM:      float = 75.0
_SENSOR_H_MM:          float = 4.8
_WORKING_DISTANCE_MM:  float = 250.0
_VFOV_RAD:             float = 2.0 * np.arctan(_SENSOR_H_MM / (2.0 * _FOCAL_LENGTH_MM))
_H_VISIBLE_MM:         float = 2.0 * _WORKING_DISTANCE_MM * np.tan(_VFOV_RAD / 2.0)  # ≈16.0 mm

_SCRIPT_DIR = Path(__file__).resolve().parent


# ═════════════════════════════════════════════════════════════════════
#  SPEC LOADER
# ═════════════════════════════════════════════════════════════════════

def load_spec(spec_path: str | Path) -> Dict[str, Any]:
    """Read and return the parsed ``voxel_grid_spec.json``."""
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_spec(spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract grid shape and volume bounds from a loaded spec dict.

    Returns
    -------
    grid_shape    : ndarray, shape (3,), int32    — e.g. [128, 128, 128]
    volume_bounds : ndarray, shape (3, 2), float32 — e.g. [[-10,10],…]
    """
    gs = np.array(spec["grid_shape"], dtype=np.int32)
    vb = np.array(spec["global_volume_bounds_mm"], dtype=np.float32)
    return gs, vb


# ═════════════════════════════════════════════════════════════════════
#  MESH LOADING  (mirrors render_engine.load_mesh exactly)
# ═════════════════════════════════════════════════════════════════════

def load_mesh(filepath: str) -> pv.PolyData:
    """Load a STEP or STL file into a PyVista PolyData mesh."""
    fp = str(Path(filepath).resolve())
    ext = Path(fp).suffix.lower()

    if ext == ".stl":
        return pv.read(fp)

    if ext in (".step", ".stp"):
        try:
            import cadquery as cq
        except ImportError:
            raise ImportError(
                "cadquery is required for STEP files.\n"
                "Install:  conda install -c conda-forge cadquery"
            )
        result = cq.importers.importStep(fp)
        tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cq.exporters.export(
                result, tmp_path, exportType="STL",
                tolerance=TESS_LINEAR, angularTolerance=TESS_ANGULAR,
            )
            return pv.read(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    raise ValueError(f"Unsupported format '{ext}'.  Use .step, .stp, or .stl.")


def get_tool_name(model_path: str) -> str:
    """Canonical tool name from filename stem (same rule as render_engine)."""
    import re as _re

    stem = Path(model_path).stem
    # Strip copy suffixes like " (1)"
    stem = _re.sub(r'\s*\(\d+\)\s*$', '', stem)

    # --- End Mill --------------------------------------------------------
    if _re.search(r'End\s+Mill', stem, _re.IGNORECASE):
        flute_match = _re.search(r'(\d+)-Flute', stem, _re.IGNORECASE)
        name = _re.sub(r'\s*\d+-Flute\s*', ' ', stem).strip() if flute_match else stem
        em = _re.search(r'End\s+Mill', name, _re.IGNORECASE)
        if em:
            name = name[:em.end()]
        name = name.replace(' ', '_')
        if flute_match:
            return f"{flute_match.group(1)}_Flute_{name}"
        return name

    # --- Drill Bit (or any "…Bit") ----------------------------------------
    drill_match = _re.search(r'Drill\s+Bit', stem, _re.IGNORECASE)
    if drill_match:
        return stem[:drill_match.end()].replace(' ', '_')

    bit_match = _re.search(r'\bBit\b', stem, _re.IGNORECASE)
    if bit_match:
        return stem[:bit_match.end()].replace(' ', '_')

    # --- Fallback ---------------------------------------------------------
    return stem.split(" ", 1)[0].strip()


# ═════════════════════════════════════════════════════════════════════
#  ALIGNMENT  (mirrors render_engine.preprocess_mesh + camera framing)
# ═════════════════════════════════════════════════════════════════════

def align_mesh(
    mesh: pv.PolyData,
    volume_bounds: np.ndarray,
    tip_from_top: float = TIP_FROM_TOP,
) -> Tuple[pv.PolyData, Dict[str, float]]:
    """
    Align mesh identically to ``render_engine.preprocess_mesh``, then
    translate so the drill tip sits at *tip_from_top* fraction from the
    top of the global bounding volume's axial axis.

    Steps  (1–2 are **identical** to render_engine)
    -----
    1. Centre XY on the rotation axis  (X = 0, Y = 0).
    2. Flip Z so the drill tip is at min-Z.
    3. Translate mesh Z so the tip sits at the target axial position
       inside the global bounding volume.

    Parameters
    ----------
    mesh          : The raw loaded mesh (modified in-place).
    volume_bounds : shape (3, 2) — global [min, max] per axis.
    tip_from_top  : 0.80 means the tip is at 80 % from the top
                    of the axial extent (matching render_engine).

    Returns
    -------
    mesh : The aligned mesh.
    info : Dict with alignment diagnostics.
    """
    # ── Step 1: Centre XY (rotation-axis alignment) ──────────────────
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    mesh.translate([-cx, -cy, 0.0], inplace=True)

    # ── Step 2: Flip Z so drill tip sits at min-Z ───────────────────
    mesh.points[:, 2] *= -1

    # ── Step 3: Translate Z for tip position in global volume ────────
    #
    # The tip position must match where the Visual Hull reconstruction
    # would place it.  The VH engine estimates cy (vertical centre in
    # pixels) as the midpoint of the visible tool extent.  We replicate
    # that camera-framing math here so both outputs agree.
    #
    #   render_engine camera:  tip at 'tip_from_top' fraction from the
    #   top of a visible window of height H_visible ≈ 16 mm.
    #
    #   frac_top_tool = 1 − tip_from_top − tool_height / H_visible
    #   (negative ⟹ tool extends past the top of the frame → clip to 0)
    #
    #   VH cy estimation:  cy = (frac_top_visible + tip_from_top) / 2
    #   ⟹  tip_Y  = H_visible × (frac_top_visible − tip_from_top) / 2
    #
    _, _, _, _, zmin, zmax = mesh.bounds
    tip_z = zmin
    tool_height = zmax - zmin

    # Where does the tool's top edge appear in the camera frame?
    frac_top_tool = 1.0 - tip_from_top - tool_height / _H_VISIBLE_MM
    frac_top_visible = max(0.0, frac_top_tool)   # clip if past frame

    # Replicate VH silhouette-analysis cy → derive the tip's grid-Y
    # For almost all tools (height > ~3.2 mm): target ≈ −6.4 mm
    target_tip = _H_VISIBLE_MM * (frac_top_visible - tip_from_top) / 2.0

    offset_z   = target_tip - tip_z
    mesh.translate([0.0, 0.0, offset_z], inplace=True)

    info = dict(
        tip_target_mm=target_tip,
        tool_height_mm=tool_height,
        frac_top_visible=frac_top_visible,
        n_vertices=mesh.n_points,
        n_faces=mesh.n_cells,
    )
    return mesh, info


# ═════════════════════════════════════════════════════════════════════
#  VOXELIZATION  — Point-in-Mesh via VTK ray-casting
# ═════════════════════════════════════════════════════════════════════

def _make_query_grid(
    grid_shape: np.ndarray,
    volume_bounds: np.ndarray,
) -> np.ndarray:
    """
    Build N³ query points whose coordinates are in the **mesh**
    coordinate system.

    Axis mapping
    ------------
    query axis 0  (mesh X)  →  voxel X   :  volume_bounds[0]
    query axis 1  (mesh Y)  →  voxel Z   :  volume_bounds[2]  (flipped in voxelize())
    query axis 2  (mesh Z)  →  voxel Y   :  volume_bounds[1]

    Returns shape (N³, 3) float64.
    """
    N = int(grid_shape[0])

    def _centres(lo: float, hi: float, n: int) -> np.ndarray:
        step = (hi - lo) / n
        return np.linspace(lo + step / 2, hi - step / 2, n)

    xs = _centres(*volume_bounds[0], N)       # mesh X → voxel X
    ys = _centres(*volume_bounds[2], N)       # mesh Y → voxel Z
    zs = _centres(*volume_bounds[1], N)       # mesh Z → voxel Y

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def voxelize(
    mesh: pv.PolyData,
    grid_shape: np.ndarray,
    volume_bounds: np.ndarray,
) -> np.ndarray:
    """
    Boolean N×N×N voxel grid via VTK's ``vtkSelectEnclosedPoints``
    (optimised ray-based inside / outside classification).

    Returns
    -------
    ndarray, shape ``grid_shape``, dtype bool.
    Axes = (voxel_X, voxel_Y, voxel_Z).
    """
    N = int(grid_shape[0])

    # ── Build query point cloud ──────────────────────────────────────
    query_pts = _make_query_grid(grid_shape, volume_bounds).astype(np.float64)
    cloud = pv.PolyData(query_pts)

    # ── Prepare mesh for robust inside / outside test ────────────────
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    # Repair open edges (non-watertight meshes)
    n_open = mesh.n_open_edges
    if n_open > 0:
        warnings.warn(
            f"Mesh has {n_open} open edges — attempting fill_holes …")
        mesh = mesh.fill_holes(hole_size=1e6)

    # Consistent outward-facing normals (critical for ray test)
    mesh = mesh.compute_normals(auto_orient_normals=True, inplace=False)

    # ── VTK enclosed-points test ─────────────────────────────────────
    sel = vtk.vtkSelectEnclosedPoints()
    sel.SetInputData(cloud)
    sel.SetSurfaceData(mesh)
    sel.SetTolerance(1e-6)
    sel.Update()

    inside = vtk_to_numpy(
        sel.GetOutput().GetPointData().GetArray("SelectedPoints")
    )

    # ── Reshape & transpose to voxel ordering ────────────────────────
    #  Meshgrid was created indexing='ij' with axes
    #     0: mesh X  →  voxel X
    #     1: mesh Y  →  voxel Z
    #     2: mesh Z  →  voxel Y
    #  Transpose (0, 2, 1) → (voxel X, voxel Y, voxel Z=+mesh_Y)
    inside_3d = inside.reshape(N, N, N).astype(bool)
    voxel_grid = np.transpose(inside_3d, (0, 2, 1))

    # ── Flip depth axis to match Visual Hull convention ──────────────
    #  The VH defines depth as VH_Z = −mesh_Y (camera sits on −Y
    #  looking toward +Y at θ=0).  After transpose voxel Z increases
    #  with +mesh_Y; flipping axis 2 corrects the direction so that
    #  voxel_Z index 0 → −10 mm and index N−1 → +10 mm in VH_Z.
    voxel_grid = voxel_grid[:, :, ::-1].copy()

    return voxel_grid


# ═════════════════════════════════════════════════════════════════════
#  NPZ I/O
# ═════════════════════════════════════════════════════════════════════

def save_npz(
    voxel_grid: np.ndarray,
    volume_bounds: np.ndarray,
    grid_shape: np.ndarray,
    path: str,
) -> None:
    """Save voxel grid as compressed ``.npz`` per the spec."""
    np.savez_compressed(
        path,
        voxel_grid=voxel_grid,
        volume_bounds=volume_bounds.astype(np.float32),
        grid_shape=grid_shape.astype(np.int32),
    )


def save_obj(
    voxel_grid: np.ndarray,
    volume_bounds: np.ndarray,
    grid_shape: np.ndarray,
    path: str,
) -> None:
    """
    Convert a boolean voxel grid to a surface mesh and save as
    Wavefront ``.obj``.

    Uses PyVista's ``ImageData`` + ``contour`` (marching cubes) to
    produce a smooth isosurface at the 0.5 threshold.  Falls back to
    a blocky extraction if marching cubes yields an empty mesh (e.g.
    very sparse grids).
    """
    N = int(grid_shape[0])
    x_lo, x_hi = volume_bounds[0]
    y_lo, y_hi = volume_bounds[1]
    z_lo, z_hi = volume_bounds[2]
    sx = (x_hi - x_lo) / N
    sy = (y_hi - y_lo) / N
    sz = (z_hi - z_lo) / N

    grid = pv.ImageData(
        dimensions=(N + 1, N + 1, N + 1),
        spacing=(sx, sy, sz),
        origin=(float(x_lo), float(y_lo), float(z_lo)),
    )
    grid.cell_data["voxels"] = voxel_grid.ravel(order="F").astype(np.float32)

    # Marching cubes on cell data → smooth surface
    surface = grid.cell_data_to_point_data().contour(
        isosurfaces=[0.5], scalars="voxels"
    )
    if surface.n_points == 0:
        # Fallback: extract occupied cells as surface
        surface = grid.threshold(0.5, scalars="voxels")
        surface = surface.extract_surface()

    if surface.n_points == 0:
        warnings.warn("Voxel grid is empty — OBJ file not written.")
        return

    pv.save_meshio(path, surface, file_format="obj")


# ═════════════════════════════════════════════════════════════════════
#  SINGLE-TOOL PIPELINE  (fully picklable for ProcessPoolExecutor)
# ═════════════════════════════════════════════════════════════════════

def voxelize_single_tool(
    model_path: str,
    output_dir: str,
    spec_path: str,
    tip_from_top: float = TIP_FROM_TOP,
    export_obj: bool = False,
) -> Dict[str, Any]:
    """
    Full pipeline for one CAD file  →  ``.npz``.

    All arguments are primitive types (str / float) so this function
    is safely picklable and can be dispatched to a
    ``ProcessPoolExecutor`` worker.

    Returns
    -------
    dict  with keys:
        tool_name, output_path, success, message, elapsed_s, n_occupied
    """
    t0 = time.perf_counter()
    tool_name = get_tool_name(model_path)

    try:
        spec = load_spec(spec_path)
        gs, vb = parse_spec(spec)

        mesh = load_mesh(model_path)
        mesh, info = align_mesh(mesh, vb, tip_from_top)

        vg = voxelize(mesh, gs, vb)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = str(out_dir / f"{tool_name}_voxels.npz")

        save_npz(vg, vb, gs, npz_path)

        obj_path = ""
        if export_obj:
            obj_path = str(out_dir / f"{tool_name}_voxels.obj")
            save_obj(vg, vb, gs, obj_path)

        n_occ = int(vg.sum())
        pct   = 100.0 * n_occ / vg.size
        dt    = time.perf_counter() - t0

        fmt_extra = f" + OBJ" if export_obj else ""
        return dict(
            tool_name=tool_name,
            output_path=npz_path,
            obj_path=obj_path,
            success=True,
            message=(
                f"{tool_name}: {n_occ:,} voxels occupied "
                f"({pct:.1f} %){fmt_extra} in {dt:.1f} s"
            ),
            elapsed_s=dt,
            n_occupied=n_occ,
        )

    except Exception as exc:
        dt = time.perf_counter() - t0
        return dict(
            tool_name=tool_name,
            output_path="",
            success=False,
            message=f"{tool_name}: FAILED — {exc}",
            elapsed_s=dt,
            n_occupied=0,
        )


# ═════════════════════════════════════════════════════════════════════
#  BATCH  (convenience wrapper used by GUI & CLI)
# ═════════════════════════════════════════════════════════════════════

def collect_cad_files(folder: str | Path) -> List[Path]:
    """Return sorted list of CAD files in *folder* and its subdirectories.

    Supports a parent folder containing tool-type subdirectories
    (e.g. ``drills/`` and ``end_mills/``) or a flat folder of CAD files.
    """
    return sorted(
        p for p in Path(folder).rglob('*')
        if p.is_file() and p.suffix.lower() in CAD_EXTENSIONS
    )


# ═════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="CAD → Voxel Ground Truth Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python voxelizer.py -m drills/tool.STEP -s spec.json\n'
            '  python voxelizer.py --batch drills/ -s spec.json -o voxels/ -j 6\n'
        ),
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("-m", "--model", help="Single CAD file")
    grp.add_argument("--batch", metavar="DIR", help="Folder of CAD files")
    ap.add_argument(
        "-s", "--spec", required=True,
        help="Path to voxel_grid_spec.json",
    )
    ap.add_argument(
        "-o", "--output", default=str(_SCRIPT_DIR / "output_voxels"),
        help="Output directory for .npz files",
    )
    ap.add_argument(
        "-j", "--jobs", type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel worker processes (default: half of CPU cores)",
    )
    ap.add_argument(
        "--tip", type=float, default=TIP_FROM_TOP,
        help=f"Tip position (fraction from top, default {TIP_FROM_TOP})",
    )
    ap.add_argument(
        "--obj", action="store_true", default=False,
        help="Also export each voxel grid as a Wavefront .obj mesh",
    )
    args = ap.parse_args()

    if args.model:
        # ── Single-tool mode ─────────────────────────────────────────
        r = voxelize_single_tool(
            args.model, args.output, args.spec, args.tip,
            export_obj=args.obj,
        )
        print(r["message"])
        sys.exit(0 if r["success"] else 1)
    else:
        # ── Batch mode ───────────────────────────────────────────────
        files = collect_cad_files(args.batch)
        if not files:
            sys.exit(f"No CAD files found in {args.batch}")

        print(f"Found {len(files)} CAD file(s).  "
              f"Processing with {args.jobs} worker(s) …\n")
        t_start = time.perf_counter()

        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    voxelize_single_tool,
                    str(f), args.output, args.spec, args.tip,
                    export_obj=args.obj,
                ): f
                for f in files
            }
            ok = 0
            for fut in as_completed(futures):
                r = fut.result()
                print(f"  {'✓' if r['success'] else '✗'}  {r['message']}")
                ok += r["success"]

        dt = time.perf_counter() - t_start
        print(f"\nDone — {ok}/{len(files)} succeeded in {dt:.1f} s total.")


if __name__ == "__main__":
    main()
