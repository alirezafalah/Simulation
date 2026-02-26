#!/usr/bin/env python3
"""
render_engine.py — Sim2Real Binary Mask Generator for Drill Bit CAD Models
===========================================================================

Loads a single 3D CAD model (STEP or STL), aligns the drill tip at exactly
60% down from the top edge of the rendered frame, rotates the model 360°
around its Z-axis, and saves 360 binary mask images (8-bit, 0 or 255)
in PNG or TIFF format.

Camera emulates a VS-LDV75 (75 mm focal length) on a 1/2" sensor (6.4×4.8 mm)
using a standard perspective projection (NOT telecentric).

Rendering:  Pure white unlit silhouette on a pure black background.
GPU:        Hardware-accelerated OpenGL via VTK (Intel Iris Xe compatible).
CPU:        Multi-threaded image I/O with ThreadPoolExecutor.

Dependencies
------------
    pip install pyvista numpy Pillow vtk
    # For STEP loading (pick ONE):
    conda install -c conda-forge cadquery        # recommended
    # — or —
    pip install cadquery

Usage
-----
    # Render all 360 masks
    python render_engine.py -m "drills/2860A14_Carbide Drill Bit.STEP" -o output/

    # Interactive debug preview (single frame, verifies alignment)
    python render_engine.py -m "drills/2860A14_Carbide Drill Bit.STEP" --debug
"""

from __future__ import annotations

# ── Standard library ─────────────────────────────────────────────────
import os
import sys
import argparse
import time
import tempfile
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, Any

# ── Third-party ──────────────────────────────────────────────────────
import pyvista as pv
import vtk
from PIL import Image


# ═════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Physical Camera & Rendering Parameters
# ═════════════════════════════════════════════════════════════════════

# --- Camera optics (VS-LDV75 on a 1/2" sensor) ----------------------
FOCAL_LENGTH_MM: float = 75.0          # Lens focal length
SENSOR_W_MM: float     = 6.4           # 1/2" sensor width
SENSOR_H_MM: float     = 4.8           # 1/2" sensor height

# Derived fields of view (narrow, matching the telephoto lens)
VFOV_DEG: float = 2.0 * np.degrees(np.arctan(SENSOR_H_MM / (2.0 * FOCAL_LENGTH_MM)))
HFOV_DEG: float = 2.0 * np.degrees(np.arctan(SENSOR_W_MM / (2.0 * FOCAL_LENGTH_MM)))

# --- Image resolution (4:3 to match sensor aspect ratio) ------------
IMG_W: int = 1280
IMG_H: int = 960

# --- Real-world setup ------------------------------------------------
WORKING_DISTANCE_MM: float = 250.0   # Lens-to-object distance (mm)
TIP_FROM_TOP: float        = 0.60    # Drill tip positioned at 60 % from the top

# --- Multi-threading (image saving is I/O-bound) --------------------
#     Intel i-series: 8 physical / 16 logical cores.
#     Reserve ~4 for OS → 12 worker threads for parallel PNG encoding.
NUM_IO_THREADS: int = 12

# --- STEP tessellation quality ---------------------------------------
TESS_LINEAR: float  = 0.005   # Linear deviation tolerance (model units)
TESS_ANGULAR: float = 0.1     # Angular tolerance (radians ≈ 5.7°)

# --- Rotation --------------------------------------------------------
N_FRAMES: int = 360            # One frame per degree

# --- Output mask format (one-line switch) ---------------------------
# Supported values: "PNG" or "TIFF"
OUTPUT_IMAGE_FORMAT: str = "PNG"
_OUTPUT_FORMAT = OUTPUT_IMAGE_FORMAT.strip().upper()
_FORMAT_TO_EXTENSION = {"PNG": "png", "TIFF": "tiff"}
if _OUTPUT_FORMAT not in _FORMAT_TO_EXTENSION:
    raise ValueError("OUTPUT_IMAGE_FORMAT must be 'PNG' or 'TIFF'.")


# ═════════════════════════════════════════════════════════════════════
#  MESH LOADING
# ═════════════════════════════════════════════════════════════════════

def load_mesh(filepath: str) -> pv.PolyData:
    """
    Load a 3D mesh from a STEP (.step/.stp) or STL (.stl) file.

    STEP files are converted to triangulated meshes via *cadquery*
    (which wraps OpenCascade). STL files are read directly by VTK.
    """
    filepath = str(Path(filepath).resolve())
    ext = Path(filepath).suffix.lower()

    if ext == ".stl":
        print(f"[LOAD] Reading STL … {Path(filepath).name}")
        mesh = pv.read(filepath)

    elif ext in (".step", ".stp"):
        print(f"[LOAD] Reading STEP via cadquery … {Path(filepath).name}")
        try:
            import cadquery as cq
        except ImportError:
            sys.exit(
                "[ERROR] cadquery is required for STEP files.\n"
                "        Install: conda install -c conda-forge cadquery\n"
                "             or: pip install cadquery"
            )
        # Import the STEP geometry
        result = cq.importers.importStep(filepath)

        # Tessellate → temporary binary STL → PyVista mesh
        tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cq.exporters.export(
                result, tmp_path, exportType="STL",
                tolerance=TESS_LINEAR, angularTolerance=TESS_ANGULAR,
            )
            mesh = pv.read(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        sys.exit(f"[ERROR] Unsupported format '{ext}'. Use .step, .stp, or .stl.")

    print(f"[LOAD] Mesh: {mesh.n_points:,} vertices · {mesh.n_cells:,} faces")
    return mesh


def get_tool_name(model_path: str) -> str:
    """
    Extract canonical tool name from filename stem.

    Rule: keep everything before the first space.
    Example:
        "29195A079_Carbide Drill Bit.STEP" -> "29195A079_Carbide"
    """
    stem = Path(model_path).stem
    return stem.split(" ", 1)[0].strip()


# ═════════════════════════════════════════════════════════════════════
#  MESH PREPROCESSING — Centre + Tip Identification
# ═════════════════════════════════════════════════════════════════════

def preprocess_mesh(
    mesh: pv.PolyData,
) -> Tuple[pv.PolyData, float, float, float]:
    """
    Centre the mesh on the Z-axis and identify the drill tip.

    Steps
    -----
    1. Translate so the XY centroid sits on the Z-axis (X=0, Y=0).
    2. Record the tip (min Z) and top (max Z) coordinates.
    3. Compute the maximum XY radius (rotation-clearance envelope).

    Returns
    -------
    mesh            Centred mesh (modified in-place).
    tip_z           Z-coordinate of the drill tip (lowest point).
    top_z           Z-coordinate of the upper end (highest point).
    max_xy_radius   Max distance from the Z-axis across all vertices.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

    # Centre in XY so rotation axis = Z-axis
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    mesh.translate([-cx, -cy, 0.0], inplace=True)

    # Refresh bounds after centring
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

    # In this dataset the CAD tip sits at max-Z.  Flip the mesh so that
    # the tip ends up at min-Z, which is what the camera framing expects.
    mesh.points[:, 2] *= -1
    print("[TIP]  Flipped mesh Z so drill tip is at min-Z")

    # Refresh bounds after flip
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    tip_z: float = zmin
    top_z: float = zmax

    # Rotation-clearance: maximum distance of any vertex from the Z-axis
    pts = mesh.points  # (N, 3) numpy array
    max_xy_radius: float = float(np.max(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)))

    print(f"[PREP] Tip Z   = {tip_z:.4f}")
    print(f"[PREP] Top Z   = {top_z:.4f}")
    print(f"[PREP] Height  = {top_z - tip_z:.4f}")
    print(f"[PREP] Max R   = {max_xy_radius:.4f}")
    return mesh, tip_z, top_z, max_xy_radius


# ═════════════════════════════════════════════════════════════════════
#  CAMERA SETUP — Perspective Matching 75 mm Lens on 1/2" Sensor
# ═════════════════════════════════════════════════════════════════════

def compute_camera(
    tip_z: float,
    top_z: float,
    max_xy_radius: float,
) -> Dict[str, Any]:
    """
    Place the camera at the real-world working distance (250 mm) with
    the 75 mm / 1/2" sensor FOV.  Only the tip region is in frame —
    exactly like the physical VS-LDV75 setup at F4.

    Visible area at 250 mm
    ----------------------
    H_visible ≈ 16.0 mm   (vertical)
    W_visible ≈ 21.3 mm   (horizontal)

    Camera convention
    -----------------
    Camera on **−Y** looking toward **+Y**.
    World Z ↑ → image vertical, world X → image horizontal.

    Frame coordinate *f* (fraction from bottom, 0 = bottom, 1 = top):
        z(f) = z_cam − H/2 + f · H
    """
    half_vfov = np.radians(VFOV_DEG / 2.0)

    # Fixed working distance — matches the real inspection rig
    d = WORKING_DISTANCE_MM          # 250 mm

    # Visible height at that distance
    H_visible = 2.0 * d * np.tan(half_vfov)

    # Position tip at TIP_FROM_TOP fraction from the top of the frame
    # f_tip = fraction from BOTTOM
    f_tip = 1.0 - TIP_FROM_TOP       # e.g. 0.40 from bottom
    z_cam = tip_z + H_visible * (0.5 - f_tip)

    # Clipping planes
    clip_near = d * 0.01
    clip_far  = d * 10.0

    cam = dict(
        position      = (0.0, -d, z_cam),
        focal_point   = (0.0, 0.0, z_cam),
        view_up       = (0.0, 0.0, 1.0),
        view_angle    = VFOV_DEG,
        clipping_range= (clip_near, clip_far),
        distance      = d,
        H_visible     = H_visible,
    )

    print(f"[CAM]  VFOV      = {VFOV_DEG:.3f}°")
    print(f"[CAM]  HFOV      = {HFOV_DEG:.3f}°")
    print(f"[CAM]  Work Dist = {d:.1f} mm")
    print(f"[CAM]  Z centre  = {z_cam:.4f}")
    print(f"[CAM]  Visible H = {H_visible:.4f} mm  (≈ tip region only)")
    return cam


# ═════════════════════════════════════════════════════════════════════
#  IMAGE I/O
# ═════════════════════════════════════════════════════════════════════

def _to_binary_mask(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB screenshot to a strict binary mask (0 / 255)."""
    gray = np.max(rgb, axis=2)                     # any white channel → white
    return np.where(gray > 127, 255, 0).astype(np.uint8)


def _save_mask(mask: np.ndarray, path: str) -> None:
    """Write an 8-bit grayscale mask using the configured format."""
    Image.fromarray(mask, mode="L").save(path, format=_OUTPUT_FORMAT)


# ═════════════════════════════════════════════════════════════════════
#  GPU DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════

def _print_gpu_info(plotter: pv.Plotter) -> None:
    """Print the OpenGL renderer string (verifies GPU is being used)."""
    try:
        ren_win = plotter.ren_win
        ren_win.Render()                            # initialise GL context
        info = ren_win.ReportCapabilities()
        for line in info.split("\n"):
            low = line.lower()
            if "renderer" in low or "vendor" in low or "version" in low:
                print(f"[GPU]  {line.strip()}")
    except Exception:
        print("[GPU]  Could not query OpenGL capabilities.")


# ═════════════════════════════════════════════════════════════════════
#  APPLY CAMERA TO PLOTTER
# ═════════════════════════════════════════════════════════════════════

def _apply_camera(plotter: pv.Plotter, cam: Dict[str, Any]) -> None:
    """Configure the plotter camera from our parameter dict."""
    c = plotter.camera
    c.position       = cam["position"]
    c.focal_point    = cam["focal_point"]
    c.up             = cam["view_up"]
    c.view_angle     = cam["view_angle"]
    c.clipping_range = cam["clipping_range"]


# ═════════════════════════════════════════════════════════════════════
#  DEBUG MODE — Interactive Preview
# ═════════════════════════════════════════════════════════════════════

def _render_debug(mesh: pv.PolyData, cam: Dict[str, Any]) -> None:
    """
    Open an interactive PyVista window showing frame 0 for visual QA.

    Overlays:
    • Red dashed line at 60 % from top (where the tip should sit).
    • Green dashed line at 5 % from top (upper margin boundary).
    • HUD text with camera parameters.
    """
    print("[DEBUG] Opening interactive preview …")
    plotter = pv.Plotter(window_size=[IMG_W, IMG_H], title="render_engine — DEBUG")
    plotter.set_background("black")

    # Unlit white silhouette
    plotter.add_mesh(
        mesh,
        color="white",
        ambient=1.0, diffuse=0.0, specular=0.0,
        lighting=False,
        show_edges=False,
    )

    _apply_camera(plotter, cam)

    # ── Reference lines (3D lines at Z heights in the object plane) ──
    d         = cam["distance"]
    H         = cam["H_visible"]
    z_cam     = cam["focal_point"][2]
    z_bottom  = z_cam - H / 2.0
    half_w    = d * np.tan(np.radians(HFOV_DEG / 2.0))
    span      = half_w * 0.9          # slightly narrower than full frame

    # 60 % from top → f = 0.40 from bottom
    z_tip_line = z_bottom + 0.40 * H
    tip_line = pv.Line((-span, 0.0, z_tip_line), (span, 0.0, z_tip_line))
    plotter.add_mesh(tip_line, color="red", line_width=2, label="Tip line (60 % from top)")

    # 5 % from top → f = 0.95
    z_top_line = z_bottom + 0.95 * H
    top_line = pv.Line((-span, 0.0, z_top_line), (span, 0.0, z_top_line))
    plotter.add_mesh(top_line, color="lime", line_width=2, label="Top margin (5 % from top)")

    # HUD
    plotter.add_text(
        f"VFOV {VFOV_DEG:.2f}°  |  Dist {d:.1f}  |  H_vis {H:.2f}",
        position="upper_left", font_size=10, color="yellow",
    )
    plotter.add_text(
        "Red  = tip target (60 % from top)\n"
        "Green = top margin  (5 % from top)",
        position="lower_left", font_size=8, color="cyan",
    )

    _print_gpu_info(plotter)
    plotter.add_legend(bcolor=(0.1, 0.1, 0.1), border=True)
    plotter.show()


# ═════════════════════════════════════════════════════════════════════
#  PRODUCTION MODE — Off-screen 360° Rendering
# ═════════════════════════════════════════════════════════════════════

def _render_360(
    mesh: pv.PolyData,
    cam: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Render 360 binary mask frames using GPU-accelerated off-screen VTK
    and save them in parallel with a thread pool.

    Pipeline per frame
    ------------------
    1. Set actor rotation (Z-axis) via vtkTransform.     [GPU transform]
    2. Render via OpenGL.                                 [GPU rasterise]
    3. Read-back screenshot.                              [GPU → CPU]
    4. Convert to binary mask (NumPy threshold).          [CPU]
    5. Encode & save PNG (ThreadPoolExecutor).            [CPU I/O]

    Steps 4-5 run concurrently with the next frame's step 1-3.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Set up off-screen plotter ────────────────────────────────────
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=[IMG_W, IMG_H])
    plotter.set_background("black")
    plotter.disable_anti_aliasing()          # sharp edges → cleaner binary masks

    # Add mesh — unlit white silhouette
    actor = plotter.add_mesh(
        mesh,
        color="white",
        ambient=1.0, diffuse=0.0, specular=0.0,
        lighting=False,
        show_edges=False,
    )

    _apply_camera(plotter, cam)
    _print_gpu_info(plotter)

    # ── Render loop ──────────────────────────────────────────────────
    print(f"\n[RENDER] {N_FRAMES} frames @ {IMG_W}×{IMG_H} → {out.resolve()}")
    t0 = time.perf_counter()
    ext = _FORMAT_TO_EXTENSION[_OUTPUT_FORMAT]

    with ThreadPoolExecutor(max_workers=NUM_IO_THREADS) as pool:
        futures = []
        for angle in range(N_FRAMES):
            # ① Rotate mesh around Z-axis
            xform = vtk.vtkTransform()
            xform.RotateZ(float(angle))
            actor.SetUserTransform(xform)

            # ② GPU render + read-back
            plotter.render()
            rgb = plotter.screenshot(return_img=True)

            # ③ Binary mask conversion (CPU)
            mask = _to_binary_mask(rgb)

            # ④ Submit async save to thread pool
            fpath = str(out / f"mask_{angle:03d}.{ext}")
            futures.append(pool.submit(_save_mask, mask, fpath))

            # Progress every 36 frames (10 % increments)
            if (angle + 1) % 36 == 0:
                elapsed = time.perf_counter() - t0
                fps = (angle + 1) / elapsed
                print(f"         {angle + 1:3d}/{N_FRAMES}  [{fps:.1f} fps]")

        # Wait for all I/O to finish
        for f in as_completed(futures):
            f.result()                       # re-raise any exceptions

    t_total = time.perf_counter() - t0
    print(f"\n[DONE]  {N_FRAMES} masks saved in {t_total:.2f} s "
          f"({N_FRAMES / t_total:.1f} fps overall)")

    plotter.close()


# ═════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    global IMG_W, IMG_H

    parser = argparse.ArgumentParser(
        description="Sim2Real Binary Mask Generator — 360° drill silhouettes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python render_engine.py -m "drills/2860A14_Carbide.STEP"\n'
            '  python render_engine.py -m drill.stl -o masks/ --width 2048 --height 1536\n'
            '  python render_engine.py -m drill.STEP --debug\n'
        ),
    )
    parser.add_argument(
        "-m", "--model", required=True,
        help="Path to 3D model file (.step, .stp, or .stl)",
    )
    # Anchor default output to the script's own directory so the result
    # is the same regardless of where the user runs the command from.
    _SCRIPT_DIR = str(Path(__file__).resolve().parent)
    _DEFAULT_OUTPUT = str(Path(_SCRIPT_DIR) / "output")
    parser.add_argument(
        "-o", "--output", default=_DEFAULT_OUTPUT,
        help="Root output directory (default: <script_dir>/output/)"
             " — frames are saved to <output>/<tool_name>/",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Open an interactive preview window instead of batch rendering",
    )
    parser.add_argument("--width",  type=int, default=IMG_W, help=f"Image width  (default {IMG_W})")
    parser.add_argument("--height", type=int, default=IMG_H, help=f"Image height (default {IMG_H})")
    args = parser.parse_args()

    # Allow runtime resolution override
    IMG_W = args.width
    IMG_H = args.height
    tool_name = get_tool_name(args.model)
    output_dir = str(Path(args.output) / tool_name)

    # ── Pipeline ─────────────────────────────────────────────────────
    print("=" * 62)
    print("  render_engine.py — Sim2Real Binary Mask Generator")
    print("=" * 62)
    print(f"  Model  : {args.model}")
    print(f"  Tool   : {tool_name}")
    print(f"  Output : {output_dir}")
    print(f"  Size   : {IMG_W} × {IMG_H}")
    print(f"  Debug  : {args.debug}")
    print(f"  Threads: {NUM_IO_THREADS}")
    print(f"  VFOV   : {VFOV_DEG:.3f}°  (75 mm on 1/2\" sensor)")
    print("=" * 62)

    # 1 — Load
    mesh = load_mesh(args.model)

    # 2 — Pre-process
    mesh, tip_z, top_z, max_r = preprocess_mesh(mesh)

    # 3 — Camera
    cam = compute_camera(tip_z, top_z, max_r)

    # 4 — Render
    if args.debug:
        _render_debug(mesh, cam)
    else:
        _render_360(mesh, cam, output_dir)

    print("[EXIT]")


if __name__ == "__main__":
    main()
