#!/usr/bin/env python3
"""
noise_injector.py — Localized Contour-Dent Noise for Binary Sim Masks
======================================================================

Takes a perfectly clean binary mask (numpy uint8, 0/255) rendered by
``render_engine.py`` and degrades it to look like it was segmented via
background removal from a real camera frame.

Real-world artefact being simulated
------------------------------------
Background-subtraction segmentation sometimes slightly under-segments:
the mask boundary sits a few pixels *inside* the true tool edge.  This
does NOT happen uniformly — it occurs in **localised patches** of
~100-200 contour pixels, creating small smooth dents/erosions along the
edge.  When these masks are used for 3-D reconstruction the dents
produce small surface indentations that gradually fade out.

Noise Pipeline
--------------
    1. Find the main contour of the binary mask.
    2. Pick ``n_dents`` random contiguous segments along the contour,
       each spanning ``dent_span_min``–``dent_span_max`` contour pixels.
    3. For each segment, assign a random erosion depth (1–``max_depth``
       pixels) with a smooth cosine-window falloff at both ends.
    4. Use ``cv2.distanceTransform`` to determine each foreground
       pixel's distance from the boundary.
    5. Erode (set to 0) every foreground pixel whose distance to the
       edge is **less than** the local dent depth.

The solid interior and background are untouched.  The result is a
plausible partially-under-segmented mask.

Performance
-----------
All operations are vectorised NumPy / OpenCV on uint8 arrays.
Batch mode uses ``concurrent.futures.ThreadPoolExecutor``.

Dependencies
------------
    pip install numpy opencv-python Pillow

Usage
-----
    # ── As a library ─────────────────────────────────────────────
    from noise_injector import inject_noise, inject_noise_batch

    noisy = inject_noise(clean_mask)                    # defaults
    noisy = inject_noise(clean_mask,                    # custom
                         n_dents=4,
                         dent_span_min=80,
                         dent_span_max=250,
                         max_depth=3.0,
                         seed=42)

    inject_noise_batch(input_dir, output_dir)

    # ── Standalone CLI ───────────────────────────────────────────
    python noise_injector.py -i output/2396N63_Carbide
    python noise_injector.py -i output/2396N63_Carbide --n-dents 5 --max-depth 3
"""

from __future__ import annotations

# ── Standard library ─────────────────────────────────────────────────
import argparse
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ── Third-party ──────────────────────────────────────────────────────
import cv2
from PIL import Image


# ═════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Default Noise Parameters
# ═════════════════════════════════════════════════════════════════════

N_DENTS: int         = 3        # Number of localised erosion patches.
DENT_SPAN_MIN: int   = 80       # Minimum contour-pixel span per dent.
DENT_SPAN_MAX: int   = 250      # Maximum contour-pixel span per dent.
MAX_DEPTH: float     = 2.5      # Maximum inward erosion depth (pixels).
                                 # Each dent picks a random depth in
                                 # [1.0, max_depth].

# --- Output format (matches render_engine convention) ----------------
OUTPUT_IMAGE_FORMAT: str = "PNG"

# --- Threading (matches render_engine) --------------------------------
NUM_IO_THREADS: int  = 12


# ═════════════════════════════════════════════════════════════════════
#  CORE — Single-Mask Noise Injection
# ═════════════════════════════════════════════════════════════════════

def inject_noise(
    mask: np.ndarray,
    *,
    n_dents: int             = N_DENTS,
    dent_span_min: int       = DENT_SPAN_MIN,
    dent_span_max: int       = DENT_SPAN_MAX,
    max_depth: float         = MAX_DEPTH,
    seed: Optional[int]      = None,
) -> np.ndarray:
    """
    Inject localised contour-erosion dents into a clean binary mask,
    simulating real background-subtraction under-segmentation.

    Parameters
    ----------
    mask : np.ndarray
        Input binary image (uint8, values 0 and 255 only).
    n_dents : int
        Number of independent erosion patches along the contour.
    dent_span_min / dent_span_max : int
        Range (in contour pixels) for each dent's arc length.
    max_depth : float
        Maximum inward erosion in pixels.  Each dent picks a random
        depth in [1.0, max_depth].  Values < 1 effectively disable
        denting.
    seed : int or None
        RNG seed for reproducibility.  None → non-deterministic.

    Returns
    -------
    np.ndarray
        Degraded binary mask (uint8, 0 / 255), same shape as input.
    """
    assert mask.ndim == 2, "Expected a 2-D (H, W) array"
    assert mask.dtype == np.uint8, "Expected uint8 dtype"

    if n_dents <= 0 or max_depth < 0.5:
        return mask.copy()

    rng = np.random.default_rng(seed)
    out = mask.copy()

    # ── 1. Find the main contour ─────────────────────────────────────
    contours, _ = cv2.findContours(
        out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return out
    contour = max(contours, key=cv2.contourArea)
    # contour shape: (N, 1, 2)  →  squeeze to (N, 2)
    pts = contour.squeeze(axis=1)          # [[x0,y0], [x1,y1], …]
    n_pts = len(pts)
    if n_pts < 20:                         # too small to dent
        return out

    # ── 2. Distance transform (how far each FG pixel is from edge) ───
    dist = cv2.distanceTransform(out, cv2.DIST_L2, 5)

    # ── 3. Build a dent-depth map ────────────────────────────────────
    #    For each dent: pick a random start on the contour, a random
    #    span, and a random depth.  Paint a cosine-windowed depth
    #    profile along those contour points.
    dent_map = np.zeros(mask.shape, dtype=np.float32)

    for _ in range(n_dents):
        span = rng.integers(dent_span_min, dent_span_max + 1)
        span = min(span, n_pts)            # clamp to contour length
        start = rng.integers(0, n_pts)
        depth = rng.uniform(1.0, max(1.0, max_depth))

        # Cosine window: 0 at edges, *depth* at centre
        t = np.linspace(0, np.pi, span)
        weights = (1.0 - np.cos(t)) / 2.0 * depth   # 0 → depth → 0

        # Paint dent depth along contour points.  We paint a small
        # filled circle at each contour point so the dent zone has
        # width (otherwise single-pixel lines leave gaps).
        radius = max(1, int(np.ceil(depth)))
        for k in range(span):
            idx = (start + k) % n_pts
            cx, cy = int(pts[idx, 0]), int(pts[idx, 1])
            # Use cv2.circle to paint smoothly (filled)
            cv2.circle(dent_map, (cx, cy), radius,
                       float(weights[k]), thickness=-1)

    # Smooth the dent map so transitions are gradual, not pixelated
    if dent_map.max() > 0:
        ksize = max(3, (int(np.ceil(max_depth)) * 2) | 1)
        dent_map = cv2.GaussianBlur(dent_map, (ksize, ksize), sigmaX=1.0)

    # ── 4. Erode: remove FG pixels where dist < dent_depth ───────────
    erode_mask = (dist > 0) & (dist < dent_map)
    out[erode_mask] = 0

    return out


# ═════════════════════════════════════════════════════════════════════
#  BATCH — Multi-threaded Folder Processing
# ═════════════════════════════════════════════════════════════════════

def _process_one(
    src: Path,
    dst: Path,
    n_dents: int,
    dent_span_min: int,
    dent_span_max: int,
    max_depth: float,
    seed: Optional[int],
) -> str:
    """Load → inject noise → save.  Returns the destination path string."""
    img = np.array(Image.open(src).convert("L"), dtype=np.uint8)
    img = np.where(img > 127, 255, 0).astype(np.uint8)
    noisy = inject_noise(
        img,
        n_dents=n_dents,
        dent_span_min=dent_span_min,
        dent_span_max=dent_span_max,
        max_depth=max_depth,
        seed=seed,
    )
    Image.fromarray(noisy, mode="L").save(str(dst), format=OUTPUT_IMAGE_FORMAT)
    return str(dst)


def inject_noise_batch(
    input_dir: str,
    output_dir: str,
    *,
    n_dents: int             = N_DENTS,
    dent_span_min: int       = DENT_SPAN_MIN,
    dent_span_max: int       = DENT_SPAN_MAX,
    max_depth: float         = MAX_DEPTH,
    seed: Optional[int]      = None,
    num_threads: int         = NUM_IO_THREADS,
) -> None:
    """
    Apply noise injection to every PNG/TIFF mask in *input_dir* and
    write results to *output_dir* using a thread pool.

    Parameters match ``inject_noise``; see its docstring for details.
    """
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [f for f in inp.iterdir()
         if f.suffix.lower() in (".png", ".tiff", ".tif")],
    )
    if not files:
        print(f"[NOISE] No image files found in {inp}")
        return

    print(f"[NOISE] {len(files)} masks  |  dents={n_dents}  "
          f"span={dent_span_min}–{dent_span_max}  "
          f"max_depth={max_depth:.1f} px")
    print(f"[NOISE] {inp}  →  {out}")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {}
        for f in files:
            dst = out / f.name
            fut = pool.submit(
                _process_one, f, dst,
                n_dents, dent_span_min, dent_span_max,
                max_depth, seed,
            )
            futures[fut] = f.name

        done = 0
        for fut in as_completed(futures):
            fut.result()  # propagate exceptions
            done += 1
            if done % 36 == 0 or done == len(files):
                elapsed = time.perf_counter() - t0
                fps = done / elapsed
                print(f"         {done:3d}/{len(files)}  [{fps:.1f} img/s]")

    t_total = time.perf_counter() - t0
    print(f"\n[NOISE] Done — {len(files)} masks in {t_total:.2f} s "
          f"({len(files) / t_total:.1f} img/s)")


# ═════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    _SCRIPT_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Inject localised contour-dent noise into clean binary masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python noise_injector.py -i output/2396N63_Carbide\n'
            '  python noise_injector.py -i output/2396N63_Carbide -o output/noisy\n'
            '  python noise_injector.py -i output/2396N63_Carbide --n-dents 5 --max-depth 3\n'
        ),
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Directory containing clean binary mask images",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory (default: <input>_noisy)",
    )
    parser.add_argument(
        "--n-dents", type=int, default=N_DENTS,
        help=f"Number of localised erosion patches (default {N_DENTS})",
    )
    parser.add_argument(
        "--span-min", type=int, default=DENT_SPAN_MIN,
        help=f"Minimum dent arc length in contour pixels (default {DENT_SPAN_MIN})",
    )
    parser.add_argument(
        "--span-max", type=int, default=DENT_SPAN_MAX,
        help=f"Maximum dent arc length in contour pixels (default {DENT_SPAN_MAX})",
    )
    parser.add_argument(
        "--max-depth", type=float, default=MAX_DEPTH,
        help=f"Maximum erosion depth in pixels (default {MAX_DEPTH})",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducible noise (default: non-deterministic)",
    )
    parser.add_argument(
        "--threads", type=int, default=NUM_IO_THREADS,
        help=f"I/O thread count (default {NUM_IO_THREADS})",
    )
    args = parser.parse_args()

    # Resolve input/output (anchored to script dir if relative)
    inp = Path(args.input)
    if not inp.is_absolute():
        inp = _SCRIPT_DIR / inp

    if args.output is None:
        out = inp.parent / (inp.name + "_noisy")
    else:
        out = Path(args.output)
        if not out.is_absolute():
            out = _SCRIPT_DIR / out

    print("=" * 62)
    print("  noise_injector.py — Contour-Dent Noise Injector")
    print("=" * 62)
    print(f"  Input    : {inp}")
    print(f"  Output   : {out}")
    print(f"  Dents    : {args.n_dents}")
    print(f"  Span     : {args.span_min}–{args.span_max} contour px")
    print(f"  Depth    : {args.max_depth:.1f} px")
    print(f"  Seed     : {args.seed}")
    print(f"  Threads  : {args.threads}")
    print("=" * 62)

    inject_noise_batch(
        str(inp), str(out),
        n_dents=args.n_dents,
        dent_span_min=args.span_min,
        dent_span_max=args.span_max,
        max_depth=args.max_depth,
        seed=args.seed,
        num_threads=args.threads,
    )

    print("[EXIT]")


if __name__ == "__main__":
    main()
