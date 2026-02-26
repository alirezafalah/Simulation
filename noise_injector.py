#!/usr/bin/env python3
"""
noise_injector.py — Realistic Edge-Noise Injector for Binary Sim Masks
========================================================================

Takes a perfectly clean binary mask (numpy uint8, 0/255) rendered by
``render_engine.py`` and degrades it to look like it was segmented from
a real camera frame captured on a metallic drill under a VS-LDV75 lens.

Noise Pipeline (all applied **only** within a narrow band around the
silhouette boundary — the solid interior and background are untouched):

    1. **Edge aliasing**   — Gaussian blur on the boundary zone followed
       by re-thresholding, producing sub-pixel jaggedness that mimics a
       real sensor grid.

    2. **Specular-flip noise** — Random pixel flips (white↔black) along
       the boundary zone to simulate segmentation failures caused by
       shiny metallic reflections.  Fully configurable: can be turned
       off, or dialled from subtle to aggressive.

Performance
-----------
All operations are vectorised NumPy / OpenCV on uint8 arrays.
Batch mode uses ``concurrent.futures.ThreadPoolExecutor`` with 12
workers (tuned for an 8P/16L Intel i-series CPU, reserving 4 threads
for OS / GPU).

Dependencies
------------
    pip install numpy opencv-python Pillow

Usage
-----
    # ── As a library (from render_engine or any script) ──────────
    from noise_injector import inject_noise, inject_noise_batch

    noisy = inject_noise(clean_mask)                 # default params
    noisy = inject_noise(clean_mask,                 # custom
                         edge_width=5,
                         blur_sigma=1.0,
                         flip_probability=0.03,
                         flip_enabled=True,
                         seed=42)

    inject_noise_batch(input_dir, output_dir)        # all PNGs in folder

    # ── Standalone CLI ───────────────────────────────────────────
    python noise_injector.py -i output/2396N63_Carbide -o output/2396N63_Carbide_noisy
    python noise_injector.py -i output/2396N63_Carbide --edge-width 7 --flip-off
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

# --- Edge detection / boundary band ----------------------------------
EDGE_WIDTH: int      = 5         # Morphological dilation radius (pixels)
                                 # that defines the "boundary zone".
                                 # Larger → wider noisy border.

# --- Stage 1: Blur + re-threshold (aliasing) -------------------------
BLUR_SIGMA: float    = 1.0      # Gaussian sigma for edge blur.
                                 # 0.5–1.0 = subtle jaggedness,
                                 # 1.5–2.5 = visible softening.

# --- Stage 2: Specular-flip noise ------------------------------------
FLIP_ENABLED: bool   = True     # Master switch for flip noise.
FLIP_PROB: float     = 0.03     # Probability of flipping each boundary
                                 # pixel.  0.01 = very subtle,
                                 # 0.05 = moderate, 0.10 = aggressive.

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
    edge_width: int          = EDGE_WIDTH,
    blur_sigma: float        = BLUR_SIGMA,
    flip_probability: float  = FLIP_PROB,
    flip_enabled: bool       = FLIP_ENABLED,
    seed: Optional[int]      = None,
) -> np.ndarray:
    """
    Inject realistic camera / segmentation noise into a clean binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Input binary image (uint8, values 0 and 255 only).
    edge_width : int
        Half-width (in pixels) of the boundary band where noise is applied.
    blur_sigma : float
        Gaussian sigma for edge-aliasing blur.  Set to 0 to skip.
    flip_probability : float
        Per-pixel probability of white↔black flip in the boundary zone.
    flip_enabled : bool
        Master on/off switch for specular-flip noise.
    seed : int or None
        RNG seed for reproducibility.  None → non-deterministic.

    Returns
    -------
    np.ndarray
        Degraded binary mask (uint8, 0 / 255), same shape as input.
    """
    assert mask.ndim == 2, "Expected a 2-D (H, W) array"
    assert mask.dtype == np.uint8, "Expected uint8 dtype"

    rng = np.random.default_rng(seed)
    out = mask.copy()  # never mutate the caller's array

    # ── 1. Build boundary band mask ──────────────────────────────────
    #    Morphological gradient (dilation − erosion) gives a 1-px-wide
    #    edge; we dilate that to the desired width.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3),
    )
    edge_1px = cv2.morphologyEx(out, cv2.MORPH_GRADIENT, kernel)

    # Widen the boundary to `edge_width` pixels on each side.
    widen_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * edge_width + 1, 2 * edge_width + 1),
    )
    band = cv2.dilate(edge_1px, widen_kernel, iterations=1)
    band_bool = band > 0  # boolean index into the boundary zone

    # ── 2. Edge aliasing — blur + re-threshold ───────────────────────
    if blur_sigma > 0:
        # Kernel size must be odd and large enough for the sigma.
        ksize = int(np.ceil(blur_sigma * 6)) | 1  # ensure odd
        blurred = cv2.GaussianBlur(
            out, (ksize, ksize), sigmaX=blur_sigma, sigmaY=blur_sigma,
        )
        # Replace only the boundary zone pixels with blurred values,
        # then re-threshold to stay binary.
        out[band_bool] = blurred[band_bool]
        out = np.where(out > 127, 255, 0).astype(np.uint8)

    # ── 3. Specular-flip noise ───────────────────────────────────────
    if flip_enabled and flip_probability > 0:
        # Generate random flips only for pixels inside the band.
        n_band = int(np.count_nonzero(band_bool))
        flips = rng.random(n_band) < flip_probability
        # XOR flip: 255 ^ 255 = 0, 0 ^ 255 = 255.
        vals = out[band_bool]
        vals[flips] ^= 255
        out[band_bool] = vals

    return out


# ═════════════════════════════════════════════════════════════════════
#  BATCH — Multi-threaded Folder Processing
# ═════════════════════════════════════════════════════════════════════

def _process_one(
    src: Path,
    dst: Path,
    edge_width: int,
    blur_sigma: float,
    flip_probability: float,
    flip_enabled: bool,
    seed: Optional[int],
) -> str:
    """Load → inject noise → save.  Returns the destination path string."""
    img = np.array(Image.open(src).convert("L"), dtype=np.uint8)
    # Ensure strictly binary input (handle any prior anti-aliasing).
    img = np.where(img > 127, 255, 0).astype(np.uint8)
    noisy = inject_noise(
        img,
        edge_width=edge_width,
        blur_sigma=blur_sigma,
        flip_probability=flip_probability,
        flip_enabled=flip_enabled,
        seed=seed,
    )
    Image.fromarray(noisy, mode="L").save(str(dst), format=OUTPUT_IMAGE_FORMAT)
    return str(dst)


def inject_noise_batch(
    input_dir: str,
    output_dir: str,
    *,
    edge_width: int          = EDGE_WIDTH,
    blur_sigma: float        = BLUR_SIGMA,
    flip_probability: float  = FLIP_PROB,
    flip_enabled: bool       = FLIP_ENABLED,
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

    print(f"[NOISE] {len(files)} masks  |  edge_w={edge_width}  "
          f"blur_σ={blur_sigma}  flip={'ON' if flip_enabled else 'OFF'}"
          f"{'  p=' + str(flip_probability) if flip_enabled else ''}")
    print(f"[NOISE] {inp}  →  {out}")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {}
        for f in files:
            dst = out / f.name
            fut = pool.submit(
                _process_one, f, dst,
                edge_width, blur_sigma, flip_probability,
                flip_enabled, seed,
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
        description="Inject realistic edge noise into clean binary masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python noise_injector.py -i output/2396N63_Carbide\n'
            '  python noise_injector.py -i output/2396N63_Carbide -o output/noisy --flip-off\n'
            '  python noise_injector.py -i output/2396N63_Carbide --blur-sigma 2.0 --flip-prob 0.08\n'
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
        "--edge-width", type=int, default=EDGE_WIDTH,
        help=f"Boundary band half-width in pixels (default {EDGE_WIDTH})",
    )
    parser.add_argument(
        "--blur-sigma", type=float, default=BLUR_SIGMA,
        help=f"Gaussian sigma for edge aliasing (default {BLUR_SIGMA}; 0 = off)",
    )
    parser.add_argument(
        "--flip-prob", type=float, default=FLIP_PROB,
        help=f"Per-pixel flip probability in boundary (default {FLIP_PROB})",
    )
    parser.add_argument(
        "--flip-off", action="store_true", default=False,
        help="Disable specular-flip noise entirely",
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
    print("  noise_injector.py — Edge-Noise Injector")
    print("=" * 62)
    print(f"  Input    : {inp}")
    print(f"  Output   : {out}")
    print(f"  Edge W   : {args.edge_width} px")
    print(f"  Blur σ   : {args.blur_sigma}")
    print(f"  Flip     : {'OFF' if args.flip_off else 'ON'}"
          f"{'  p=' + str(args.flip_prob) if not args.flip_off else ''}")
    print(f"  Seed     : {args.seed}")
    print(f"  Threads  : {args.threads}")
    print("=" * 62)

    inject_noise_batch(
        str(inp), str(out),
        edge_width=args.edge_width,
        blur_sigma=args.blur_sigma,
        flip_probability=args.flip_prob,
        flip_enabled=not args.flip_off,
        seed=args.seed,
        num_threads=args.threads,
    )

    print("[EXIT]")


if __name__ == "__main__":
    main()
