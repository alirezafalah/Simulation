#!/usr/bin/env python3
"""
noise_injector.py — Temporally-Coherent Contour-Dent Noise
============================================================

Simulates the localised under-segmentation artefact produced by
background-subtraction on metallic CNC tools:

    • A dent (inward erosion of the mask edge) appears at a random
      contour position.
    • It persists for several consecutive rotation frames (e.g. 5-15),
      gradually growing then shrinking (cosine temporal envelope).
    • 1-2 such events occur per tool, each at a different random
      location and angle range.

This models the real-world effect where specular reflection at certain
angles causes the segmentation boundary to sit a few pixels inside the
true tool edge for a short arc of the rotation.

API Overview
------------
    plan_dent_events(...)        → List[DentEvent]     # plan once per tool
    inject_noise_frame(mask, events, frame_index)      # apply per frame
    inject_noise(mask, ...)      → np.ndarray          # single-mask preview

    inject_noise_batch(input_dir, output_dir, ...)     # CLI batch
"""

from __future__ import annotations

import argparse
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any

import cv2
from PIL import Image


# ═════════════════════════════════════════════════════════════════════
#  CONFIGURATION — defaults
# ═════════════════════════════════════════════════════════════════════

N_EVENTS: int        = 1        # dent events per tool (1–2 typical)
FRAME_SPAN_MIN: int  = 10        # minimum consecutive frames per event
FRAME_SPAN_MAX: int  = 20       # maximum consecutive frames per event
DENT_SPAN_MIN: int   = 250       # min contour-pixel arc per dent
DENT_SPAN_MAX: int   = 500      # max contour-pixel arc per dent
MAX_DEPTH: float     = 3.5      # peak erosion depth (px)
N_FRAMES: int        = 360      # total frames in one rotation

OUTPUT_IMAGE_FORMAT: str = "PNG"
NUM_IO_THREADS: int  = 12


# ═════════════════════════════════════════════════════════════════════
#  DENT EVENT — one temporally-coherent erosion patch
# ═════════════════════════════════════════════════════════════════════

@dataclass
class DentEvent:
    """Describes one localised erosion patch and its temporal envelope."""
    contour_frac: float     # 0–1: where on the contour the dent centre is
    dent_span: int          # arc length in contour pixels
    peak_depth: float       # max erosion depth (pixels) at peak frame
    start_frame: int        # first frame where the dent appears
    end_frame: int          # last frame (inclusive) where the dent is visible
    peak_frame: int         # frame at which depth = peak_depth

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame + 1

    def depth_at(self, frame: int) -> float:
        """Temporal cosine envelope: 0 → peak → 0."""
        if frame < self.start_frame or frame > self.end_frame:
            return 0.0
        half = max((self.end_frame - self.start_frame) / 2.0, 0.5)
        mid = (self.start_frame + self.end_frame) / 2.0
        t = abs(frame - mid) / half          # 0 at centre, 1 at edges
        t = min(t, 1.0)
        return self.peak_depth * (1.0 + np.cos(np.pi * t)) / 2.0


def plan_dent_events(
    *,
    n_events: int       = N_EVENTS,
    n_frames: int       = N_FRAMES,
    frame_span_min: int = FRAME_SPAN_MIN,
    frame_span_max: int = FRAME_SPAN_MAX,
    dent_span_min: int  = DENT_SPAN_MIN,
    dent_span_max: int  = DENT_SPAN_MAX,
    max_depth: float    = MAX_DEPTH,
    seed: Optional[int] = None,
) -> List[DentEvent]:
    """
    Pre-plan dent events for one tool's full 360° rotation.

    Call this **once** per tool, then pass the list to
    ``inject_noise_frame()`` for each frame.

    Returns
    -------
    list[DentEvent]
    """
    rng = np.random.default_rng(seed)
    events: List[DentEvent] = []

    for _ in range(n_events):
        # Temporal: how many frames, and where in the rotation
        fspan = rng.integers(frame_span_min, frame_span_max + 1)
        fspan = min(fspan, n_frames)
        start = rng.integers(0, n_frames)
        end   = start + fspan - 1               # may wrap past n_frames
        mid   = start + fspan // 2

        # Spatial: where on the contour and how big
        contour_frac = rng.uniform(0.0, 1.0)
        dent_span    = rng.integers(dent_span_min, dent_span_max + 1)
        depth        = rng.uniform(1.0, max(1.0, max_depth))

        events.append(DentEvent(
            contour_frac=contour_frac,
            dent_span=dent_span,
            peak_depth=depth,
            start_frame=start,
            end_frame=end,
            peak_frame=mid,
        ))

    return events


# ═════════════════════════════════════════════════════════════════════
#  CORE — apply dents to a single mask
# ═════════════════════════════════════════════════════════════════════

def _apply_dents(
    mask: np.ndarray,
    dents: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Low-level: apply a list of dent specs to *mask*.

    Each dict: ``{contour_frac, dent_span, depth}``
    (depth already includes the temporal envelope).

    Returns a new uint8 array.
    """
    out = mask.copy()

    contours, _ = cv2.findContours(
        out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return out
    contour = max(contours, key=cv2.contourArea)
    pts = contour.squeeze(axis=1)
    n_pts = len(pts)
    if n_pts < 20:
        return out

    dist = cv2.distanceTransform(out, cv2.DIST_L2, 5)
    dent_map = np.zeros(mask.shape, dtype=np.float32)

    max_d = 0.0
    for d in dents:
        depth = float(d["depth"])
        if depth < 0.3:
            continue
        span = min(int(d["dent_span"]), n_pts)
        frac = float(d["contour_frac"])
        start = int(round(frac * n_pts)) % n_pts

        # Cosine spatial window: 0 → depth → 0
        t = np.linspace(0, np.pi, span)
        weights = (1.0 - np.cos(t)) / 2.0 * depth

        radius = max(1, int(np.ceil(depth)))
        for k in range(span):
            idx = (start + k) % n_pts
            cx, cy = int(pts[idx, 0]), int(pts[idx, 1])
            cv2.circle(dent_map, (cx, cy), radius,
                       float(weights[k]), thickness=-1)
        max_d = max(max_d, depth)

    if max_d > 0:
        ksize = max(3, (int(np.ceil(max_d)) * 2) | 1)
        dent_map = cv2.GaussianBlur(dent_map, (ksize, ksize), sigmaX=1.0)

    erode_mask = (dist > 0) & (dist < dent_map)
    out[erode_mask] = 0
    return out


# ── Public: per-frame with temporal plan ─────────────────────────────

def inject_noise_frame(
    mask: np.ndarray,
    events: List[DentEvent],
    frame_index: int,
    n_frames: int = N_FRAMES,
) -> np.ndarray:
    """
    Apply temporally-coherent dents for a specific frame.

    Parameters
    ----------
    mask : np.ndarray
        Clean binary mask (uint8, 0/255) for this frame.
    events : list[DentEvent]
        Pre-planned dent events (from ``plan_dent_events``).
    frame_index : int
        Current frame number (0-based).
    n_frames : int
        Total number of frames in the rotation.

    Returns
    -------
    np.ndarray
        Degraded mask.
    """
    assert mask.ndim == 2 and mask.dtype == np.uint8

    active_dents = []
    for ev in events:
        # Handle wrap-around (event may span past n_frames)
        if ev.end_frame >= n_frames:
            # Split: [start..n_frames-1] and [0..wrap]
            in_range = (frame_index >= ev.start_frame or
                        frame_index <= ev.end_frame % n_frames)
        else:
            in_range = ev.start_frame <= frame_index <= ev.end_frame

        if not in_range:
            continue

        # Compute temporal depth with wrap-aware frame distance
        if ev.end_frame >= n_frames:
            # unwrap frame_index for depth calculation
            fi = frame_index if frame_index >= ev.start_frame \
                else frame_index + n_frames
            half = max((ev.end_frame - ev.start_frame) / 2.0, 0.5)
            mid = (ev.start_frame + ev.end_frame) / 2.0
            t = abs(fi - mid) / half
        else:
            half = max((ev.end_frame - ev.start_frame) / 2.0, 0.5)
            mid = (ev.start_frame + ev.end_frame) / 2.0
            t = abs(frame_index - mid) / half

        t = min(t, 1.0)
        depth = ev.peak_depth * (1.0 + np.cos(np.pi * t)) / 2.0

        if depth >= 0.3:
            active_dents.append(dict(
                contour_frac=ev.contour_frac,
                dent_span=ev.dent_span,
                depth=depth,
            ))

    if not active_dents:
        return mask.copy()

    return _apply_dents(mask, active_dents)


# ── Public: single-mask preview (no temporal info) ───────────────────

def inject_noise(
    mask: np.ndarray,
    *,
    n_dents: int             = N_EVENTS,
    dent_span_min: int       = DENT_SPAN_MIN,
    dent_span_max: int       = DENT_SPAN_MAX,
    max_depth: float         = MAX_DEPTH,
    seed: Optional[int]      = None,
) -> np.ndarray:
    """
    Single-mask preview: apply *n_dents* erosion patches at **full peak
    depth** (no temporal envelope).  Used by the GUI live preview.

    For temporally-coherent noise during dataset generation, use
    ``plan_dent_events()`` + ``inject_noise_frame()`` instead.
    """
    assert mask.ndim == 2 and mask.dtype == np.uint8
    if n_dents <= 0 or max_depth < 0.5:
        return mask.copy()

    rng = np.random.default_rng(seed)
    dents = []
    for _ in range(n_dents):
        dents.append(dict(
            contour_frac=rng.uniform(0.0, 1.0),
            dent_span=int(rng.integers(dent_span_min, dent_span_max + 1)),
            depth=float(rng.uniform(1.0, max(1.0, max_depth))),
        ))
    return _apply_dents(mask, dents)


# ═════════════════════════════════════════════════════════════════════
#  BATCH — folder processing (sequential, with temporal coherence)
# ═════════════════════════════════════════════════════════════════════

def inject_noise_batch(
    input_dir: str,
    output_dir: str,
    *,
    n_events: int            = N_EVENTS,
    n_frames: int            = N_FRAMES,
    frame_span_min: int      = FRAME_SPAN_MIN,
    frame_span_max: int      = FRAME_SPAN_MAX,
    dent_span_min: int       = DENT_SPAN_MIN,
    dent_span_max: int       = DENT_SPAN_MAX,
    max_depth: float         = MAX_DEPTH,
    seed: Optional[int]      = None,
) -> None:
    """
    Apply temporally-coherent noise to a folder of ordered mask frames.

    Files are sorted by name so frame ordering is preserved.
    """
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in inp.iterdir()
        if f.suffix.lower() in (".png", ".tiff", ".tif")
    )
    if not files:
        print(f"[NOISE] No image files found in {inp}")
        return

    total = len(files)
    events = plan_dent_events(
        n_events=n_events,
        n_frames=total,
        frame_span_min=frame_span_min,
        frame_span_max=frame_span_max,
        dent_span_min=dent_span_min,
        dent_span_max=dent_span_max,
        max_depth=max_depth,
        seed=seed,
    )

    print(f"[NOISE] {total} masks  |  events={n_events}  "
          f"frames/ev={frame_span_min}–{frame_span_max}  "
          f"span={dent_span_min}–{dent_span_max}  "
          f"depth≤{max_depth:.1f} px")
    for i, ev in enumerate(events):
        print(f"         event #{i}: contour {ev.contour_frac:.2f}, "
              f"frames {ev.start_frame}–{ev.end_frame}, "
              f"span={ev.dent_span}, peak={ev.peak_depth:.1f} px")
    print(f"[NOISE] {inp}  →  {out}")

    t0 = time.perf_counter()
    for frame_idx, src in enumerate(files):
        img = np.array(Image.open(src).convert("L"), dtype=np.uint8)
        img = np.where(img > 127, 255, 0).astype(np.uint8)

        noisy = inject_noise_frame(img, events, frame_idx, n_frames=total)

        dst = out / src.name
        Image.fromarray(noisy, mode="L").save(str(dst), format="PNG")

        if (frame_idx + 1) % 36 == 0 or frame_idx + 1 == total:
            elapsed = time.perf_counter() - t0
            fps = (frame_idx + 1) / elapsed
            print(f"         {frame_idx+1:3d}/{total}  [{fps:.1f} img/s]")

    t_total = time.perf_counter() - t0
    print(f"\n[NOISE] Done — {total} masks in {t_total:.2f} s "
          f"({total / t_total:.1f} img/s)")


# ═════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    _SCRIPT_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Temporally-coherent contour-dent noise for sim masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python noise_injector.py -i output/2396N63_Carbide\n'
            '  python noise_injector.py -i output/2396N63_Carbide --n-events 1\n'
            '  python noise_injector.py -i output/2396N63_Carbide --max-depth 3 --frame-span-max 20\n'
        ),
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Directory of ordered mask frames (sorted by name)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory (default: <input>_noisy)",
    )
    parser.add_argument(
        "--n-events", type=int, default=N_EVENTS,
        help=f"Number of dent events per tool (default {N_EVENTS})",
    )
    parser.add_argument(
        "--frame-span-min", type=int, default=FRAME_SPAN_MIN,
        help=f"Min frames per event (default {FRAME_SPAN_MIN})",
    )
    parser.add_argument(
        "--frame-span-max", type=int, default=FRAME_SPAN_MAX,
        help=f"Max frames per event (default {FRAME_SPAN_MAX})",
    )
    parser.add_argument(
        "--span-min", type=int, default=DENT_SPAN_MIN,
        help=f"Min contour-pixel arc per dent (default {DENT_SPAN_MIN})",
    )
    parser.add_argument(
        "--span-max", type=int, default=DENT_SPAN_MAX,
        help=f"Max contour-pixel arc per dent (default {DENT_SPAN_MAX})",
    )
    parser.add_argument(
        "--max-depth", type=float, default=MAX_DEPTH,
        help=f"Peak erosion depth in pixels (default {MAX_DEPTH})",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducibility",
    )
    args = parser.parse_args()

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
    print("  noise_injector.py — Temporal Contour-Dent Noise")
    print("=" * 62)
    print(f"  Input       : {inp}")
    print(f"  Output      : {out}")
    print(f"  Events      : {args.n_events}")
    print(f"  Frame span  : {args.frame_span_min}–{args.frame_span_max}")
    print(f"  Dent span   : {args.span_min}–{args.span_max} contour px")
    print(f"  Max depth   : {args.max_depth:.1f} px")
    print(f"  Seed        : {args.seed}")
    print("=" * 62)

    inject_noise_batch(
        str(inp), str(out),
        n_events=args.n_events,
        frame_span_min=args.frame_span_min,
        frame_span_max=args.frame_span_max,
        dent_span_min=args.span_min,
        dent_span_max=args.span_max,
        max_depth=args.max_depth,
        seed=args.seed,
    )

    print("[EXIT]")


if __name__ == "__main__":
    main()
