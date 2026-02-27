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
import json
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple

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
#  NOISE PRESETS — named parameter sets for quick selection
# ═════════════════════════════════════════════════════════════════════

NOISE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Default (1 event, depth 3.5)": dict(
        n_events=1, frame_span_min=10, frame_span_max=20,
        dent_span_min=250, dent_span_max=500, max_depth=3.5, seed=None,
    ),
    "Moderate (2 events, depth 5.0)": dict(
        n_events=2, frame_span_min=10, frame_span_max=20,
        dent_span_min=250, dent_span_max=500, max_depth=5.0, seed=None,
    ),
    "Aggressive (3 events, depth 4.0)": dict(
        n_events=3, frame_span_min=10, frame_span_max=20,
        dent_span_min=300, dent_span_max=600, max_depth=4.0, seed=None,
    ),
}

NOISE_PRESET_NAMES: List[str] = list(NOISE_PRESETS.keys())


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
        fs_lo = min(frame_span_min, frame_span_max)
        fs_hi = max(frame_span_min, frame_span_max)
        fspan = rng.integers(fs_lo, fs_hi + 1)
        fspan = min(fspan, n_frames)
        start = rng.integers(0, n_frames)
        end   = start + fspan - 1               # may wrap past n_frames
        mid   = start + fspan // 2

        # Spatial: where on the contour and how big
        contour_frac = rng.uniform(0.0, 1.0)
        ds_lo = min(dent_span_min, dent_span_max)
        ds_hi = max(dent_span_min, dent_span_max)
        dent_span    = rng.integers(ds_lo, ds_hi + 1)
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


# ═════════════════════════════════════════════════════════════════════#  VERSIONED OUTPUT — auto-numbered noise run folders
# ═════════════════════════════════════════════════════════════════

def _next_noise_run_dir(parent: Path) -> Path:
    """
    Return the next available ``noise_NNN`` subdirectory under *parent*.

    Scans existing ``noise_NNN`` folders and picks NNN+1.
    """
    parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        int(d.name.split("_", 1)[1])
        for d in parent.iterdir()
        if d.is_dir() and d.name.startswith("noise_")
           and d.name.split("_", 1)[1].isdigit()
    )
    next_id = (existing[-1] + 1) if existing else 1
    return parent / f"noise_{next_id:03d}"


def _write_noise_config(
    run_dir: Path,
    noise_cfg: Dict[str, Any],
    *,
    tool_names: Optional[List[str]] = None,
) -> Path:
    """
    Write a ``noise_config.json`` into *run_dir* with all parameters
    and metadata for reproducibility.
    """
    meta = {
        "noise_type": "temporal_contour_dent",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            k: v for k, v in noise_cfg.items()
        },
    }
    if tool_names is not None:
        meta["tools"] = tool_names
    path = run_dir / "noise_config.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


# ═════════════════════════════════════════════════════════════════#  BATCH — folder processing (sequential, with temporal coherence)
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
        fmt = "TIFF" if src.suffix.lower() in (".tiff", ".tif") else "PNG"
        Image.fromarray(noisy, mode="L").save(str(dst), format=fmt)

        if (frame_idx + 1) % 36 == 0 or frame_idx + 1 == total:
            elapsed = time.perf_counter() - t0
            fps = (frame_idx + 1) / elapsed
            print(f"         {frame_idx+1:3d}/{total}  [{fps:.1f} img/s]")

    t_total = time.perf_counter() - t0
    print(f"\n[NOISE] Done — {total} masks in {t_total:.2f} s "
          f"({total / t_total:.1f} img/s)")


def inject_noise_batch_all(
    clean_root: str,
    noisy_root: str,
    *,
    n_events: int            = N_EVENTS,
    frame_span_min: int      = FRAME_SPAN_MIN,
    frame_span_max: int      = FRAME_SPAN_MAX,
    dent_span_min: int       = DENT_SPAN_MIN,
    dent_span_max: int       = DENT_SPAN_MAX,
    max_depth: float         = MAX_DEPTH,
    seed: Optional[int]      = None,
    progress_callback=None,
    log_callback=None,
    abort_flag=None,
) -> Optional[Path]:
    """
    Apply noise to **all** tool subfolders under *clean_root*.

    Creates a versioned ``noise_NNN`` subfolder in *noisy_root* with a
    ``noise_config.json`` documenting the parameters used.  Each tool's
    noisy masks land in ``noise_NNN/<tool_name>/``.

    Parameters
    ----------
    clean_root : str
        Parent directory containing tool subfolders with clean masks.
    noisy_root : str
        Parent directory where versioned noise runs are stored.
    progress_callback : callable(tool_idx, tool_count, frame_idx, frame_count)
        Optional progress reporter.
    log_callback : callable(msg: str)
        Optional log message sink.
    abort_flag : object with bool ``_abort`` attribute, or None.

    Returns
    -------
    Path | None
        The ``noise_NNN`` directory that was created, or None on abort.
    """
    _log = log_callback or print
    clean_p = Path(clean_root)
    noisy_p = Path(noisy_root)

    # Discover tool subfolders (each contains mask PNGs)
    tool_dirs = sorted(
        d for d in clean_p.iterdir()
        if d.is_dir() and any(
            f.suffix.lower() in (".png", ".tiff", ".tif")
            for f in d.iterdir()
        )
    )
    if not tool_dirs:
        _log(f"[NOISE] No tool subfolders with masks found in {clean_p}")
        return None

    # Create versioned run folder
    run_dir = _next_noise_run_dir(noisy_p)
    run_dir.mkdir(parents=True, exist_ok=True)

    noise_cfg = dict(
        n_events=n_events,
        frame_span_min=frame_span_min,
        frame_span_max=frame_span_max,
        dent_span_min=dent_span_min,
        dent_span_max=dent_span_max,
        max_depth=max_depth,
        seed=seed,
    )
    tool_names = [d.name for d in tool_dirs]
    _write_noise_config(run_dir, noise_cfg, tool_names=tool_names)

    n_tools = len(tool_dirs)
    _log(f"[NOISE] {n_tools} tools → {run_dir.name}")

    for ti, tool_dir in enumerate(tool_dirs):
        if abort_flag is not None and getattr(abort_flag, '_abort', False):
            _log("[ABORT] Noise batch cancelled.")
            return None

        tool_name = tool_dir.name
        _log(f"── {ti+1}/{n_tools}: {tool_name} ──")

        files = sorted(
            f for f in tool_dir.iterdir()
            if f.suffix.lower() in (".png", ".tiff", ".tif")
        )
        if not files:
            _log(f"   (no mask files, skipping)")
            continue

        total = len(files)
        out_tool = run_dir / tool_name
        out_tool.mkdir(parents=True, exist_ok=True)

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

        for fi, src in enumerate(files):
            if abort_flag is not None and getattr(abort_flag, '_abort', False):
                _log("[ABORT] Noise batch cancelled.")
                return None

            img = np.array(Image.open(src).convert("L"), dtype=np.uint8)
            img = np.where(img > 127, 255, 0).astype(np.uint8)
            noisy = inject_noise_frame(img, events, fi, n_frames=total)
            fmt = "TIFF" if src.suffix.lower() in (".tiff", ".tif") else "PNG"
            Image.fromarray(noisy, mode="L").save(
                str(out_tool / src.name), format=fmt)

            if progress_callback is not None:
                progress_callback(ti, n_tools, fi + 1, total)

    _log(f"[DONE] Noise run complete → {run_dir}")
    return run_dir


# ═════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    _SCRIPT_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Temporally-coherent contour-dent noise for sim masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  Multi-tool (default):  -i points to clean root with tool subfolders\n"
            "      → auto-creates output_noisy/noise_NNN/ with noise_config.json\n\n"
            "  Single-folder:  --single  -i points to one folder of masks\n"
            "      → writes noisy masks directly into -o\n\n"
            "Examples:\n"
            '  python noise_injector.py -i output\n'
            '  python noise_injector.py -i output -o output_noisy --n-events 2\n'
            '  python noise_injector.py --single -i output/2396N63_Carbide\n'
        ),
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Clean mask root (multi-tool) or single folder (--single)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Noisy output root (default: <input>_noisy)",
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Single-folder mode: input is one folder of masks, not a root",
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
    print(f"  Mode        : {'single-folder' if args.single else 'multi-tool (versioned)'}")
    print(f"  Input       : {inp}")
    print(f"  Output root : {out}")
    print(f"  Events      : {args.n_events}")
    print(f"  Frame span  : {args.frame_span_min}–{args.frame_span_max}")
    print(f"  Dent span   : {args.span_min}–{args.span_max} contour px")
    print(f"  Max depth   : {args.max_depth:.1f} px")
    print(f"  Seed        : {args.seed}")
    print("=" * 62)

    common_kw = dict(
        n_events=args.n_events,
        frame_span_min=args.frame_span_min,
        frame_span_max=args.frame_span_max,
        dent_span_min=args.span_min,
        dent_span_max=args.span_max,
        max_depth=args.max_depth,
        seed=args.seed,
    )

    if args.single:
        inject_noise_batch(str(inp), str(out), **common_kw)
    else:
        run_dir = inject_noise_batch_all(
            str(inp), str(out), **common_kw)
        if run_dir is not None:
            print(f"\n  Created: {run_dir}")

    print("[EXIT]")


if __name__ == "__main__":
    main()
