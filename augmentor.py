#!/usr/bin/env python3
"""
augmentor.py — CAD Geometry Augmentation for Drill Bit STEP Files
==================================================================

Applies non-uniform scaling (axis-wise stretch / squash) to STEP CAD
models to create geometrically diverse training data for ML pipelines.

Example transforms:
    • Stretch Z-axis by 1.10× → longer flutes
    • Squash X/Y by 0.90× → thinner drill body
    • Combine both for a "long & thin" variant

All transforms use OpenCascade's ``BRepBuilderAPI_GTransform`` through
the *cadquery* wrapper, so the output is a proper STEP B-Rep (not a
facetted mesh).

API Overview
------------
    PRESETS                          → dict of 6 named (sx, sy, sz) tuples
    augment_step(in, out, sx, sy, sz)  → transform one file
    augment_batch(in_dir, out_root, ...)  → all files, versioned folder
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# cadquery MUST be imported before OCP so that the OCC kernel is
# properly initialised (Python 3.13 / conda-forge quirk).
import cadquery as cq
from OCP.gp import gp_GTrsf, gp_Mat
from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone


# ═════════════════════════════════════════════════════════════════════
#  PRESETS — 6 sensible default parameter sets for drill augmentation
# ═════════════════════════════════════════════════════════════════════

PRESETS: Dict[str, Tuple[float, float, float]] = {
    "Longer Flutes  (Z×1.10)":           (1.00, 1.00, 1.10),
    "Thinner Drill  (XY×0.90)":          (0.90, 0.90, 1.00),
    "Wider Drill  (XY×1.10)":            (1.10, 1.10, 1.00),
    "Long & Thin  (Z×1.08, XY×0.92)":    (0.92, 0.92, 1.08),
    "Short & Wide  (Z×0.92, XY×1.08)":   (1.08, 1.08, 0.92),
    "Uniform Up-scale  (×1.05)":         (1.05, 1.05, 1.05),
}

PRESET_NAMES: List[str] = list(PRESETS.keys())


# ═════════════════════════════════════════════════════════════════════
#  CORE — single file augmentation
# ═════════════════════════════════════════════════════════════════════

def augment_step(
    input_path: str,
    output_path: str,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    scale_z: float = 1.0,
) -> None:
    """
    Load a STEP file, apply non-uniform scaling, and write a new STEP.

    Parameters
    ----------
    input_path : str
        Source STEP file (.step / .stp).
    output_path : str
        Destination STEP file.
    scale_x, scale_y, scale_z : float
        Axis-wise scale factors.  1.0 = no change.
    """
    wp = cq.importers.importStep(str(input_path))
    shape = wp.val().wrapped

    # Build a general (non-orthogonal-ok) affine transform
    gtrsf = gp_GTrsf()
    gtrsf.SetVectorialPart(gp_Mat(
        scale_x, 0.0,     0.0,
        0.0,     scale_y, 0.0,
        0.0,     0.0,     scale_z,
    ))

    builder = BRepBuilderAPI_GTransform(shape, gtrsf, True)
    builder.Build()
    new_shape = builder.Shape()

    # Write STEP via the low-level OCC writer
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = STEPControl_Writer()
    writer.Transfer(new_shape, STEPControl_AsIs)
    status = writer.Write(str(out))
    if status != IFSelect_RetDone:
        raise RuntimeError(
            f"STEP export failed (status {status}) for {out}")


# ═════════════════════════════════════════════════════════════════════
#  VERSIONED OUTPUT — auto-numbered augmentation run folders
# ═════════════════════════════════════════════════════════════════════

def _next_aug_run_dir(parent: Path) -> Path:
    """Return the next ``aug_NNN`` subdirectory under *parent*."""
    parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        int(d.name.split("_", 1)[1])
        for d in parent.iterdir()
        if d.is_dir() and d.name.startswith("aug_")
           and d.name.split("_", 1)[1].isdigit()
    )
    next_id = (existing[-1] + 1) if existing else 1
    return parent / f"aug_{next_id:03d}"


def _write_aug_config(
    run_dir: Path,
    aug_cfg: Dict[str, Any],
    *,
    file_names: Optional[List[str]] = None,
) -> Path:
    """Write ``augmentation_config.json`` into *run_dir*."""
    meta = {
        "augmentation_type": "non_uniform_scale",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": aug_cfg,
    }
    if file_names is not None:
        meta["files"] = file_names
    path = run_dir / "augmentation_config.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


# ═════════════════════════════════════════════════════════════════════
#  BATCH — process every STEP file in a folder
# ═════════════════════════════════════════════════════════════════════

def augment_batch(
    input_dir: str,
    output_root: str,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    scale_z: float = 1.0,
    progress_callback=None,
    log_callback=None,
    abort_flag=None,
) -> Optional[Path]:
    """
    Augment all STEP files in *input_dir* and save to a versioned
    ``aug_NNN`` subfolder under *output_root*.

    Parameters
    ----------
    input_dir : str
        Folder containing .step / .stp files.
    output_root : str
        Parent folder where ``aug_NNN/`` is created.
    scale_x, scale_y, scale_z : float
        Axis-wise scale factors.
    progress_callback : callable(file_idx, total) or None
        Per-file progress reporter.
    log_callback : callable(msg) or None
        Log message sink.
    abort_flag : object with ``_abort`` attribute, or None.

    Returns
    -------
    Path | None
        The ``aug_NNN`` directory that was created, or None on abort.
    """
    _log = log_callback or print
    inp = Path(input_dir)
    out_root = Path(output_root)

    files = sorted(
        f for f in inp.iterdir()
        if f.suffix.lower() in (".step", ".stp")
    )
    if not files:
        _log(f"[AUG] No STEP files found in {inp}")
        return None

    # Create versioned run folder
    run_dir = _next_aug_run_dir(out_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    aug_cfg = dict(scale_x=scale_x, scale_y=scale_y, scale_z=scale_z)
    file_names = [f.name for f in files]
    _write_aug_config(run_dir, aug_cfg, file_names=file_names)

    total = len(files)
    _log(f"[AUG] {total} STEP files → {run_dir.name}  "
         f"(sx={scale_x:.3f}, sy={scale_y:.3f}, sz={scale_z:.3f})")

    t0 = time.perf_counter()
    for i, src in enumerate(files):
        if abort_flag is not None and getattr(abort_flag, '_abort', False):
            _log("[ABORT] Augmentation cancelled.")
            return None

        _log(f"  [{i+1}/{total}] {src.name}")
        dst = run_dir / src.name
        augment_step(
            str(src), str(dst),
            scale_x=scale_x, scale_y=scale_y, scale_z=scale_z,
        )

        if progress_callback is not None:
            progress_callback(i + 1, total)

    elapsed = time.perf_counter() - t0
    _log(f"[DONE] Augmented {total} files in {elapsed:.1f} s → {run_dir}")
    return run_dir


# ═════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    _SCRIPT_DIR = Path(__file__).resolve().parent

    preset_help = "\n".join(
        f"  {i+1}. {name}: sx={v[0]}, sy={v[1]}, sz={v[2]}"
        for i, (name, v) in enumerate(PRESETS.items())
    )

    parser = argparse.ArgumentParser(
        description="Non-uniform scaling augmentation for drill STEP files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Presets (use --preset N):\n" + preset_help + "\n\n"
            "Examples:\n"
            "  python augmentor.py -i drills\n"
            "  python augmentor.py -i drills --preset 1\n"
            "  python augmentor.py -i drills --sx 0.95 --sy 0.95 --sz 1.12\n"
        ),
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Folder containing STEP files",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output root folder (default: <input>_augmented)",
    )
    parser.add_argument(
        "--preset", type=int, default=None,
        help="Preset number (1-6). Overrides --sx/--sy/--sz.",
    )
    parser.add_argument("--sx", type=float, default=1.0, help="Scale X")
    parser.add_argument("--sy", type=float, default=1.0, help="Scale Y")
    parser.add_argument("--sz", type=float, default=1.0, help="Scale Z")
    args = parser.parse_args()

    # Resolve preset
    if args.preset is not None:
        if not (1 <= args.preset <= len(PRESETS)):
            parser.error(f"--preset must be 1-{len(PRESETS)}")
        name = PRESET_NAMES[args.preset - 1]
        sx, sy, sz = PRESETS[name]
        print(f"Using preset {args.preset}: {name}")
    else:
        sx, sy, sz = args.sx, args.sy, args.sz

    inp = Path(args.input)
    if not inp.is_absolute():
        inp = _SCRIPT_DIR / inp

    if args.output is None:
        out = inp.parent / (inp.name + "_augmented")
    else:
        out = Path(args.output)
        if not out.is_absolute():
            out = _SCRIPT_DIR / out

    print("=" * 62)
    print("  augmentor.py — CAD Geometry Augmentation")
    print("=" * 62)
    print(f"  Input  : {inp}")
    print(f"  Output : {out}")
    print(f"  Scale  : X={sx:.3f}  Y={sy:.3f}  Z={sz:.3f}")
    print("=" * 62)

    run_dir = augment_batch(
        str(inp), str(out),
        scale_x=sx, scale_y=sy, scale_z=sz,
    )
    if run_dir is not None:
        print(f"\n  Created: {run_dir}")
    print("[EXIT]")


if __name__ == "__main__":
    main()
