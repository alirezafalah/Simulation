# Simulation — Sim2Real Mask Generation

Generates synthetic binary mask datasets from 3D CAD models and injects realistic edge noise to bridge the sim-to-real gap.

## Project Structure

| File | Description |
|---|---|
| `render_engine.py` | 360° binary mask renderer (CLI) — loads STEP/STL, places a virtual camera at 250 mm working distance, outputs one mask per degree of rotation |
| `noise_injector.py` | Edge-only noise pipeline (CLI) — blur + re-threshold aliasing and specular pixel flips along the tool boundary |
| `simulation_gui.py` | Desktop GUI (PySide6) — integrates both modules with live previews, parameter tuning, and batch processing |
| `drills/` | CAD model files (.STEP) |
| `output/` | Default output directory for rendered masks |

## Dependencies

```bash
pip install pyvista vtk cadquery numpy opencv-python Pillow PySide6
```

## Run the Desktop GUI (recommended)

From the Simulation folder:

```bash
python simulation_gui.py
```

Dark-themed PySide6 application with two tabs:

- **Render** — select CAD files, configure camera (working distance, resolution, frame count), preview renders, and batch-generate clean mask datasets
- **Noise** — tune edge-noise parameters with a live before/after preview on any sample mask, run standalone batch noise injection on existing mask folders

## Render Engine (CLI)

Generate 360° binary mask silhouettes from a single STEP/STL model:

```bash
python render_engine.py -m "drills/2396N63_Carbide.STEP"
```

Options:

| Flag | Description | Default |
|---|---|---|
| `-m`, `--model` | Path to 3D model (.step, .stp, .stl) | *required* |
| `-o`, `--output` | Output directory | `output/` |
| `--debug` | Open interactive preview instead of batch render | off |
| `--width` | Image width in px | 1024 |
| `--height` | Image height in px | 768 |

## Noise Injector (CLI)

Apply edge-only noise to a folder of clean masks:

```bash
python noise_injector.py -i output/2396N63_Carbide
```

Options:

| Flag | Description | Default |
|---|---|---|
| `-i`, `--input` | Directory of clean masks | *required* |
| `-o`, `--output` | Output directory | `<input>_noisy` |
| `--edge-width` | Boundary band half-width in px | 5 |
| `--blur-sigma` | Gaussian sigma for edge aliasing (0 = off) | 1.0 |
| `--flip-prob` | Per-pixel flip probability in boundary | 0.03 |
| `--flip-off` | Disable specular-flip noise | off |
| `--seed` | RNG seed for reproducibility | random |
| `--threads` | I/O thread count | 12 |