# Simulation — Sim2Real Mask Generation

Generates synthetic binary mask datasets from 3D CAD models, injects realistic edge noise, and augments CAD geometry — all to bridge the sim-to-real gap for ML training.

## Project Structure

| File | Description |
|---|---|
| `render_engine.py` | 360° binary mask renderer (CLI) — loads STEP/STL, places a virtual camera at 250 mm working distance, outputs one mask per degree of rotation |
| `noise_injector.py` | Temporally-coherent contour-dent noise (CLI) — erodes the mask boundary at random locations with a cosine fade-in/fade-out across consecutive frames |
| `augmentor.py` | CAD geometry augmentation (CLI) — applies non-uniform scaling to STEP files to create geometrically diverse training data |
| `simulation_gui.py` | Desktop GUI (PySide6) — integrates all three modules with live previews, frame scrubber, parameter tuning, presets, and batch processing |
| `drills/` | CAD model files (.STEP) |
| `output/` | Default output directory for rendered clean masks |
| `output_noisy/` | Default output directory for noisy masks (versioned `noise_NNN/` subfolders) |

## Dependencies

```bash
pip install pyvista vtk cadquery numpy opencv-python Pillow PySide6
```

## Run the Desktop GUI (recommended)

From the Simulation folder:

```bash
python simulation_gui.py
```

Dark-themed PySide6 application with three tabs:

- **Render** — select CAD files, configure camera (working distance, resolution, frame count, image format), preview renders, and batch-generate clean mask datasets (default format: TIFF)
- **Noise** — select a noise preset (Default / Moderate / Aggressive) or tune custom parameters with a live before/after preview and frame scrubber; run standalone batch noise injection on existing mask folders
- **Augment** — select a geometry preset or set custom scale X/Y/Z factors; batch-augment all STEP files in a folder with versioned output

### Noise Presets

Three built-in presets, selectable in the GUI or applied all at once during generation:

| Preset | Events | Frame Span | Dent Arc (px) | Max Depth |
|---|---|---|---|---|
| **Default** | 1 | 10–20 | 250–500 | 3.5 |
| **Moderate** | 2 | 10–20 | 250–500 | 5.0 |
| **Aggressive** | 3 | 10–20 | 300–600 | 4.0 |

**Generation modes:**
- **"Apply current noise preset"** — uses the single currently-selected preset (or custom values), writing one `noise_NNN/` folder
- **"Apply ALL noise presets"** — renders clean masks once, then applies every preset, creating one `noise_NNN/` folder per preset (overrides the single-preset checkbox)

### Augmentation Presets

Six geometry presets for non-uniform STEP file scaling:

| Preset | Scale X | Scale Y | Scale Z |
|---|---|---|---|
| Longer Flutes (Z×1.10) | 1.00 | 1.00 | 1.10 |
| Thinner Drill (XY×0.90) | 0.90 | 0.90 | 1.00 |
| Wider Drill (XY×1.10) | 1.10 | 1.10 | 1.00 |
| Long & Thin (Z×1.08, XY×0.92) | 0.92 | 0.92 | 1.08 |
| Short & Wide (Z×0.92, XY×1.08) | 1.08 | 1.08 | 0.92 |
| Uniform Up-scale (×1.05) | 1.05 | 1.05 | 1.05 |

Output is saved in versioned `aug_NNN/` subfolders with an `augmentation_config.json` metadata file.

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

Apply temporally-coherent contour-dent noise to a folder of ordered mask frames:

```bash
python noise_injector.py -i output/2396N63_Carbide
python noise_injector.py -i output/2396N63_Carbide --n-events 2 --max-depth 5
python noise_injector.py -i output/2396N63_Carbide --frame-span-max 20 --span-min 250
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `-i`, `--input` | Directory of ordered mask frames (sorted by name) | *required* |
| `-o`, `--output` | Output directory | `<input>_noisy` |
| `--n-events` | Number of dent events per tool | 1 |
| `--frame-span-min` | Minimum consecutive frames each dent persists | 10 |
| `--frame-span-max` | Maximum consecutive frames each dent persists | 20 |
| `--span-min` | Minimum contour-pixel arc length per dent | 250 |
| `--span-max` | Maximum contour-pixel arc length per dent | 500 |
| `--max-depth` | Peak erosion depth in pixels | 3.5 |
| `--seed` | RNG seed for reproducibility | random |

### Parameter Reference

These parameters apply to both the CLI and the GUI noise tab.

| Parameter | GUI Widget | What it controls |
|---|---|---|
| **Events per Tool** (`--n-events`) | `spin_n_events` | How many independent dent artifacts appear during one 360° rotation. Each event = one erosion patch at a fixed contour position. 1–2 is realistic for real-world specular artifacts. |
| **Frame Span Min** (`--frame-span-min`) | `spin_frame_span_min` | Minimum number of consecutive frames each dent event persists. The dent follows a cosine temporal envelope: gradually fades in, peaks at the middle frame, then fades out. |
| **Frame Span Max** (`--frame-span-max`) | `spin_frame_span_max` | Maximum number of consecutive frames each dent event persists. Actual span per event is sampled uniformly from [min, max]. |
| **Dent Span Min** (`--span-min`) | `spin_span_min` | Minimum arc length along the mask contour (in contour pixels) that gets eroded. Controls how wide the bite is. |
| **Dent Span Max** (`--span-max`) | `spin_span_max` | Maximum arc length along the mask contour (in contour pixels). Actual span per event is sampled uniformly from [min, max]. |
| **Max Depth** (`--max-depth`) | `spin_max_depth` | Peak inward erosion distance in pixels. Each event picks a random depth in [1.0, max_depth]. 1–2 = subtle, 2–3 = visible, 3+ = aggressive. |
| **Seed** (`--seed`) | `spin_seed` | RNG seed for reproducible dent placement. -1 in the GUI means non-deterministic (random). |

### Noise Model

The noise simulates localised under-segmentation caused by specular reflection on metallic CNC tools:

1. **Planning** (`plan_dent_events`): Called once per tool. Generates a list of `DentEvent` objects, each with a random contour position, arc span, peak erosion depth, and a random frame window within the 360° rotation.

2. **Per-frame application** (`inject_noise_frame`): For each frame, checks which events are active. Active events contribute an inward erosion of the mask contour, modulated by:
   - A **spatial cosine window** along the contour arc (smooth tapering at the dent edges)
   - A **temporal cosine envelope** across frames (fade in → peak → fade out)

3. **Wrap-around**: Events that span past frame 359 wrap back to frame 0, maintaining continuity.

The GUI's **frame scrubber** lets you preview this temporal behaviour on a loaded sample mask. The "Jump to Peak" button takes you to the frame with maximum dent visibility.

## Augmentor (CLI)

Apply non-uniform scaling to STEP CAD files to create geometrically varied training data:

```bash
python augmentor.py -i drills
python augmentor.py -i drills --preset 1
python augmentor.py -i drills --sx 0.95 --sy 0.95 --sz 1.12
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `-i`, `--input` | Folder containing STEP files | *required* |
| `-o`, `--output` | Output root folder | `<input>_augmented` |
| `--preset` | Preset number (1–6, see table above) | none |
| `--sx` | Scale factor for X axis | 1.0 |
| `--sy` | Scale factor for Y axis | 1.0 |
| `--sz` | Scale factor for Z axis | 1.0 |

Output is written to a versioned `aug_NNN/` subfolder with `augmentation_config.json` recording the parameters used. The transform uses OpenCascade's `BRepBuilderAPI_GTransform` so the result is a proper B-Rep STEP file (not a facetted mesh).