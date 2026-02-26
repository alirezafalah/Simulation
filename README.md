# Simulation — Sim2Real Mask Generation

Generates synthetic binary mask datasets from 3D CAD models and injects realistic edge noise to bridge the sim-to-real gap.

## Project Structure

| File | Description |
|---|---|
| `render_engine.py` | 360° binary mask renderer (CLI) — loads STEP/STL, places a virtual camera at 250 mm working distance, outputs one mask per degree of rotation |
| `noise_injector.py` | Temporally-coherent contour-dent noise (CLI) — erodes the mask boundary at random locations with a cosine fade-in/fade-out across consecutive frames |
| `simulation_gui.py` | Desktop GUI (PySide6) — integrates both modules with live previews, frame scrubber, parameter tuning, and batch processing |
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

Apply temporally-coherent contour-dent noise to a folder of ordered mask frames:

```bash
python noise_injector.py -i output/2396N63_Carbide
python noise_injector.py -i output/2396N63_Carbide --n-events 1 --max-depth 3
python noise_injector.py -i output/2396N63_Carbide --frame-span-max 20 --span-min 100
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `-i`, `--input` | Directory of ordered mask frames (sorted by name) | *required* |
| `-o`, `--output` | Output directory | `<input>_noisy` |
| `--n-events` | Number of dent events per tool | 2 |
| `--frame-span-min` | Minimum consecutive frames each dent persists | 5 |
| `--frame-span-max` | Maximum consecutive frames each dent persists | 15 |
| `--span-min` | Minimum contour-pixel arc length per dent | 80 |
| `--span-max` | Maximum contour-pixel arc length per dent | 250 |
| `--max-depth` | Peak erosion depth in pixels | 2.5 |
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