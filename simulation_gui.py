#!/usr/bin/env python3
"""
simulation_gui.py — Dark-themed Desktop GUI for Sim2Real Mask Generation
==========================================================================

Integrates **render_engine** (360° binary mask rendering) and
**noise_injector** (realistic edge-noise degradation) into a single
PySide6 application with:

    • Two tabs: Render Settings & Noise Settings
    • Multi-CAD-file selection  (each tool → its own subfolder)
    • Separate output directories for clean and noisy masks
    • Debug preview button (off-screen single frame shown in the GUI)
    • Noise tab: live before / after preview, standalone folder batch
    • Click any preview thumbnail → full-resolution pop-up window
    • "Generate Dataset" button with two progress bars
    • All heavy work on QThread (GUI never freezes)

Dependencies
------------
    pip install PySide6 pyvista numpy opencv-python Pillow vtk

Usage
-----
    python simulation_gui.py
"""

from __future__ import annotations

import sys
import traceback
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# ── Qt ───────────────────────────────────────────────────────────────
from PySide6.QtCore import (
    Qt, Signal, Slot, QThread, QObject, QSize,
)
from PySide6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QFileDialog, QProgressBar,
    QListWidget, QAbstractItemView, QGroupBox, QTextEdit,
    QSplitter, QMessageBox, QSizePolicy, QDialog, QScrollArea,
    QSlider,
)

# ── Local modules (same folder) ─────────────────────────────────────
import render_engine as RE
import noise_injector as NI
import augmentor as AUG

_SCRIPT_DIR = Path(__file__).resolve().parent


# ═════════════════════════════════════════════════════════════════════
#  NUMPY ↔ QPixmap HELPERS
# ═════════════════════════════════════════════════════════════════════

def _gray_to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert an H×W uint8 grayscale array to a QPixmap."""
    h, w = arr.shape[:2]
    qimg = QImage(arr.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg)


def _rgb_to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert an H×W×3 uint8 RGB array to a QPixmap."""
    h, w, ch = arr.shape
    qimg = QImage(arr.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ═════════════════════════════════════════════════════════════════════
#  IMAGE POP-UP — click to expand any preview to a resizable window
# ═════════════════════════════════════════════════════════════════════

class ImagePopup(QDialog):
    """
    Resizable non-modal dialog showing a pixmap with scroll-wheel
    zoom.  Opened when the user clicks a preview thumbnail.
    """

    _ZOOM_MIN = 0.1
    _ZOOM_MAX = 10.0
    _ZOOM_STEP = 1.15          # 15 % per wheel notch

    def __init__(
        self,
        pixmap: QPixmap,
        title: str = "Preview",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(480, 360)
        self.setAttribute(Qt.WA_DeleteOnClose)   # clean up on close
        # Start at 80 % of pixmap size, capped to 1400×1000
        w = min(int(pixmap.width() * 0.8), 1400)
        h = min(int(pixmap.height() * 0.8), 1000)
        self.resize(max(w, 480), max(h, 360))

        self._source_pixmap = pixmap
        self._zoom = 1.0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setStyleSheet("background-color: #111;")

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setPixmap(pixmap)
        self._label.resize(pixmap.size())
        self._scroll.setWidget(self._label)
        lay.addWidget(self._scroll)

        self._hint = QLabel(self._hint_text())
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet("color: #888; font-size: 10px; padding: 2px;")
        lay.addWidget(self._hint)

    def _hint_text(self) -> str:
        pct = int(self._zoom * 100)
        return f"Scroll wheel to zoom ({pct} %)  •  Esc to close"

    def _apply_zoom(self) -> None:
        new_size = self._source_pixmap.size() * self._zoom
        scaled = self._source_pixmap.scaled(
            new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())
        self._hint.setText(self._hint_text())

    def wheelEvent(self, event):  # noqa: N802
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom = min(self._zoom * self._ZOOM_STEP, self._ZOOM_MAX)
        elif delta < 0:
            self._zoom = max(self._zoom / self._ZOOM_STEP, self._ZOOM_MIN)
        self._apply_zoom()
        event.accept()


class ClickablePreview(QLabel):
    """
    A QLabel that stores full-resolution pixmaps and opens a
    *non-modal* ImagePopup on click (so you can still interact
    with the main window while the popup is open).
    """
    clicked = Signal()

    def __init__(self, placeholder: str = "", parent: Optional[QWidget] = None):
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self._full_pixmap: Optional[QPixmap] = None
        self._popup_title: str = "Preview"
        self._popup: Optional[ImagePopup] = None

    def set_preview(self, pixmap: QPixmap, title: str = "Preview") -> None:
        """Store full pixmap and show scaled thumbnail."""
        self._full_pixmap = pixmap
        self._popup_title = title
        self._update_thumbnail()
        # Live-update the open popup if it still exists
        try:
            if self._popup is not None and self._popup.isVisible():
                self._popup._source_pixmap = pixmap
                self._popup._apply_zoom()
        except RuntimeError:
            # C++ object already deleted (user closed the popup)
            self._popup = None

    def _update_thumbnail(self) -> None:
        if self._full_pixmap is None:
            return
        scaled = self._full_pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._update_thumbnail()

    def mousePressEvent(self, event):  # noqa: N802
        if self._full_pixmap is not None:
            # Non-modal: .show() instead of .exec()
            self._popup = ImagePopup(
                self._full_pixmap, self._popup_title, self.window())
            self._popup.show()
        self.clicked.emit()


# ═════════════════════════════════════════════════════════════════════
#  WORKER — dataset generation on QThread
# ═════════════════════════════════════════════════════════════════════

class _WorkerSignals(QObject):
    tool_started   = Signal(int, str)
    frame_done     = Signal(int, int)
    tool_done      = Signal(int, int)
    log            = Signal(str)
    preview_ready  = Signal(object)          # numpy array
    finished       = Signal()
    error          = Signal(str)


class DatasetWorker(QThread):
    sig = _WorkerSignals()

    def __init__(
        self,
        model_paths: List[str],
        clean_output_dir: str,
        noisy_output_dir: str,
        render_cfg: Dict[str, Any],
        noise_cfgs: List[Dict[str, Any]],
        apply_noise: bool,
    ):
        super().__init__()
        self.model_paths      = model_paths
        self.clean_output_dir = clean_output_dir
        self.noisy_output_dir = noisy_output_dir
        self.render_cfg       = render_cfg
        self.noise_cfgs       = noise_cfgs      # list of 1+ noise configs
        self.apply_noise      = apply_noise
        self._abort           = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        import pyvista as pv
        import vtk
        from PIL import Image

        try:
            n_tools = len(self.model_paths)
            rc = self.render_cfg

            img_w  = int(rc["img_w"])
            img_h  = int(rc["img_h"])
            n_frames     = int(rc["n_frames"])
            tip_from_top = float(rc["tip_from_top"])
            working_dist = float(rc["working_distance"])
            img_format   = rc["image_format"].strip().upper()
            ext = "png" if img_format == "PNG" else "tiff"

            # Prepare versioned noisy output folders — one per noise config
            noisy_runs: List[Tuple[Path, Dict[str, Any]]] = []
            if self.apply_noise:
                for nc in self.noise_cfgs:
                    run_dir = NI._next_noise_run_dir(
                        Path(self.noisy_output_dir))
                    run_dir.mkdir(parents=True, exist_ok=True)
                    label = nc.get("preset_label", "")
                    self.sig.log.emit(
                        f"[NOISE] Versioned output → {run_dir.name}"
                        + (f"  ({label})" if label else ""))
                    noisy_runs.append((run_dir, nc))

            tool_names: list = []

            for ti, model_path in enumerate(self.model_paths):
                if self._abort:
                    self.sig.log.emit("[ABORT] Cancelled by user.")
                    break

                tool_name = RE.get_tool_name(model_path)
                self.sig.tool_started.emit(ti, tool_name)
                self.sig.log.emit(f"── Tool {ti+1}/{n_tools}: {tool_name} ──")
                tool_names.append(tool_name)

                self.sig.log.emit(f"[LOAD] {Path(model_path).name}")
                mesh = RE.load_mesh(model_path)
                mesh, tip_z, top_z, max_r = RE.preprocess_mesh(mesh)

                orig_wd  = RE.WORKING_DISTANCE_MM
                orig_tip = RE.TIP_FROM_TOP
                RE.WORKING_DISTANCE_MM = working_dist
                RE.TIP_FROM_TOP        = tip_from_top
                cam = RE.compute_camera(tip_z, top_z, max_r)
                RE.WORKING_DISTANCE_MM = orig_wd
                RE.TIP_FROM_TOP        = orig_tip

                pv.OFF_SCREEN = True
                plotter = pv.Plotter(off_screen=True,
                                     window_size=[img_w, img_h])
                plotter.set_background("black")
                plotter.disable_anti_aliasing()
                actor = plotter.add_mesh(
                    mesh, color="white",
                    ambient=1.0, diffuse=0.0, specular=0.0,
                    lighting=False, show_edges=False,
                )
                RE._apply_camera(plotter, cam)

                clean_dir = Path(self.clean_output_dir) / tool_name
                clean_dir.mkdir(parents=True, exist_ok=True)

                # Plan dent events for each noise preset
                run_data: List[Tuple[Path, list]] = []
                for (run_dir, nc) in noisy_runs:
                    nd = run_dir / tool_name
                    nd.mkdir(parents=True, exist_ok=True)
                    events = NI.plan_dent_events(
                        n_events=int(nc["n_events"]),
                        n_frames=n_frames,
                        frame_span_min=int(nc["frame_span_min"]),
                        frame_span_max=int(nc["frame_span_max"]),
                        dent_span_min=int(nc["dent_span_min"]),
                        dent_span_max=int(nc["dent_span_max"]),
                        max_depth=float(nc["max_depth"]),
                        seed=nc.get("seed"),
                    )
                    run_data.append((nd, events))

                for angle in range(n_frames):
                    if self._abort:
                        break

                    xform = vtk.vtkTransform()
                    xform.RotateZ(float(angle))
                    actor.SetUserTransform(xform)

                    plotter.render()
                    rgb  = plotter.screenshot(return_img=True)
                    mask = RE._to_binary_mask(rgb)

                    clean_path = str(clean_dir / f"mask_{angle:03d}.{ext}")
                    Image.fromarray(mask, mode="L").save(
                        clean_path, format=img_format)

                    # Apply each noise preset to same clean mask
                    for (noisy_dir, dent_events) in run_data:
                        noisy = NI.inject_noise_frame(
                            mask, dent_events, angle,
                            n_frames=n_frames,
                        )
                        noisy_path = str(
                            noisy_dir / f"mask_{angle:03d}.{ext}")
                        Image.fromarray(noisy, mode="L").save(
                            noisy_path, format=img_format)

                    self.sig.frame_done.emit(angle + 1, n_frames)

                plotter.close()
                self.sig.tool_done.emit(ti + 1, n_tools)

            # Write noise_config.json for each run folder
            for (run_dir, nc) in noisy_runs:
                NI._write_noise_config(
                    run_dir, nc, tool_names=tool_names)
                self.sig.log.emit(
                    f"[NOISE] Config saved: {run_dir / 'noise_config.json'}")

            self.sig.log.emit("[DONE] Dataset generation complete.")
        except Exception:
            self.sig.error.emit(traceback.format_exc())
        finally:
            self.sig.finished.emit()


class PreviewWorker(QThread):
    """Render a single debug frame for the render tab preview."""
    sig = _WorkerSignals()

    def __init__(self, model_path: str, render_cfg: Dict[str, Any]):
        super().__init__()
        self.model_path = model_path
        self.render_cfg = render_cfg

    def run(self) -> None:
        import pyvista as pv

        try:
            rc = self.render_cfg
            img_w  = int(rc["img_w"])
            img_h  = int(rc["img_h"])
            tip_from_top = float(rc["tip_from_top"])
            working_dist = float(rc["working_distance"])

            mesh = RE.load_mesh(self.model_path)
            mesh, tip_z, top_z, max_r = RE.preprocess_mesh(mesh)

            orig_wd  = RE.WORKING_DISTANCE_MM
            orig_tip = RE.TIP_FROM_TOP
            RE.WORKING_DISTANCE_MM = working_dist
            RE.TIP_FROM_TOP        = tip_from_top
            cam = RE.compute_camera(tip_z, top_z, max_r)
            RE.WORKING_DISTANCE_MM = orig_wd
            RE.TIP_FROM_TOP        = orig_tip

            pv.OFF_SCREEN = True
            plotter = pv.Plotter(off_screen=True,
                                 window_size=[img_w, img_h])
            plotter.set_background("black")
            plotter.disable_anti_aliasing()
            plotter.add_mesh(
                mesh, color="white",
                ambient=1.0, diffuse=0.0, specular=0.0,
                lighting=False, show_edges=False,
            )
            RE._apply_camera(plotter, cam)
            plotter.render()
            rgb = plotter.screenshot(return_img=True)
            plotter.close()

            self.sig.preview_ready.emit(rgb)
            self.sig.log.emit("[DEBUG] Preview rendered.")
        except Exception:
            self.sig.error.emit(traceback.format_exc())
        finally:
            self.sig.finished.emit()


class AugmentWorker(QThread):
    """Thread: augment all STEP files in a folder with versioned output."""
    sig = _WorkerSignals()

    def __init__(
        self,
        input_dir: str,
        output_root: str,
        aug_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.input_dir   = input_dir
        self.output_root = output_root
        self.aug_cfg     = aug_cfg
        self._abort      = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        try:
            ac = self.aug_cfg
            run_dir = AUG.augment_batch(
                self.input_dir,
                self.output_root,
                scale_x=float(ac["scale_x"]),
                scale_y=float(ac["scale_y"]),
                scale_z=float(ac["scale_z"]),
                progress_callback=lambda done, total: self.sig.frame_done.emit(done, total),
                log_callback=lambda msg: self.sig.log.emit(msg),
                abort_flag=self,
            )
            if run_dir is not None:
                self.sig.log.emit(
                    f"[DONE] Augmentation complete → {run_dir.name}")
        except Exception:
            self.sig.error.emit(traceback.format_exc())
        finally:
            self.sig.finished.emit()


class NoiseBatchWorker(QThread):
    """Standalone thread: apply noise to ALL tool subfolders, versioned output."""
    sig = _WorkerSignals()

    def __init__(
        self,
        clean_root: str,
        noisy_root: str,
        noise_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.clean_root = clean_root
        self.noisy_root = noisy_root
        self.noise_cfg  = noise_cfg
        self._abort     = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        try:
            nc = self.noise_cfg

            def _progress(ti, n_tools, fi, n_frames):
                # Emit combined progress: tool*frames linearised
                total = n_tools * n_frames
                done  = ti * n_frames + fi
                self.sig.frame_done.emit(done, total)

            run_dir = NI.inject_noise_batch_all(
                self.clean_root,
                self.noisy_root,
                n_events=int(nc["n_events"]),
                frame_span_min=int(nc["frame_span_min"]),
                frame_span_max=int(nc["frame_span_max"]),
                dent_span_min=int(nc["dent_span_min"]),
                dent_span_max=int(nc["dent_span_max"]),
                max_depth=float(nc["max_depth"]),
                seed=nc.get("seed"),
                progress_callback=_progress,
                log_callback=lambda msg: self.sig.log.emit(msg),
                abort_flag=self,
            )
            if run_dir is not None:
                self.sig.log.emit(
                    f"[DONE] Noise run complete → {run_dir.name}")
        except Exception:
            self.sig.error.emit(traceback.format_exc())
        finally:
            self.sig.finished.emit()


# ═════════════════════════════════════════════════════════════════════
#  DARK PALETTE
# ═════════════════════════════════════════════════════════════════════

def _apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window,          QColor(30, 30, 30))
    p.setColor(QPalette.WindowText,      QColor(208, 208, 208))
    p.setColor(QPalette.Base,            QColor(22, 22, 22))
    p.setColor(QPalette.AlternateBase,   QColor(35, 35, 35))
    p.setColor(QPalette.ToolTipBase,     QColor(40, 40, 40))
    p.setColor(QPalette.ToolTipText,     QColor(208, 208, 208))
    p.setColor(QPalette.Text,            QColor(208, 208, 208))
    p.setColor(QPalette.Button,          QColor(45, 45, 45))
    p.setColor(QPalette.ButtonText,      QColor(208, 208, 208))
    p.setColor(QPalette.BrightText,      QColor(255, 50, 50))
    p.setColor(QPalette.Link,            QColor(70, 150, 255))
    p.setColor(QPalette.Highlight,       QColor(70, 130, 200))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(p)
    app.setStyleSheet(
        "QToolTip { color: #d0d0d0; background-color: #2a2a2a; "
        "border: 1px solid #555; }"
    )


# ═════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═════════════════════════════════════════════════════════════════════

class SimulationGUI(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sim2Real Mask Generator")
        self.setMinimumSize(960, 780)
        self.resize(1080, 880)

        self._worker: Optional[DatasetWorker] = None
        self._preview_worker: Optional[PreviewWorker] = None
        self._noise_batch_worker: Optional[NoiseBatchWorker] = None

        # Cached sample mask for live noise preview
        self._sample_mask: Optional[np.ndarray] = None
        # Cached planned dent events for temporal preview
        self._preview_events: list = []
        self._preview_n_frames: int = 360

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────
    #  UI CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Tabs ─────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_render_tab(), "⚙  Render")
        self.tabs.addTab(self._build_noise_tab(),  "🔊  Noise")
        self.tabs.addTab(self._build_augment_tab(), "🔧  Augment")
        root.addWidget(self.tabs, stretch=1)

        # ── Progress bars ────────────────────────────────────────────
        prog_group = QGroupBox("Progress")
        prog_lay = QGridLayout(prog_group)

        prog_lay.addWidget(QLabel("Overall (Tools):"), 0, 0)
        self.prog_tools = QProgressBar()
        self.prog_tools.setTextVisible(True)
        self.prog_tools.setFormat("%v / %m  tools")
        prog_lay.addWidget(self.prog_tools, 0, 1)

        prog_lay.addWidget(QLabel("Current Tool (Frames):"), 1, 0)
        self.prog_frames = QProgressBar()
        self.prog_frames.setTextVisible(True)
        self.prog_frames.setFormat("%v / %m  frames")
        prog_lay.addWidget(self.prog_frames, 1, 1)

        root.addWidget(prog_group)

        # ── Log console ──────────────────────────────────────────────
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(110)
        self.log_box.setStyleSheet(
            "background-color: #111; color: #9acd32; "
            "font-family: Consolas, monospace; font-size: 11px;"
        )
        root.addWidget(self.log_box)

        # ── Buttons row ──────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.btn_generate = QPushButton("  Generate Dataset  ")
        self.btn_generate.setMinimumHeight(48)
        self.btn_generate.setStyleSheet(
            "QPushButton { background-color: #1a6b1a; color: white; "
            "font-size: 16px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #228B22; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.btn_generate.clicked.connect(self._on_generate)
        btn_row.addWidget(self.btn_generate)

        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setMinimumHeight(48)
        self.btn_abort.setEnabled(False)
        self.btn_abort.setStyleSheet(
            "QPushButton { background-color: #8b1a1a; color: white; "
            "font-size: 14px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #b22222; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.btn_abort.clicked.connect(self._on_abort)
        btn_row.addWidget(self.btn_abort)

        root.addLayout(btn_row)

    # ── RENDER TAB ───────────────────────────────────────────────────

    def _build_render_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(8)

        # --- CAD file list ---
        file_group = QGroupBox("CAD Models (.step / .stp / .stl)")
        fg_lay = QVBoxLayout(file_group)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setMinimumHeight(60)
        self.file_list.setMaximumHeight(100)
        fg_lay.addWidget(self.file_list)
        fbtn = QHBoxLayout()
        btn_add = QPushButton("Add Files …")
        btn_add.clicked.connect(self._add_cad_files)
        btn_rem = QPushButton("Remove Selected")
        btn_rem.clicked.connect(self._remove_cad_files)
        fbtn.addWidget(btn_add)
        fbtn.addWidget(btn_rem)
        fg_lay.addLayout(fbtn)
        lay.addWidget(file_group)

        # --- Output directory ---
        out_group = QGroupBox("Output Directory (clean masks)")
        og_lay = QHBoxLayout(out_group)
        self.render_out_edit = QLineEdit(str(_SCRIPT_DIR / "output"))
        og_lay.addWidget(self.render_out_edit, stretch=1)
        btn_br = QPushButton("Browse …")
        btn_br.clicked.connect(
            lambda: self._browse_dir(self.render_out_edit))
        og_lay.addWidget(btn_br)
        lay.addWidget(out_group)

        # --- Camera / render params ---
        cam_group = QGroupBox("Camera & Render Parameters")
        form = QFormLayout(cam_group)

        self.spin_wd = QDoubleSpinBox()
        self.spin_wd.setRange(10, 5000)
        self.spin_wd.setValue(RE.WORKING_DISTANCE_MM)
        self.spin_wd.setSuffix("  mm")
        self.spin_wd.setDecimals(1)
        form.addRow("Working Distance:", self.spin_wd)

        self.spin_tip = QDoubleSpinBox()
        self.spin_tip.setRange(0.01, 0.99)
        self.spin_tip.setValue(RE.TIP_FROM_TOP)
        self.spin_tip.setSingleStep(0.05)
        self.spin_tip.setDecimals(2)
        self.spin_tip.setToolTip(
            "Fraction from the TOP of the frame where the drill tip sits.\n"
            "0.80 = tip at 80 % from top (near bottom)."
        )
        form.addRow("Tip Offset (from top):", self.spin_tip)

        self.spin_w = QSpinBox()
        self.spin_w.setRange(320, 8192)
        self.spin_w.setValue(RE.IMG_W)
        self.spin_w.setSingleStep(64)
        self.spin_w.setSuffix("  px")
        form.addRow("Image Width:", self.spin_w)

        self.spin_h = QSpinBox()
        self.spin_h.setRange(240, 8192)
        self.spin_h.setValue(RE.IMG_H)
        self.spin_h.setSingleStep(64)
        self.spin_h.setSuffix("  px")
        form.addRow("Image Height:", self.spin_h)

        self.spin_frames = QSpinBox()
        self.spin_frames.setRange(1, 3600)
        self.spin_frames.setValue(RE.N_FRAMES)
        form.addRow("Frames (rotation):", self.spin_frames)

        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems(["TIFF", "PNG"])
        form.addRow("Image Format:", self.combo_fmt)

        lay.addWidget(cam_group)

        # --- Debug preview (clickable → full-res popup) ---
        prev_group = QGroupBox(
            "Render Preview  (click image to expand full-resolution)")
        prev_lay = QVBoxLayout(prev_group)
        self.render_preview = ClickablePreview(
            "No preview yet.  Select a CAD file and click Debug Preview.")
        self.render_preview.setMinimumHeight(160)
        self.render_preview.setStyleSheet(
            "background-color: #111; border: 1px solid #333; "
            "border-radius: 4px;"
        )
        prev_lay.addWidget(self.render_preview)

        self.btn_debug = QPushButton("  Debug Preview (selected file)  ")
        self.btn_debug.setMinimumHeight(34)
        self.btn_debug.setStyleSheet(
            "QPushButton { background-color: #1a4a6b; color: white; "
            "font-size: 13px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #1e6090; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.btn_debug.clicked.connect(self._on_debug_preview)
        prev_lay.addWidget(self.btn_debug)
        lay.addWidget(prev_group, stretch=1)

        return tab

    # ── NOISE TAB ────────────────────────────────────────────────────

    def _build_noise_tab(self) -> QWidget:
        # Wrap everything in a QScrollArea so it works on small monitors
        tab = QScrollArea()
        tab.setWidgetResizable(True)
        tab.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setSpacing(8)

        # --- Enable / disable during dataset generation ---
        self.chk_noise = QCheckBox(
            "Apply current noise preset during generation "
            "(single noise_NNN folder)")
        self.chk_noise.setChecked(True)
        self.chk_noise.setToolTip(
            "Applies the noise preset currently selected above\n"
            "(or your custom values) to each rendered frame,\n"
            "saving results in one versioned noise_NNN folder.")
        self.chk_noise.setStyleSheet("font-size: 13px; font-weight: bold;")
        lay.addWidget(self.chk_noise)

        # --- Apply all presets checkbox ---
        self.chk_all_noise_presets = QCheckBox(
            "Apply ALL noise presets during generation "
            "(one noise_NNN folder per preset)")
        self.chk_all_noise_presets.setChecked(False)
        self.chk_all_noise_presets.setToolTip(
            "When checked, Generate Dataset renders clean masks once\n"
            "then applies EVERY noise preset (Default / Moderate /\n"
            "Aggressive), saving each to its own versioned noise_NNN\n"
            "subfolder.  Overrides the single-preset checkbox above.")
        self.chk_all_noise_presets.setStyleSheet(
            "font-size: 12px; color: #bbb;")
        lay.addWidget(self.chk_all_noise_presets)

        # --- Noisy output dir (used by Generate Dataset) ---
        nout_group = QGroupBox("Noisy Output Directory (dataset generation)")
        no_lay = QHBoxLayout(nout_group)
        self.noisy_out_edit = QLineEdit(str(_SCRIPT_DIR / "output_noisy"))
        no_lay.addWidget(self.noisy_out_edit, stretch=1)
        btn_bn = QPushButton("Browse …")
        btn_bn.clicked.connect(
            lambda: self._browse_dir(self.noisy_out_edit))
        no_lay.addWidget(btn_bn)
        lay.addWidget(nout_group)

        # --- Noise preset selector ---
        preset_group = QGroupBox("Noise Preset")
        ps_lay = QFormLayout(preset_group)

        self.combo_noise_preset = QComboBox()
        self.combo_noise_preset.addItems(NI.NOISE_PRESET_NAMES)
        self.combo_noise_preset.addItem("Custom")
        self.combo_noise_preset.setToolTip(
            "Select a preset to auto-fill all noise parameters,\n"
            "or choose 'Custom' to set values manually.")
        self.combo_noise_preset.currentIndexChanged.connect(
            self._on_noise_preset_changed)
        ps_lay.addRow("Preset:", self.combo_noise_preset)
        lay.addWidget(preset_group)

        # --- Noise parameters ---
        noise_group = QGroupBox("Dent Event Parameters (temporally-coherent contour erosion)")
        form = QFormLayout(noise_group)

        self.spin_n_events = QSpinBox()
        self.spin_n_events.setRange(0, 10)
        self.spin_n_events.setValue(NI.N_EVENTS)
        self.spin_n_events.setToolTip(
            "Number of dent events per tool (1\u20132 typical).\n"
            "Each event appears at a random contour position\n"
            "for several consecutive frames, then fades away."
        )
        form.addRow("Events per Tool:", self.spin_n_events)

        self.spin_frame_span_min = QSpinBox()
        self.spin_frame_span_min.setRange(1, 100)
        self.spin_frame_span_min.setValue(NI.FRAME_SPAN_MIN)
        self.spin_frame_span_min.setSuffix("  frames")
        self.spin_frame_span_min.setToolTip(
            "Minimum number of consecutive frames each\n"
            "dent event persists (cosine fade in/out)."
        )
        form.addRow("Frame Span Min:", self.spin_frame_span_min)

        self.spin_frame_span_max = QSpinBox()
        self.spin_frame_span_max.setRange(1, 200)
        self.spin_frame_span_max.setValue(NI.FRAME_SPAN_MAX)
        self.spin_frame_span_max.setSuffix("  frames")
        self.spin_frame_span_max.setToolTip(
            "Maximum number of consecutive frames each\n"
            "dent event persists."
        )
        form.addRow("Frame Span Max:", self.spin_frame_span_max)

        self.spin_span_min = QSpinBox()
        self.spin_span_min.setRange(10, 1000)
        self.spin_span_min.setValue(NI.DENT_SPAN_MIN)
        self.spin_span_min.setSuffix("  contour px")
        self.spin_span_min.setToolTip(
            "Minimum arc length (contour pixels) of each dent."
        )
        form.addRow("Dent Span Min:", self.spin_span_min)

        self.spin_span_max = QSpinBox()
        self.spin_span_max.setRange(10, 2000)
        self.spin_span_max.setValue(NI.DENT_SPAN_MAX)
        self.spin_span_max.setSuffix("  contour px")
        self.spin_span_max.setToolTip(
            "Maximum arc length (contour pixels) of each dent."
        )
        form.addRow("Dent Span Max:", self.spin_span_max)

        self.spin_max_depth = QDoubleSpinBox()
        self.spin_max_depth.setRange(0.5, 10.0)
        self.spin_max_depth.setValue(NI.MAX_DEPTH)
        self.spin_max_depth.setSingleStep(0.5)
        self.spin_max_depth.setDecimals(1)
        self.spin_max_depth.setSuffix("  px")
        self.spin_max_depth.setToolTip(
            "Peak inward erosion depth per dent (pixels).\n"
            "Each event picks a random depth in [1.0, max_depth].\n"
            "1\u20132 = subtle,  2\u20133 = visible,  3+ = aggressive."
        )
        form.addRow("Max Depth:", self.spin_max_depth)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 2_147_483_647)
        self.spin_seed.setValue(-1)
        self.spin_seed.setToolTip("RNG seed.  -1 = non-deterministic.")
        form.addRow("Seed (-1 = random):", self.spin_seed)

        lay.addWidget(noise_group)

        # --- Live before / after preview ---
        prev_group = QGroupBox(
            "Noise Preview  (click either image to expand full-resolution)")
        prev_lay = QVBoxLayout(prev_group)

        # Load sample mask button
        load_row = QHBoxLayout()
        self.btn_load_mask = QPushButton("Load Sample Mask …")
        self.btn_load_mask.setToolTip(
            "Pick any clean binary mask PNG/TIFF to preview noise on.")
        self.btn_load_mask.clicked.connect(self._on_load_sample_mask)
        self.noise_mask_path_label = QLabel("No mask loaded")
        self.noise_mask_path_label.setStyleSheet("color: #888;")
        load_row.addWidget(self.btn_load_mask)
        load_row.addWidget(self.noise_mask_path_label, stretch=1)
        prev_lay.addLayout(load_row)

        # Frame scrubber for temporal preview
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Preview frame:"))

        self.noise_frame_slider = QSlider(Qt.Horizontal)
        self.noise_frame_slider.setRange(0, 359)
        self.noise_frame_slider.setValue(0)
        self.noise_frame_slider.setToolTip(
            "Scrub through frames to see the temporal dent\n"
            "fade in / peak / fade out on the loaded mask.")
        frame_row.addWidget(self.noise_frame_slider, stretch=1)

        self.noise_frame_spin = QSpinBox()
        self.noise_frame_spin.setRange(0, 359)
        self.noise_frame_spin.setValue(0)
        self.noise_frame_spin.setSuffix(" / 359")
        self.noise_frame_spin.setFixedWidth(110)
        frame_row.addWidget(self.noise_frame_spin)

        self.btn_jump_peak = QPushButton("Jump to Peak")
        self.btn_jump_peak.setToolTip(
            "Jump to the frame where the first dent event\n"
            "reaches maximum depth.")
        self.btn_jump_peak.setFixedWidth(110)
        frame_row.addWidget(self.btn_jump_peak)

        prev_lay.addLayout(frame_row)

        # Keep slider ↔ spinbox in sync
        self.noise_frame_slider.valueChanged.connect(
            self.noise_frame_spin.setValue)
        self.noise_frame_spin.valueChanged.connect(
            self.noise_frame_slider.setValue)
        # Scrubbing updates the noisy preview
        self.noise_frame_spin.valueChanged.connect(
            self._update_noise_preview)
        self.btn_jump_peak.clicked.connect(self._jump_to_peak_frame)

        # Side-by-side: Original | Noisy
        side = QHBoxLayout()

        orig_box = QVBoxLayout()
        orig_title = QLabel("Original (clean)")
        orig_title.setAlignment(Qt.AlignCenter)
        orig_title.setStyleSheet("font-weight: bold; color: #aaa;")
        orig_box.addWidget(orig_title)
        self.noise_orig_preview = ClickablePreview("—")
        self.noise_orig_preview.setMinimumHeight(140)
        self.noise_orig_preview.setStyleSheet(
            "background-color: #111; border: 1px solid #333; "
            "border-radius: 4px;"
        )
        orig_box.addWidget(self.noise_orig_preview, stretch=1)
        side.addLayout(orig_box)

        noisy_box = QVBoxLayout()
        noisy_title = QLabel("After noise")
        noisy_title.setAlignment(Qt.AlignCenter)
        noisy_title.setStyleSheet("font-weight: bold; color: #aaa;")
        noisy_box.addWidget(noisy_title)
        self.noise_result_preview = ClickablePreview("—")
        self.noise_result_preview.setMinimumHeight(140)
        self.noise_result_preview.setStyleSheet(
            "background-color: #111; border: 1px solid #333; "
            "border-radius: 4px;"
        )
        noisy_box.addWidget(self.noise_result_preview, stretch=1)
        side.addLayout(noisy_box)

        prev_lay.addLayout(side, stretch=1)
        lay.addWidget(prev_group, stretch=1)

        # Connect noise param widgets → replan events (which auto-updates preview)
        self.spin_n_events.valueChanged.connect(self._replan_noise_events)
        self.spin_frame_span_min.valueChanged.connect(self._replan_noise_events)
        self.spin_frame_span_max.valueChanged.connect(self._replan_noise_events)
        self.spin_span_min.valueChanged.connect(self._replan_noise_events)
        self.spin_span_max.valueChanged.connect(self._replan_noise_events)
        self.spin_max_depth.valueChanged.connect(self._replan_noise_events)
        self.spin_seed.valueChanged.connect(self._replan_noise_events)

        # When user edits a spinbox manually, switch combo to "Custom"
        for spin in (self.spin_n_events, self.spin_frame_span_min,
                     self.spin_frame_span_max, self.spin_span_min,
                     self.spin_span_max, self.spin_max_depth, self.spin_seed):
            spin.valueChanged.connect(self._noise_param_manual_change)

        # ── Standalone batch section ─────────────────────────────────
        batch_group = QGroupBox("Standalone Batch — Apply Noise to Folder")
        bg_lay = QVBoxLayout(batch_group)

        dir_row = QGridLayout()
        dir_row.addWidget(QLabel("Input folder:"), 0, 0)
        self.noise_batch_in = QLineEdit(str(_SCRIPT_DIR / "output"))
        dir_row.addWidget(self.noise_batch_in, 0, 1)
        btn_bi = QPushButton("Browse …")
        btn_bi.clicked.connect(
            lambda: self._browse_dir(self.noise_batch_in))
        dir_row.addWidget(btn_bi, 0, 2)

        dir_row.addWidget(QLabel("Noisy output root:"), 1, 0)
        self.noise_batch_out = QLineEdit(str(_SCRIPT_DIR / "output_noisy"))
        self.noise_batch_out.setToolTip(
            "Parent folder where versioned noise_NNN subfolders are created.")
        dir_row.addWidget(self.noise_batch_out, 1, 1)
        btn_bo = QPushButton("Browse …")
        btn_bo.clicked.connect(
            lambda: self._browse_dir(self.noise_batch_out))
        dir_row.addWidget(btn_bo, 1, 2)
        bg_lay.addLayout(dir_row)

        self.prog_noise_batch = QProgressBar()
        self.prog_noise_batch.setTextVisible(True)
        self.prog_noise_batch.setFormat("%v / %m  masks")
        bg_lay.addWidget(self.prog_noise_batch)

        nbtn_row = QHBoxLayout()
        self.btn_noise_batch = QPushButton("  🚀  Run Noise on All Tools  ")
        self.btn_noise_batch.setMinimumHeight(36)
        self.btn_noise_batch.setStyleSheet(
            "QPushButton { background-color: #5a3a8a; color: white; "
            "font-size: 13px; font-weight: bold; border-radius: 5px; }"
            "QPushButton:hover { background-color: #7244b0; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.btn_noise_batch.clicked.connect(self._on_noise_batch)
        nbtn_row.addWidget(self.btn_noise_batch)

        self.btn_noise_batch_abort = QPushButton("Abort")
        self.btn_noise_batch_abort.setEnabled(False)
        self.btn_noise_batch_abort.setMinimumHeight(36)
        self.btn_noise_batch_abort.setStyleSheet(
            "QPushButton { background-color: #8b1a1a; color: white; "
            "font-size: 12px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #b22222; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.btn_noise_batch_abort.clicked.connect(self._on_noise_batch_abort)
        nbtn_row.addWidget(self.btn_noise_batch_abort)
        bg_lay.addLayout(nbtn_row)

        lay.addWidget(batch_group)

        tab.setWidget(inner)
        return tab

    # ── AUGMENTATION TAB ────────────────────────────────────────────

    def _build_augment_tab(self) -> QWidget:
        tab = QScrollArea()
        tab.setWidgetResizable(True)
        tab.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setSpacing(8)

        # --- Info banner ---
        info = QLabel(
            "<b>CAD Geometry Augmentation</b> — apply non-uniform scaling "
            "to STEP files to create geometrically diverse training data.\n"
            "Output is a versioned <code>aug_NNN/</code> subfolder with "
            "<code>augmentation_config.json</code>."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; padding: 4px;")
        lay.addWidget(info)

        # --- Preset selector ---
        preset_group = QGroupBox("Augmentation Preset")
        pg_lay = QFormLayout(preset_group)

        self.combo_aug_preset = QComboBox()
        self.combo_aug_preset.addItems(AUG.PRESET_NAMES)
        self.combo_aug_preset.addItem("Custom")
        self.combo_aug_preset.setToolTip(
            "Select a preset to auto-fill Scale X/Y/Z,\n"
            "or choose 'Custom' to set values manually.")
        self.combo_aug_preset.currentIndexChanged.connect(
            self._on_aug_preset_changed)
        pg_lay.addRow("Preset:", self.combo_aug_preset)
        lay.addWidget(preset_group)

        # --- Scale parameters ---
        scale_group = QGroupBox("Scale Factors")
        sf_lay = QFormLayout(scale_group)

        self.spin_aug_sx = QDoubleSpinBox()
        self.spin_aug_sx.setRange(0.50, 2.00)
        self.spin_aug_sx.setSingleStep(0.01)
        self.spin_aug_sx.setDecimals(3)
        self.spin_aug_sx.setToolTip("Scale factor for the X axis.")
        sf_lay.addRow("Scale X:", self.spin_aug_sx)

        self.spin_aug_sy = QDoubleSpinBox()
        self.spin_aug_sy.setRange(0.50, 2.00)
        self.spin_aug_sy.setSingleStep(0.01)
        self.spin_aug_sy.setDecimals(3)
        self.spin_aug_sy.setToolTip("Scale factor for the Y axis.")
        sf_lay.addRow("Scale Y:", self.spin_aug_sy)

        self.spin_aug_sz = QDoubleSpinBox()
        self.spin_aug_sz.setRange(0.50, 2.00)
        self.spin_aug_sz.setSingleStep(0.01)
        self.spin_aug_sz.setDecimals(3)
        self.spin_aug_sz.setToolTip(
            "Scale factor for the Z axis (drill length axis).")
        sf_lay.addRow("Scale Z:", self.spin_aug_sz)

        lay.addWidget(scale_group)

        # Fill initial preset values
        self._on_aug_preset_changed(0)

        # --- Input / output directories ---
        dir_group = QGroupBox("Directories")
        dg_lay = QGridLayout(dir_group)

        dg_lay.addWidget(QLabel("STEP input folder:"), 0, 0)
        self.aug_input_edit = QLineEdit(str(_SCRIPT_DIR / "drills"))
        self.aug_input_edit.setToolTip(
            "Folder containing .step / .stp files to augment.")
        dg_lay.addWidget(self.aug_input_edit, 0, 1)
        btn_ai = QPushButton("Browse …")
        btn_ai.clicked.connect(
            lambda: self._browse_dir(self.aug_input_edit))
        dg_lay.addWidget(btn_ai, 0, 2)

        dg_lay.addWidget(QLabel("Augmented output root:"), 1, 0)
        self.aug_output_edit = QLineEdit(
            str(_SCRIPT_DIR / "drills_augmented"))
        self.aug_output_edit.setToolTip(
            "Parent folder where versioned aug_NNN/ subfolders are created.")
        dg_lay.addWidget(self.aug_output_edit, 1, 1)
        btn_ao = QPushButton("Browse …")
        btn_ao.clicked.connect(
            lambda: self._browse_dir(self.aug_output_edit))
        dg_lay.addWidget(btn_ao, 1, 2)

        lay.addWidget(dir_group)

        # --- Progress bar ---
        self.prog_augment = QProgressBar()
        self.prog_augment.setTextVisible(True)
        self.prog_augment.setFormat("%v / %m  STEP files")
        lay.addWidget(self.prog_augment)

        # --- Action buttons ---
        abtn_row = QHBoxLayout()

        self.btn_augment = QPushButton(
            "  🔧  Augment All STEP Files  ")
        self.btn_augment.setMinimumHeight(36)
        self.btn_augment.setStyleSheet(
            "QPushButton { background-color: #1a5a7a; color: white; "
            "font-size: 13px; font-weight: bold; border-radius: 5px; }"
            "QPushButton:hover { background-color: #2080b0; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.btn_augment.clicked.connect(self._on_augment)
        abtn_row.addWidget(self.btn_augment)

        self.btn_augment_abort = QPushButton("Abort")
        self.btn_augment_abort.setEnabled(False)
        self.btn_augment_abort.setMinimumHeight(36)
        self.btn_augment_abort.setStyleSheet(
            "QPushButton { background-color: #8b1a1a; color: white; "
            "font-size: 12px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #b22222; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.btn_augment_abort.clicked.connect(self._on_augment_abort)
        abtn_row.addWidget(self.btn_augment_abort)

        lay.addLayout(abtn_row)

        # Stretch at bottom
        lay.addStretch(1)

        tab.setWidget(inner)
        return tab

    # ─────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _browse_dir(self, line_edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select Directory", line_edit.text())
        if d:
            line_edit.setText(d)

    def _add_cad_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select CAD Models",
            str(_SCRIPT_DIR / "drills"),
            "CAD Files (*.step *.stp *.stl);;All Files (*)",
        )
        existing = {self.file_list.item(i).text()
                    for i in range(self.file_list.count())}
        for f in files:
            if f not in existing:
                self.file_list.addItem(f)

    def _remove_cad_files(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def _collect_render_cfg(self) -> Dict[str, Any]:
        return dict(
            img_w=self.spin_w.value(),
            img_h=self.spin_h.value(),
            n_frames=self.spin_frames.value(),
            tip_from_top=self.spin_tip.value(),
            working_distance=self.spin_wd.value(),
            image_format=self.combo_fmt.currentText(),
        )

    def _collect_noise_cfg(self) -> Dict[str, Any]:
        seed_val = self.spin_seed.value()
        return dict(
            n_events=self.spin_n_events.value(),
            frame_span_min=self.spin_frame_span_min.value(),
            frame_span_max=self.spin_frame_span_max.value(),
            dent_span_min=self.spin_span_min.value(),
            dent_span_max=self.spin_span_max.value(),
            max_depth=self.spin_max_depth.value(),
            seed=None if seed_val < 0 else seed_val,
        )

    def _log(self, text: str) -> None:
        self.log_box.append(text)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_busy(self, busy: bool) -> None:
        self.btn_generate.setEnabled(not busy)
        self.btn_abort.setEnabled(busy)
        self.btn_debug.setEnabled(not busy)
        self.tabs.setEnabled(not busy)

    # ─────────────────────────────────────────────────────────────────
    #  RENDER DEBUG PREVIEW
    # ─────────────────────────────────────────────────────────────────

    @Slot()
    def _on_debug_preview(self) -> None:
        sel = self.file_list.selectedItems()
        if not sel:
            if self.file_list.count() == 0:
                QMessageBox.warning(self, "No files",
                                    "Add at least one CAD file first.")
                return
            model_path = self.file_list.item(0).text()
        else:
            model_path = sel[0].text()

        self.btn_debug.setEnabled(False)
        self.render_preview.setText("Rendering preview …")
        self._log(f"[DEBUG] Rendering preview for {Path(model_path).name} …")

        self._preview_worker = PreviewWorker(
            model_path, self._collect_render_cfg())
        self._preview_worker.sig.preview_ready.connect(
            self._show_render_preview)
        self._preview_worker.sig.log.connect(self._log)
        self._preview_worker.sig.error.connect(self._on_error)
        self._preview_worker.sig.finished.connect(
            lambda: self.btn_debug.setEnabled(True))
        self._preview_worker.start()

    @Slot(object)
    def _show_render_preview(self, rgb: np.ndarray) -> None:
        pm = _rgb_to_pixmap(rgb)
        self.render_preview.set_preview(pm, "Render Preview (full resolution)")

    # ─────────────────────────────────────────────────────────────────
    #  NOISE PRESET
    # ─────────────────────────────────────────────────────────────────

    _noise_preset_updating: bool = False   # guard flag

    @Slot(int)
    def _on_noise_preset_changed(self, idx: int) -> None:
        """Fill noise spinboxes from the selected preset."""
        if idx < len(NI.NOISE_PRESET_NAMES):
            p = NI.NOISE_PRESETS[NI.NOISE_PRESET_NAMES[idx]]
            self._noise_preset_updating = True
            self.spin_n_events.setValue(int(p["n_events"]))
            self.spin_frame_span_min.setValue(int(p["frame_span_min"]))
            self.spin_frame_span_max.setValue(int(p["frame_span_max"]))
            self.spin_span_min.setValue(int(p["dent_span_min"]))
            self.spin_span_max.setValue(int(p["dent_span_max"]))
            self.spin_max_depth.setValue(float(p["max_depth"]))
            seed_v = p.get("seed")
            self.spin_seed.setValue(-1 if seed_v is None else seed_v)
            self._noise_preset_updating = False
            self._replan_noise_events()

    @Slot()
    def _noise_param_manual_change(self) -> None:
        """Switch combo to 'Custom' when user edits a spinbox manually."""
        if not self._noise_preset_updating:
            custom_idx = len(NI.NOISE_PRESET_NAMES)   # last item
            if self.combo_noise_preset.currentIndex() != custom_idx:
                self.combo_noise_preset.blockSignals(True)
                self.combo_noise_preset.setCurrentIndex(custom_idx)
                self.combo_noise_preset.blockSignals(False)

    # ─────────────────────────────────────────────────────────────────
    #  NOISE LIVE PREVIEW
    # ─────────────────────────────────────────────────────────────────

    @Slot()
    def _on_load_sample_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a clean binary mask",
            str(_SCRIPT_DIR / "output"),
            "Images (*.png *.tiff *.tif);;All Files (*)",
        )
        if not path:
            return

        from PIL import Image
        img = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        img = np.where(img > 127, 255, 0).astype(np.uint8)
        self._sample_mask = img

        name = Path(path).name
        self.noise_mask_path_label.setText(name)
        self._log(f"[NOISE] Loaded sample mask: {name}")

        # Show original
        pm_orig = _gray_to_pixmap(img)
        self.noise_orig_preview.set_preview(pm_orig, f"Original — {name}")

        # Plan events and jump to the peak frame
        self._replan_noise_events()

    @Slot()
    def _replan_noise_events(self) -> None:
        """Re-plan temporal dent events from current GUI params,
        auto-jump the frame slider to the peak frame, and refresh preview."""
        nc = self._collect_noise_cfg()
        n_frames = self._preview_n_frames

        # Update slider/spinbox range to match n_frames
        self.noise_frame_slider.setRange(0, max(0, n_frames - 1))
        self.noise_frame_spin.setRange(0, max(0, n_frames - 1))
        self.noise_frame_spin.setSuffix(f" / {n_frames - 1}")

        self._preview_events = NI.plan_dent_events(
            n_events=int(nc["n_events"]),
            n_frames=n_frames,
            frame_span_min=int(nc["frame_span_min"]),
            frame_span_max=int(nc["frame_span_max"]),
            dent_span_min=int(nc["dent_span_min"]),
            dent_span_max=int(nc["dent_span_max"]),
            max_depth=float(nc["max_depth"]),
            seed=nc.get("seed"),
        )

        # Auto-jump to peak frame of first event
        if self._preview_events:
            peak = self._preview_events[0].peak_frame % n_frames
            self.noise_frame_spin.setValue(peak)   # triggers _update_noise_preview
        else:
            self._update_noise_preview()

    @Slot()
    def _jump_to_peak_frame(self) -> None:
        """Jump the frame slider to the peak frame of the first event."""
        if self._preview_events:
            peak = self._preview_events[0].peak_frame % self._preview_n_frames
            self.noise_frame_spin.setValue(peak)

    @Slot()
    def _update_noise_preview(self) -> None:
        """Apply temporal noise at the current preview frame and refresh."""
        if self._sample_mask is None:
            return

        frame_idx = self.noise_frame_spin.value()
        n_frames = self._preview_n_frames

        if self._preview_events:
            noisy = NI.inject_noise_frame(
                self._sample_mask,
                self._preview_events,
                frame_idx,
                n_frames=n_frames,
            )
        else:
            noisy = self._sample_mask.copy()

        pm = _gray_to_pixmap(noisy)
        self.noise_result_preview.set_preview(
            pm, f"Noisy — frame {frame_idx}")

    # ─────────────────────────────────────────────────────────────────
    #  STANDALONE NOISE BATCH
    # ─────────────────────────────────────────────────────────────────

    @Slot()
    def _on_noise_batch(self) -> None:
        inp = self.noise_batch_in.text().strip()
        out = self.noise_batch_out.text().strip()
        if not inp or not Path(inp).is_dir():
            QMessageBox.warning(
                self, "Invalid input",
                "Select a valid clean mask root folder\n"
                "(parent directory containing tool subfolders).")
            return

        self.btn_noise_batch.setEnabled(False)
        self.btn_noise_batch_abort.setEnabled(True)
        self.prog_noise_batch.setValue(0)

        self._noise_batch_worker = NoiseBatchWorker(
            inp, out, self._collect_noise_cfg())
        self._noise_batch_worker.sig.frame_done.connect(
            self._on_noise_batch_progress)
        self._noise_batch_worker.sig.log.connect(self._log)
        self._noise_batch_worker.sig.error.connect(self._on_error)
        self._noise_batch_worker.sig.finished.connect(
            self._on_noise_batch_finished)
        self._noise_batch_worker.start()

    @Slot()
    def _on_noise_batch_abort(self) -> None:
        if self._noise_batch_worker is not None:
            self._noise_batch_worker.abort()
            self._log("[USER] Noise batch abort requested …")

    @Slot(int, int)
    def _on_noise_batch_progress(self, done: int, total: int) -> None:
        self.prog_noise_batch.setMaximum(total)
        self.prog_noise_batch.setValue(done)

    @Slot()
    def _on_noise_batch_finished(self) -> None:
        self.btn_noise_batch.setEnabled(True)
        self.btn_noise_batch_abort.setEnabled(False)
        self._log("── Noise batch finished ──")

    # ─────────────────────────────────────────────────────────────────
    #  AUGMENTATION HANDLERS
    # ─────────────────────────────────────────────────────────────────

    @Slot(int)
    def _on_aug_preset_changed(self, idx: int) -> None:
        """Fill scale spinboxes from the selected preset."""
        if idx < len(AUG.PRESET_NAMES):
            name = AUG.PRESET_NAMES[idx]
            sx, sy, sz = AUG.PRESETS[name]
            self.spin_aug_sx.setValue(sx)
            self.spin_aug_sy.setValue(sy)
            self.spin_aug_sz.setValue(sz)
            self.spin_aug_sx.setEnabled(False)
            self.spin_aug_sy.setEnabled(False)
            self.spin_aug_sz.setEnabled(False)
        else:
            # "Custom" selected — unlock spinboxes
            self.spin_aug_sx.setEnabled(True)
            self.spin_aug_sy.setEnabled(True)
            self.spin_aug_sz.setEnabled(True)

    def _collect_aug_cfg(self) -> Dict[str, Any]:
        idx = self.combo_aug_preset.currentIndex()
        preset_name = (
            AUG.PRESET_NAMES[idx]
            if idx < len(AUG.PRESET_NAMES)
            else "Custom"
        )
        return dict(
            preset=preset_name,
            scale_x=self.spin_aug_sx.value(),
            scale_y=self.spin_aug_sy.value(),
            scale_z=self.spin_aug_sz.value(),
        )

    @Slot()
    def _on_augment(self) -> None:
        inp = self.aug_input_edit.text().strip()
        out = self.aug_output_edit.text().strip()
        if not inp or not Path(inp).is_dir():
            QMessageBox.warning(
                self, "Invalid input",
                "Select a valid folder containing STEP files.")
            return

        self.btn_augment.setEnabled(False)
        self.btn_augment_abort.setEnabled(True)
        self.prog_augment.setValue(0)

        self._augment_worker = AugmentWorker(
            inp, out, self._collect_aug_cfg())
        self._augment_worker.sig.frame_done.connect(
            self._on_augment_progress)
        self._augment_worker.sig.log.connect(self._log)
        self._augment_worker.sig.error.connect(self._on_error)
        self._augment_worker.sig.finished.connect(
            self._on_augment_finished)
        self._augment_worker.start()

    @Slot()
    def _on_augment_abort(self) -> None:
        if hasattr(self, '_augment_worker') and self._augment_worker is not None:
            self._augment_worker.abort()
            self._log("[USER] Augmentation abort requested …")

    @Slot(int, int)
    def _on_augment_progress(self, done: int, total: int) -> None:
        self.prog_augment.setMaximum(total)
        self.prog_augment.setValue(done)

    @Slot()
    def _on_augment_finished(self) -> None:
        self.btn_augment.setEnabled(True)
        self.btn_augment_abort.setEnabled(False)
        self._log("── Augmentation batch finished ──")

    # ─────────────────────────────────────────────────────────────────
    #  GENERATE DATASET
    # ─────────────────────────────────────────────────────────────────

    @Slot()
    def _on_generate(self) -> None:
        n = self.file_list.count()
        if n == 0:
            QMessageBox.warning(self, "No files",
                                "Add at least one CAD file first.")
            return

        model_paths = [self.file_list.item(i).text() for i in range(n)]
        rc = self._collect_render_cfg()
        all_presets = self.chk_all_noise_presets.isChecked()
        apply_noise = self.chk_noise.isChecked() or all_presets

        # Build list of noise configs
        noise_cfgs: List[Dict[str, Any]] = []
        if apply_noise:
            if all_presets:
                # Use ALL presets (overrides single-preset checkbox)
                for name, p in NI.NOISE_PRESETS.items():
                    cfg = dict(p)          # copy
                    cfg["preset_label"] = name
                    noise_cfgs.append(cfg)
                self._log(
                    f"[NOISE] Multi-preset mode: {len(noise_cfgs)} presets")
            else:
                # Use only the current GUI values
                nc = self._collect_noise_cfg()
                nc["preset_label"] = self.combo_noise_preset.currentText()
                noise_cfgs.append(nc)

        self.prog_tools.setMaximum(n)
        self.prog_tools.setValue(0)
        self.prog_frames.setMaximum(int(rc["n_frames"]))
        self.prog_frames.setValue(0)

        self._set_busy(True)
        self._log("=" * 50)
        self._log("  Starting dataset generation …")
        self._log("=" * 50)

        self._worker = DatasetWorker(
            model_paths=model_paths,
            clean_output_dir=self.render_out_edit.text(),
            noisy_output_dir=self.noisy_out_edit.text(),
            render_cfg=rc,
            noise_cfgs=noise_cfgs,
            apply_noise=apply_noise,
        )
        self._worker.sig.tool_started.connect(self._on_tool_started)
        self._worker.sig.frame_done.connect(self._on_frame_done)
        self._worker.sig.tool_done.connect(self._on_tool_done)
        self._worker.sig.log.connect(self._log)
        self._worker.sig.error.connect(self._on_error)
        self._worker.sig.finished.connect(self._on_finished)
        self._worker.start()

    @Slot()
    def _on_abort(self) -> None:
        if self._worker is not None:
            self._worker.abort()
            self._log("[USER] Abort requested …")

    @Slot(int, str)
    def _on_tool_started(self, idx: int, name: str) -> None:
        self.prog_frames.setValue(0)
        self.prog_frames.setMaximum(self.spin_frames.value())

    @Slot(int, int)
    def _on_frame_done(self, done: int, total: int) -> None:
        self.prog_frames.setValue(done)

    @Slot(int, int)
    def _on_tool_done(self, done: int, total: int) -> None:
        self.prog_tools.setValue(done)

    @Slot(str)
    def _on_error(self, tb: str) -> None:
        self._log(f"[ERROR]\n{tb}")
        QMessageBox.critical(self, "Error",
                             f"An error occurred:\n\n{tb[:600]}")

    @Slot()
    def _on_finished(self) -> None:
        self._set_busy(False)
        self._log("── Generation finished ──")


# ═════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    app = QApplication(sys.argv)
    _apply_dark_theme(app)
    app.setApplicationName("Sim2Real Mask Generator")

    win = SimulationGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
