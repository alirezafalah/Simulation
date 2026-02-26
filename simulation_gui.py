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
from typing import List, Optional, Dict, Any

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
)

# ── Local modules (same folder) ─────────────────────────────────────
import render_engine as RE
import noise_injector as NI

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
        if self._popup is not None and self._popup.isVisible():
            self._popup._source_pixmap = pixmap
            self._popup._apply_zoom()

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
        noise_cfg: Dict[str, Any],
        apply_noise: bool,
    ):
        super().__init__()
        self.model_paths      = model_paths
        self.clean_output_dir = clean_output_dir
        self.noisy_output_dir = noisy_output_dir
        self.render_cfg       = render_cfg
        self.noise_cfg        = noise_cfg
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
            nc = self.noise_cfg

            img_w  = int(rc["img_w"])
            img_h  = int(rc["img_h"])
            n_frames     = int(rc["n_frames"])
            tip_from_top = float(rc["tip_from_top"])
            working_dist = float(rc["working_distance"])
            img_format   = rc["image_format"].strip().upper()
            ext = "png" if img_format == "PNG" else "tiff"

            for ti, model_path in enumerate(self.model_paths):
                if self._abort:
                    self.sig.log.emit("[ABORT] Cancelled by user.")
                    break

                tool_name = RE.get_tool_name(model_path)
                self.sig.tool_started.emit(ti, tool_name)
                self.sig.log.emit(f"── Tool {ti+1}/{n_tools}: {tool_name} ──")

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
                noisy_dir: Optional[Path] = None
                if self.apply_noise:
                    noisy_dir = Path(self.noisy_output_dir) / tool_name
                    noisy_dir.mkdir(parents=True, exist_ok=True)

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

                    if self.apply_noise and noisy_dir is not None:
                        noisy = NI.inject_noise(
                            mask,
                            n_dents=int(nc["n_dents"]),
                            dent_span_min=int(nc["dent_span_min"]),
                            dent_span_max=int(nc["dent_span_max"]),
                            max_depth=float(nc["max_depth"]),
                            seed=nc.get("seed"),
                        )
                        noisy_path = str(noisy_dir / f"mask_{angle:03d}.{ext}")
                        Image.fromarray(noisy, mode="L").save(
                            noisy_path, format=img_format)

                    self.sig.frame_done.emit(angle + 1, n_frames)

                plotter.close()
                self.sig.tool_done.emit(ti + 1, n_tools)

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


class NoiseBatchWorker(QThread):
    """Standalone thread: apply noise to every mask in a folder."""
    sig = _WorkerSignals()

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        noise_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.input_dir  = input_dir
        self.output_dir = output_dir
        self.noise_cfg  = noise_cfg
        self._abort     = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        from PIL import Image

        try:
            nc  = self.noise_cfg
            inp = Path(self.input_dir)
            out = Path(self.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            files = sorted(
                f for f in inp.iterdir()
                if f.suffix.lower() in (".png", ".tiff", ".tif")
            )
            if not files:
                self.sig.log.emit(f"[NOISE] No image files in {inp}")
                return

            total = len(files)
            self.sig.log.emit(
                f"[NOISE] Processing {total} masks: {inp} → {out}")

            for i, src in enumerate(files):
                if self._abort:
                    self.sig.log.emit("[ABORT] Noise batch cancelled.")
                    break

                img = np.array(
                    Image.open(src).convert("L"), dtype=np.uint8)
                img = np.where(img > 127, 255, 0).astype(np.uint8)

                noisy = NI.inject_noise(
                    img,
                    n_dents=int(nc["n_dents"]),
                    dent_span_min=int(nc["dent_span_min"]),
                    dent_span_max=int(nc["dent_span_max"]),
                    max_depth=float(nc["max_depth"]),
                    seed=nc.get("seed"),
                )
                dst = out / src.name
                Image.fromarray(noisy, mode="L").save(str(dst), format="PNG")
                self.sig.frame_done.emit(i + 1, total)

            self.sig.log.emit("[DONE] Noise batch complete.")
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
        self.combo_fmt.addItems(["PNG", "TIFF"])
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
            "Apply noise during dataset generation (Render tab)")
        self.chk_noise.setChecked(True)
        self.chk_noise.setStyleSheet("font-size: 13px; font-weight: bold;")
        lay.addWidget(self.chk_noise)

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

        # --- Noise parameters ---
        noise_group = QGroupBox("Dent Parameters (localised contour erosion)")
        form = QFormLayout(noise_group)

        self.spin_n_dents = QSpinBox()
        self.spin_n_dents.setRange(0, 30)
        self.spin_n_dents.setValue(NI.N_DENTS)
        self.spin_n_dents.setToolTip(
            "Number of independent erosion patches along the contour.\n"
            "0 = no dents (clean mask)."
        )
        form.addRow("Number of Dents:", self.spin_n_dents)

        self.spin_span_min = QSpinBox()
        self.spin_span_min.setRange(10, 1000)
        self.spin_span_min.setValue(NI.DENT_SPAN_MIN)
        self.spin_span_min.setSuffix("  contour px")
        self.spin_span_min.setToolTip(
            "Minimum arc length (in contour pixels) of each dent."
        )
        form.addRow("Span Min:", self.spin_span_min)

        self.spin_span_max = QSpinBox()
        self.spin_span_max.setRange(10, 2000)
        self.spin_span_max.setValue(NI.DENT_SPAN_MAX)
        self.spin_span_max.setSuffix("  contour px")
        self.spin_span_max.setToolTip(
            "Maximum arc length (in contour pixels) of each dent."
        )
        form.addRow("Span Max:", self.spin_span_max)

        self.spin_max_depth = QDoubleSpinBox()
        self.spin_max_depth.setRange(0.5, 10.0)
        self.spin_max_depth.setValue(NI.MAX_DEPTH)
        self.spin_max_depth.setSingleStep(0.5)
        self.spin_max_depth.setDecimals(1)
        self.spin_max_depth.setSuffix("  px")
        self.spin_max_depth.setToolTip(
            "Maximum inward erosion depth per dent (pixels).\n"
            "Each dent picks a random depth in [1.0, max_depth].\n"
            "1–2 = subtle,  2–3 = visible,  3+ = aggressive."
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

        # Connect every noise param widget to live preview update
        self.spin_n_dents.valueChanged.connect(self._update_noise_preview)
        self.spin_span_min.valueChanged.connect(self._update_noise_preview)
        self.spin_span_max.valueChanged.connect(self._update_noise_preview)
        self.spin_max_depth.valueChanged.connect(self._update_noise_preview)
        self.spin_seed.valueChanged.connect(self._update_noise_preview)

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

        dir_row.addWidget(QLabel("Output folder:"), 1, 0)
        self.noise_batch_out = QLineEdit(str(_SCRIPT_DIR / "output_noisy"))
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
        self.btn_noise_batch = QPushButton("  Process Folder  ")
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
            n_dents=self.spin_n_dents.value(),
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

        # Apply current noise and show result
        self._update_noise_preview()

    @Slot()
    def _update_noise_preview(self) -> None:
        """Re-apply noise with current GUI params to the cached sample mask."""
        if self._sample_mask is None:
            return

        nc = self._collect_noise_cfg()
        noisy = NI.inject_noise(
            self._sample_mask,
            n_dents=int(nc["n_dents"]),
            dent_span_min=int(nc["dent_span_min"]),
            dent_span_max=int(nc["dent_span_max"]),
            max_depth=float(nc["max_depth"]),
            seed=nc.get("seed"),
        )

        pm = _gray_to_pixmap(noisy)
        self.noise_result_preview.set_preview(pm, "Noisy result")

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
                "Select a valid input folder containing mask images.")
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
        nc = self._collect_noise_cfg()
        apply_noise = self.chk_noise.isChecked()

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
            noise_cfg=nc,
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
