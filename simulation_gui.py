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
    • "Generate Dataset" button with two progress bars:
          – Overall Progress (tools)
          – Current Tool Progress (360 frames)
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
    Qt, Signal, Slot, QThread, QObject,
)
from PySide6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QFileDialog, QProgressBar,
    QListWidget, QAbstractItemView, QGroupBox, QTextEdit,
    QSplitter, QMessageBox, QSizePolicy,
)

# ── Local modules (same folder) ─────────────────────────────────────
import render_engine as RE
import noise_injector as NI

_SCRIPT_DIR = Path(__file__).resolve().parent


# ═════════════════════════════════════════════════════════════════════
#  WORKER — runs on QThread, emits progress signals
# ═════════════════════════════════════════════════════════════════════

class _WorkerSignals(QObject):
    """Signals emitted by the dataset generation worker."""
    tool_started   = Signal(int, str)          # tool_index, tool_name
    frame_done     = Signal(int, int)          # frame_index, n_frames
    tool_done      = Signal(int, int)          # tool_index, n_tools
    log            = Signal(str)               # free-text log line
    preview_ready  = Signal(object)            # numpy RGB array
    finished       = Signal()
    error          = Signal(str)


class DatasetWorker(QThread):
    """
    Heavy-lifting thread: for each CAD file, load → preprocess → render
    360 frames.  Optionally apply noise per frame and save both clean
    and noisy masks.

    All VTK / PyVista calls happen inside *this* thread (OpenGL context
    is created here and stays here).
    """

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
        self.model_paths     = model_paths
        self.clean_output_dir = clean_output_dir
        self.noisy_output_dir = noisy_output_dir
        self.render_cfg      = render_cfg
        self.noise_cfg       = noise_cfg
        self.apply_noise     = apply_noise
        self._abort          = False

    # ── public abort handle ──────────────────────────────────────────
    def abort(self) -> None:
        self._abort = True

    # ── thread entry point ───────────────────────────────────────────
    def run(self) -> None:  # noqa: C901  (complex but linear)
        import pyvista as pv
        import vtk
        from PIL import Image

        try:
            n_tools = len(self.model_paths)
            rc = self.render_cfg
            nc = self.noise_cfg

            img_w = int(rc["img_w"])
            img_h = int(rc["img_h"])
            n_frames = int(rc["n_frames"])
            tip_from_top = float(rc["tip_from_top"])
            working_dist = float(rc["working_distance"])
            img_format = rc["image_format"].strip().upper()
            ext = "png" if img_format == "PNG" else "tiff"

            for ti, model_path in enumerate(self.model_paths):
                if self._abort:
                    self.sig.log.emit("[ABORT] Cancelled by user.")
                    break

                tool_name = RE.get_tool_name(model_path)
                self.sig.tool_started.emit(ti, tool_name)
                self.sig.log.emit(f"── Tool {ti + 1}/{n_tools}: {tool_name} ──")

                # ① Load
                self.sig.log.emit(f"[LOAD] {Path(model_path).name}")
                mesh = RE.load_mesh(model_path)

                # ② Preprocess
                mesh, tip_z, top_z, max_r = RE.preprocess_mesh(mesh)

                # ③ Camera — override globals temporarily
                orig_wd = RE.WORKING_DISTANCE_MM
                orig_tip = RE.TIP_FROM_TOP
                RE.WORKING_DISTANCE_MM = working_dist
                RE.TIP_FROM_TOP = tip_from_top
                cam = RE.compute_camera(tip_z, top_z, max_r)
                RE.WORKING_DISTANCE_MM = orig_wd
                RE.TIP_FROM_TOP = orig_tip

                # ④ Off-screen plotter (created per tool for a fresh GL ctx)
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

                # ⑤ Frame loop
                for angle in range(n_frames):
                    if self._abort:
                        break

                    xform = vtk.vtkTransform()
                    xform.RotateZ(float(angle))
                    actor.SetUserTransform(xform)

                    plotter.render()
                    rgb = plotter.screenshot(return_img=True)
                    mask = RE._to_binary_mask(rgb)

                    # Save clean mask
                    clean_path = str(clean_dir / f"mask_{angle:03d}.{ext}")
                    Image.fromarray(mask, mode="L").save(clean_path,
                                                         format=img_format)

                    # Optionally apply noise & save
                    if self.apply_noise and noisy_dir is not None:
                        noisy = NI.inject_noise(
                            mask,
                            edge_width=int(nc["edge_width"]),
                            blur_sigma=float(nc["blur_sigma"]),
                            flip_probability=float(nc["flip_prob"]),
                            flip_enabled=bool(nc["flip_enabled"]),
                            seed=nc.get("seed"),
                        )
                        noisy_path = str(noisy_dir / f"mask_{angle:03d}.{ext}")
                        Image.fromarray(noisy, mode="L").save(noisy_path,
                                                               format=img_format)

                    self.sig.frame_done.emit(angle + 1, n_frames)

                plotter.close()
                self.sig.tool_done.emit(ti + 1, n_tools)

            self.sig.log.emit("[DONE] Dataset generation complete.")

        except Exception:
            self.sig.error.emit(traceback.format_exc())
        finally:
            self.sig.finished.emit()


class PreviewWorker(QThread):
    """Render a single debug frame and send it back as a numpy RGB array."""

    sig = _WorkerSignals()

    def __init__(self, model_path: str, render_cfg: Dict[str, Any]):
        super().__init__()
        self.model_path = model_path
        self.render_cfg = render_cfg

    def run(self) -> None:
        import pyvista as pv
        import vtk

        try:
            rc = self.render_cfg
            img_w = int(rc["img_w"])
            img_h = int(rc["img_h"])
            tip_from_top = float(rc["tip_from_top"])
            working_dist = float(rc["working_distance"])

            mesh = RE.load_mesh(self.model_path)
            mesh, tip_z, top_z, max_r = RE.preprocess_mesh(mesh)

            orig_wd = RE.WORKING_DISTANCE_MM
            orig_tip = RE.TIP_FROM_TOP
            RE.WORKING_DISTANCE_MM = working_dist
            RE.TIP_FROM_TOP = tip_from_top
            cam = RE.compute_camera(tip_z, top_z, max_r)
            RE.WORKING_DISTANCE_MM = orig_wd
            RE.TIP_FROM_TOP = orig_tip

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


# ═════════════════════════════════════════════════════════════════════
#  DARK PALETTE
# ═════════════════════════════════════════════════════════════════════

def _apply_dark_theme(app: QApplication) -> None:
    """Fusion + custom dark palette."""
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
    """Top-level window: tabs, progress, log, generate button."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sim2Real Mask Generator")
        self.setMinimumSize(900, 740)
        self.resize(1040, 820)

        self._worker: Optional[DatasetWorker] = None
        self._preview_worker: Optional[PreviewWorker] = None

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
        root.addWidget(self.tabs, stretch=0)

        # ── Preview area ─────────────────────────────────────────────
        preview_group = QGroupBox("Debug Preview")
        preview_lay = QVBoxLayout(preview_group)
        self.preview_label = QLabel("No preview yet.  Select a CAD file and click Debug Preview.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(180)
        self.preview_label.setStyleSheet(
            "background-color: #111; border: 1px solid #333; border-radius: 4px;"
        )
        preview_lay.addWidget(self.preview_label)
        root.addWidget(preview_group, stretch=1)

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
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setStyleSheet(
            "background-color: #111; color: #9acd32; "
            "font-family: Consolas, monospace; font-size: 11px;"
        )
        root.addWidget(self.log)

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

    # ── Render tab ───────────────────────────────────────────────────

    def _build_render_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(8)

        # --- CAD file list ---
        file_group = QGroupBox("CAD Models (.step / .stp / .stl)")
        fg_lay = QVBoxLayout(file_group)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setMinimumHeight(70)
        self.file_list.setMaximumHeight(120)
        fg_lay.addWidget(self.file_list)
        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Files …")
        btn_add.clicked.connect(self._add_cad_files)
        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._remove_cad_files)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_remove)
        fg_lay.addLayout(btn_row)
        lay.addWidget(file_group)

        # --- Output directory ---
        out_group = QGroupBox("Output Directory (clean masks)")
        og_lay = QHBoxLayout(out_group)
        self.render_out_edit = QLineEdit(str(_SCRIPT_DIR / "output"))
        og_lay.addWidget(self.render_out_edit, stretch=1)
        btn_browse = QPushButton("Browse …")
        btn_browse.clicked.connect(lambda: self._browse_dir(self.render_out_edit))
        og_lay.addWidget(btn_browse)
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

        # --- Debug preview button ---
        self.btn_debug = QPushButton("  Debug Preview (selected file)  ")
        self.btn_debug.setMinimumHeight(36)
        self.btn_debug.setStyleSheet(
            "QPushButton { background-color: #1a4a6b; color: white; "
            "font-size: 13px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #1e6090; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.btn_debug.clicked.connect(self._on_debug_preview)
        lay.addWidget(self.btn_debug)

        lay.addStretch()
        return tab

    # ── Noise tab ────────────────────────────────────────────────────

    def _build_noise_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(8)

        # --- Enable / Disable noise ---
        self.chk_noise = QCheckBox("Apply noise to rendered masks")
        self.chk_noise.setChecked(True)
        self.chk_noise.setStyleSheet("font-size: 13px; font-weight: bold;")
        lay.addWidget(self.chk_noise)

        # --- Noisy output dir ---
        nout_group = QGroupBox("Output Directory (noisy masks)")
        no_lay = QHBoxLayout(nout_group)
        self.noisy_out_edit = QLineEdit(str(_SCRIPT_DIR / "output_noisy"))
        no_lay.addWidget(self.noisy_out_edit, stretch=1)
        btn_browse_n = QPushButton("Browse …")
        btn_browse_n.clicked.connect(
            lambda: self._browse_dir(self.noisy_out_edit))
        no_lay.addWidget(btn_browse_n)
        lay.addWidget(nout_group)

        # --- Noise parameters ---
        noise_group = QGroupBox("Noise Parameters")
        form = QFormLayout(noise_group)

        self.spin_edge_w = QSpinBox()
        self.spin_edge_w.setRange(1, 50)
        self.spin_edge_w.setValue(NI.EDGE_WIDTH)
        self.spin_edge_w.setSuffix("  px")
        self.spin_edge_w.setToolTip(
            "Half-width of the boundary band where noise is applied.\n"
            "Larger = wider noisy border."
        )
        form.addRow("Edge Width:", self.spin_edge_w)

        self.spin_blur = QDoubleSpinBox()
        self.spin_blur.setRange(0.0, 10.0)
        self.spin_blur.setValue(NI.BLUR_SIGMA)
        self.spin_blur.setSingleStep(0.1)
        self.spin_blur.setDecimals(2)
        self.spin_blur.setToolTip(
            "Gaussian sigma for edge-aliasing blur.\n"
            "0 = no blur,  0.5–1.0 = subtle,  1.5–2.5 = visible."
        )
        form.addRow("Blur Sigma (σ):", self.spin_blur)

        self.chk_flip = QCheckBox("Enable specular-flip noise")
        self.chk_flip.setChecked(NI.FLIP_ENABLED)
        form.addRow("Specular Flip:", self.chk_flip)

        self.spin_flip_prob = QDoubleSpinBox()
        self.spin_flip_prob.setRange(0.0, 1.0)
        self.spin_flip_prob.setValue(NI.FLIP_PROB)
        self.spin_flip_prob.setSingleStep(0.005)
        self.spin_flip_prob.setDecimals(3)
        self.spin_flip_prob.setToolTip(
            "Per-pixel probability of white↔black flip in boundary zone.\n"
            "0.01 = subtle,  0.05 = moderate,  0.10 = aggressive."
        )
        form.addRow("Flip Probability:", self.spin_flip_prob)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 2_147_483_647)
        self.spin_seed.setValue(-1)
        self.spin_seed.setToolTip(
            "RNG seed for reproducible noise.  -1 = non-deterministic."
        )
        form.addRow("Seed (-1 = random):", self.spin_seed)

        lay.addWidget(noise_group)
        lay.addStretch()
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
            edge_width=self.spin_edge_w.value(),
            blur_sigma=self.spin_blur.value(),
            flip_prob=self.spin_flip_prob.value(),
            flip_enabled=self.chk_flip.isChecked(),
            seed=None if seed_val < 0 else seed_val,
        )

    def _log(self, text: str) -> None:
        self.log.append(text)
        # Auto-scroll
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_busy(self, busy: bool) -> None:
        self.btn_generate.setEnabled(not busy)
        self.btn_abort.setEnabled(busy)
        self.btn_debug.setEnabled(not busy)
        self.tabs.setEnabled(not busy)

    # ─────────────────────────────────────────────────────────────────
    #  DEBUG PREVIEW
    # ─────────────────────────────────────────────────────────────────

    @Slot()
    def _on_debug_preview(self) -> None:
        sel = self.file_list.selectedItems()
        if not sel:
            if self.file_list.count() == 0:
                QMessageBox.warning(self, "No files",
                                    "Add at least one CAD file first.")
                return
            # Use first file if none selected
            model_path = self.file_list.item(0).text()
        else:
            model_path = sel[0].text()

        self.btn_debug.setEnabled(False)
        self.preview_label.setText("Rendering preview …")
        self._log(f"[DEBUG] Rendering preview for {Path(model_path).name} …")

        self._preview_worker = PreviewWorker(model_path,
                                             self._collect_render_cfg())
        self._preview_worker.sig.preview_ready.connect(self._show_preview)
        self._preview_worker.sig.log.connect(self._log)
        self._preview_worker.sig.error.connect(self._on_error)
        self._preview_worker.sig.finished.connect(
            lambda: self.btn_debug.setEnabled(True))
        self._preview_worker.start()

    @Slot(object)
    def _show_preview(self, rgb: np.ndarray) -> None:
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data.tobytes(), w, h, bytes_per_line,
                      QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # Scale to fit the label while keeping aspect ratio
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

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

        # Reset progress bars
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
        n_frames = self.spin_frames.value()
        self.prog_frames.setMaximum(n_frames)

    @Slot(int, int)
    def _on_frame_done(self, done: int, total: int) -> None:
        self.prog_frames.setValue(done)

    @Slot(int, int)
    def _on_tool_done(self, done: int, total: int) -> None:
        self.prog_tools.setValue(done)

    @Slot(str)
    def _on_error(self, tb: str) -> None:
        self._log(f"[ERROR]\n{tb}")
        QMessageBox.critical(self, "Error", f"An error occurred:\n\n{tb[:600]}")

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
