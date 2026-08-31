"""
AFM Cell Analyzer v6

Force-curve analysis for single-cell compression using the Lulevich two-term
model. Settings live in one place, the fit is live, and every result comes
with the diagnostics needed to tell a real measurement from a bad window.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from lulevich_model import LulevichModel, compare_couplings
from plot_utils import (
    FORCE_UNITS,
    INPUT_FORCE_UNITS,
    PlotStyle,
    autoscale_unit,
    cell_schematic,
    force_curve_figure,
    from_newtons,
    residual_figure,
    sensitivity_figure,
    to_newtons,
)

# Optional dependencies: the app must still run without Google credentials
# or the Igor toolchain installed.
try:
    from google_sheets_manager import initialize_sheets_manager

    SHEETS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local install
    initialize_sheets_manager = None
    SHEETS_IMPORT_ERROR = str(exc)

try:
    import video_analysis as va

    VIDEO_IMPORT_ERROR = None if va.available() else "OpenCV is not installed"
except Exception as exc:  # pragma: no cover
    va = None
    VIDEO_IMPORT_ERROR = str(exc)

try:
    from igor_parser import IgorParser
    from baseline_correction import BaselineCorrector

    IGOR_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    IgorParser = None
    BaselineCorrector = None
    IGOR_IMPORT_ERROR = str(exc)


# ============================================================== page setup ==

st.set_page_config(
    page_title="AFM Cell Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {padding-top: 2.2rem; max-width: 1500px;}
    .app-title {font-size: 1.9rem; font-weight: 700; color: #14425f; margin-bottom: 0;}
    .app-sub {color: #6b7785; font-size: 0.9rem; margin-top: 0.1rem;}
    .section {font-size: 1.05rem; font-weight: 650; color: #14425f;
              border-bottom: 2px solid #e3e9ee; padding-bottom: 0.3rem;
              margin: 1.4rem 0 0.9rem 0;}
    .hint {color: #6b7785; font-size: 0.83rem;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem;}
    section[data-testid="stSidebar"] div[data-testid="stExpander"] {border: none;}
</style>
""",
    unsafe_allow_html=True,
)


def _stretch_kwargs():
    """`use_container_width` is deprecated in new Streamlit and absent in old.

    Pick whichever the installed version understands so the app runs on both.
    """
    try:
        major, minor = (int(part) for part in st.__version__.split(".")[:2])
    except Exception:
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 49) else {"use_container_width": True}


STRETCH = _stretch_kwargs()


def _supports_selection():
    """Chart selection events (drag to set a window) arrived in Streamlit 1.35."""
    try:
        major, minor = (int(part) for part in st.__version__.split(".")[:2])
    except Exception:
        return False
    return (major, minor) >= (1, 35)


SUPPORTS_SELECTION = _supports_selection()


def section(text: str):
    st.markdown(f'<div class="section">{text}</div>', unsafe_allow_html=True)


def hint(text: str):
    st.markdown(f'<p class="hint">{text}</p>', unsafe_allow_html=True)


# ========================================================== session state ==

DEFAULTS = {
    # display
    "force_unit": "N",
    "data_color": "#1f77b4",
    "fit_color": "#d62728",
    "marker_size": 6,
    "line_width": 3,
    "plot_height": 520,
    "show_grid": False,
    "axis_title_size": 28,
    "tick_size": 22,
    "axis_width": 4,
    "bold_axes": True,
    "log_scale": False,
    "show_components": True,
    "show_fit_window": True,
    "show_schematic": True,
    "plot_width": 2.4,
    # geometry / model
    "radius_mode": "From height",
    "radius_aspect": 0.55,
    "cell_radius_um": 4.45,
    "membrane_thickness_nm": 4.0,
    "poisson_membrane": 0.50,
    "poisson_interior": 0.50,
    # cell type
    "cell_type": "Myoblast (C2C12)",
    "nucleus_fraction": 0.35,
    "nucleus_radius_um": 1.55,
    "nucleus_radius_mode": "From cell radius",
    "poisson_nucleus": 0.50,
    "nucleus_onset": 0.20,
    "onset_mode": "Scan for best",
    "_scanned_onset": None,
    # fitting
    "coupling": "Parallel (forces add)",
    "procedure": "All at once",
    "crossover_mode": "Scan for best",
    "crossover": 0.18,
    "use_membrane": True,
    "use_interior": True,
    "use_nucleus": False,
    "stage_of_membrane": 2,
    "stage_of_interior": 1,
    "stage_of_nucleus": 3,
    "refine_iterations": 3,
    "seed_parallel": True,
    "range_presets": {},
    "drag_target": "(off)",
    "weighting": "uniform",
    "fit_offset": False,
    "live_fit": True,
    # cell metadata
    "cell_name": "",
    "cell_height_um": 8.09,
    "spring_constant": 0.0,
    "video_link": "",
    # video
    "video_path": None,
    "video_info": None,
    "video_name": None,
    "video_track": None,
    "video_contact_frame": 0,
    "video_end_frame": 0,
    "video_roi": (0.0, 0.0, 1.0, 1.0),
    "video_roi_x": (0.0, 1.0),
    "video_roi_y": (0.0, 1.0),
    "video_sensitivity": 1.0,
    "video_strip_lines": True,
    "video_show_panel": True,
    # data / results
    "data": None,
    "results": None,
    "gs_manager": None,
    "db_enabled": False,
}

# Streamlit refuses a write to a widget's key once that widget has been built
# this run. Buttons further down the page therefore stage their changes here
# and rerun, and this block applies them before anything is drawn.
if st.session_state.pop("_pending_clear_windows", False):
    for _stale in [k for k in st.session_state if k.startswith("window_")]:
        del st.session_state[_stale]

for _key, _value in (st.session_state.pop("_pending_settings", None) or {}).items():
    st.session_state[_key] = _value

if st.session_state.pop("_reset_requested", False):
    # Widget-backed keys can only be reassigned before their widget is built,
    # so the reset button sets a flag and the actual reset happens here.
    for key, value in DEFAULTS.items():
        if key not in ("data", "results", "gs_manager"):
            st.session_state[key] = value
    for stale in [k for k in st.session_state.keys()
                  if k.startswith(("window_", "celltype_window_", "stage_of_"))]:
        del st.session_state[stale]
    st.session_state.pop("_applied_cell_type", None)

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def current_style(force_N=None) -> PlotStyle:
    """Build the PlotStyle from the sidebar settings, honouring auto units."""
    unit = st.session_state["force_unit"]
    if unit == "auto":
        unit = autoscale_unit(force_N) if force_N is not None and np.size(force_N) else "nN"
    return PlotStyle(
        force_unit=unit,
        data_color=st.session_state["data_color"],
        fit_color=st.session_state["fit_color"],
        marker_size=st.session_state["marker_size"],
        line_width=st.session_state["line_width"],
        height=st.session_state["plot_height"],
        show_grid=st.session_state["show_grid"],
        axis_title_size=st.session_state["axis_title_size"],
        tick_size=st.session_state["tick_size"],
        axis_width=st.session_state["axis_width"],
        bold_axes=st.session_state["bold_axes"],
        log_scale=st.session_state["log_scale"],
        show_components=st.session_state["show_components"],
        show_fit_window=st.session_state["show_fit_window"],
    )


@st.cache_data(show_spinner=False)
def load_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV/Excel once and keep it across reruns."""
    buffer = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    for sep in (None, ",", ";", "\t"):
        buffer.seek(0)
        try:
            df = pd.read_csv(buffer, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    buffer.seek(0)
    return pd.read_csv(buffer)


@st.cache_data(show_spinner=False)
def cached_frame(path, signature, index):
    """`signature` (size + mtime) busts the cache when the file is replaced."""
    return va.read_frame(path, index)


@st.cache_data(show_spinner=False)
def cached_detection(path, signature, index, roi, sensitivity, strip_lines):
    frame = va.read_frame(path, index)
    if frame is None:
        return None, None
    det = va.detect_cell(
        frame, roi=roi, sensitivity=sensitivity, strip_lines=strip_lines
    )
    return frame, det


@st.cache_data(show_spinner="Tracking the cell through the video…")
def cached_track(path, signature, n_samples, roi, sensitivity, strip_lines, start, end):
    track = va.track_cell(
        path,
        n_samples=n_samples,
        roi=roi,
        sensitivity=sensitivity,
        start=start,
        end=end,
    )
    # Detections hold OpenCV contours; drop them so the cached value stays small.
    return {k: v for k, v in track.items() if k != "detections"}


def video_signature():
    path = st.session_state.get("video_path")
    if not path or not os.path.exists(path):
        return None
    stat = os.stat(path)
    return (stat.st_size, int(stat.st_mtime))


def png_bytes(image_rgb):
    """Encode an RGB array as PNG for download."""
    import cv2

    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return buffer.tobytes() if ok else b""


def guess_column(columns, keywords, fallback_index):
    for i, col in enumerate(columns):
        low = str(col).lower()
        if any(k in low for k in keywords):
            return i
    return min(fallback_index, len(columns) - 1)


def build_model(epsilon, force_N) -> LulevichModel:
    """Construct the model from the current geometry settings. Metres, always."""
    height_m = float(st.session_state["cell_height_um"]) * 1e-6
    if st.session_state["radius_mode"] == "Manual":
        radius_m = float(st.session_state["cell_radius_um"]) * 1e-6
    else:
        radius_m = height_m * float(st.session_state["radius_aspect"])
    if st.session_state["nucleus_radius_mode"] == "Manual":
        nucleus_m = float(st.session_state["nucleus_radius_um"]) * 1e-6
    else:
        nucleus_m = radius_m * float(st.session_state["nucleus_fraction"])
    return LulevichModel(
        force_N,
        epsilon,
        cell_height=height_m,
        cell_radius=radius_m,
        membrane_thickness=float(st.session_state["membrane_thickness_nm"]) * 1e-9,
        poisson_membrane=float(st.session_state["poisson_membrane"]),
        poisson_interior=float(st.session_state["poisson_interior"]),
        nucleus_radius=nucleus_m,
        poisson_nucleus=float(st.session_state["poisson_nucleus"]),
        nucleus_onset=float(st.session_state["nucleus_onset"]),
        expected_ranges=CELL_TYPES.get(st.session_state["cell_type"], {}).get("expected"),
    )


# ============================================================ cell presets ==

TERM_LABELS = {
    "membrane": "membrane (Eₘ)",
    "interior": "cytoskeleton (Ec)",
    "nucleus": "nucleus (Eₙ)",
}
TERM_ORDER = ("membrane", "interior", "nucleus")

# How the elements share the load. These are physics, not fitting procedure:
# parallel and series here describe the spring network, while "All at once"
# and "Stage by stage" describe how the fit is carried out.
COUPLINGS = {
    "Parallel (forces add)":
        "Every element is squashed by the same amount; their forces add. The "
        "stiffest element dominates.",
    "Series (deformations add)":
        "Every element carries the same force; their deformations add, a spring "
        "stacked on a spring. The softest element dominates.",
    "Hybrid: parallel below, series above":
        "Parallel up to a crossover deformation, then the load path stacks.",
    "Hybrid: series below, parallel above":
        "Stacked at small deformation, sharing the squash once compressed.",
    "Auto (let the data choose)":
        "Fits all four and ranks them by AICc and cross-validation.",
}
COUPLING_KEYS = {
    "Parallel (forces add)": "parallel",
    "Series (deformations add)": "series",
    "Hybrid: parallel below, series above": "hybrid_ps",
    "Hybrid: series below, parallel above": "hybrid_sp",
    "Auto (let the data choose)": "auto",
}
STAGE_COLORS = ("#2ca02c", "#9467bd", "#e377c2", "#ff7f0e")

# Starting points, not literature constants. They set the geometry, the
# plausibility bands used for warnings, and the initial fit windows for a
# cell type. Edit them for your own line and save the result as a preset.
CELL_TYPES = {
    "Myoblast (C2C12)": {
        "cell_height_um": 8.0,
        "radius_aspect": 0.55,
        "nucleus_fraction": 0.35,
        "membrane_thickness_nm": 4.0,
        "nucleus_onset": 0.20,
        # expected bands in pascals
        "expected": {"Em": (2e5, 2e7), "Ei": (2e2, 1e4), "En": (1e3, 5e4)},
        "windows": {"membrane": (0.18, 0.32), "interior": (0.01, 0.12),
                    "nucleus": (0.24, 0.40)},
    },
    "Cardiomyocyte": {
        "cell_height_um": 12.0,
        "radius_aspect": 0.60,
        "nucleus_fraction": 0.32,
        "membrane_thickness_nm": 4.5,
        "nucleus_onset": 0.18,
        "expected": {"Em": (5e5, 5e7), "Ei": (1e3, 1e5), "En": (2e3, 2e5)},
        "windows": {"membrane": (0.15, 0.30), "interior": (0.01, 0.10),
                    "nucleus": (0.22, 0.38)},
    },
    "Custom": {
        "cell_height_um": 8.09,
        "radius_aspect": 0.55,
        "nucleus_fraction": 0.35,
        "membrane_thickness_nm": 4.0,
        "nucleus_onset": 0.15,
        "expected": {"Em": (1e3, 1e9), "Ei": (1e0, 1e7), "En": (1e1, 1e7)},
        "windows": {"membrane": (0.15, 0.30), "interior": (0.01, 0.12),
                    "nucleus": (0.22, 0.40)},
    },
}


def apply_cell_type(name):
    """Copy a cell type's defaults into the settings."""
    preset = CELL_TYPES.get(name)
    if not preset:
        return
    for key in (
        "cell_height_um",
        "radius_aspect",
        "nucleus_fraction",
        "membrane_thickness_nm",
        "nucleus_onset",
    ):
        st.session_state[key] = preset[key]
    for term, window in preset["windows"].items():
        st.session_state[f"celltype_window_{term}"] = tuple(window)
    for key in list(st.session_state.keys()):
        if key.startswith("window_"):
            del st.session_state[key]


def active_terms():
    """Terms currently switched on, always in membrane -> nucleus order."""
    return tuple(t for t in TERM_ORDER if st.session_state.get(f"use_{t}", False))


def stage_groups(terms):
    """Group the active terms by the stage number assigned to each."""
    grouped = {}
    for term in terms:
        stage = int(st.session_state.get(f"stage_of_{term}", 1))
        grouped.setdefault(stage, []).append(term)
    return [(stage, tuple(grouped[stage])) for stage in sorted(grouped)]


def default_window_for(terms, auto_window, lo, hi):
    """Starting window for a stage, from the cell type where one is defined."""
    spans = []
    for term in terms:
        window = st.session_state.get(f"celltype_window_{term}")
        if window:
            spans.append(tuple(window))
    if spans:
        window = (min(s[0] for s in spans), max(s[1] for s in spans))
    else:
        window = auto_window
    lo_w = float(np.clip(window[0], lo, hi))
    hi_w = float(np.clip(window[1], lo, hi))
    return (lo_w, hi_w) if lo_w < hi_w else auto_window


def apply_preset(preset, lo, hi):
    """
    Stage a saved set of windows and settings.

    Everything is staged rather than written directly: this runs from a button
    below the widgets it changes, and Streamlit rejects a write to a widget key
    after that widget exists. The caller reruns and the staged values are
    applied at the top of the next run.
    """
    pending = {}
    if preset.get("coupling") in COUPLINGS:
        pending["coupling"] = preset["coupling"]
    if preset.get("procedure") in ("All at once", "Stage by stage"):
        pending["procedure"] = preset["procedure"]
    for term in TERM_ORDER:
        pending[f"use_{term}"] = term in preset.get("terms", [])
    for term, stage in (preset.get("stage_of") or {}).items():
        pending[f"stage_of_{term}"] = int(stage)
    if preset.get("nucleus_onset") is not None:
        pending["nucleus_onset"] = float(preset["nucleus_onset"])
    if preset.get("crossover") is not None:
        pending["crossover"] = float(preset["crossover"])

    def clamped(window):
        low = float(np.clip(window[0], lo, hi))
        high = float(np.clip(window[1], lo, hi))
        return (low, high) if low < high else None

    for term, window in (preset.get("term_windows") or {}).items():
        value = clamped(window)
        if value:
            pending[f"window_term_{term}"] = value
    if preset.get("combined_window"):
        value = clamped(preset["combined_window"])
        if value:
            pending["window_combined"] = value

    st.session_state["_pending_settings"] = pending


def plot_selection_kwargs():
    """
    Enable box selection on the chart where Streamlit supports it.

    Selection events arrived in 1.35; on anything older the chart is rendered
    normally and the sliders remain the only way to set a window.
    """
    if not SUPPORTS_SELECTION or st.session_state.get("drag_target", "(off)") == "(off)":
        return {}
    return {"on_select": "rerun", "selection_mode": "box"}


def apply_plot_drag(chart_key, lo, hi):
    """Turn a box selection on the chart into the chosen window."""
    target = st.session_state.get("drag_target", "(off)")
    if not SUPPORTS_SELECTION or target == "(off)":
        return
    event = st.session_state.get(chart_key)
    boxes = ((event or {}).get("selection") or {}).get("box") or []
    if not boxes:
        return
    xs = boxes[0].get("x") or []
    if len(xs) < 2:
        return
    window = (
        float(np.clip(min(xs), lo, hi)),
        float(np.clip(max(xs), lo, hi)),
    )
    if window[1] - window[0] < 1e-6:
        return

    key = (st.session_state.get("_drag_keys") or {}).get(target)
    if not key:
        return
    if st.session_state.get(key) != window:
        st.session_state["_pending_settings"] = {key: window}
        st.rerun()



# ================================================================ sidebar ==

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    hint("These apply everywhere in the app.")

    previous_type = st.session_state.get("_applied_cell_type")
    st.selectbox(
        "Cell type",
        list(CELL_TYPES.keys()),
        key="cell_type",
        help="Sets geometry, bilayer thickness, the plausibility bands used for "
        "warnings, and the starting fit windows. Everything stays editable.",
    )
    if previous_type != st.session_state["cell_type"]:
        apply_cell_type(st.session_state["cell_type"])
        st.session_state["_applied_cell_type"] = st.session_state["cell_type"]

    with st.expander("📐 Cell geometry", expanded=True):
        st.number_input(
            "Cell height h₀ (μm)",
            min_value=0.1,
            max_value=100.0,
            step=0.01,
            format="%.2f",
            key="cell_height_um",
            help="Initial (undeformed) height. Sets both the geometry prefactors "
            "and the conversion from relative deformation to indentation.",
        )
        st.radio(
            "Cell radius R₀",
            ["From height", "Manual"],
            key="radius_mode",
            horizontal=True,
            help="R₀ enters the membrane term linearly and the Hertzian term as √R₀.",
        )
        if st.session_state["radius_mode"] == "From height":
            st.slider(
                "R₀ / h₀ aspect factor",
                0.30,
                1.50,
                step=0.01,
                key="radius_aspect",
            )
            st.caption(
                f"R₀ = {st.session_state['cell_height_um'] * st.session_state['radius_aspect']:.2f} μm"
            )
        else:
            st.number_input(
                "Cell radius R₀ (μm)",
                min_value=0.1,
                max_value=100.0,
                step=0.01,
                format="%.2f",
                key="cell_radius_um",
            )

    with st.expander("🧬 Model constants"):
        st.number_input(
            "Lipid bilayer thickness hₘ (nm)",
            min_value=0.5,
            max_value=100.0,
            step=0.1,
            key="membrane_thickness_nm",
            help="A lipid bilayer is about 4 to 5 nm. The ε³ term measures the "
            "product Eₘ·hₘ, so Eₘ scales inversely with whatever you assume here: "
            "assume 8 nm instead of 4 and Eₘ halves, with the data unchanged. The "
            "app reports Eₘ·hₘ alongside Eₘ for that reason.",
        )
        st.slider("Poisson ratio, membrane νₘ", 0.0, 0.5, step=0.01, key="poisson_membrane")
        st.slider("Poisson ratio, cytoskeleton νc", 0.0, 0.5, step=0.01, key="poisson_interior")
        st.caption("0.5 = incompressible, the usual choice for living cells.")

    with st.expander("🟣 Nucleus"):
        st.radio(
            "Nucleus radius",
            ["From cell radius", "Manual"],
            key="nucleus_radius_mode",
            horizontal=True,
        )
        if st.session_state["nucleus_radius_mode"] == "From cell radius":
            st.slider(
                "R_nucleus / R₀", 0.10, 0.90, step=0.01, key="nucleus_fraction"
            )
        else:
            st.number_input(
                "Nucleus radius (μm)", min_value=0.1, max_value=50.0, step=0.05,
                format="%.2f", key="nucleus_radius_um",
            )
        st.slider("Poisson ratio, nucleus νₙ", 0.0, 0.5, step=0.01, key="poisson_nucleus")
        st.radio(
            "Onset deformation ε₀",
            ["Scan for best", "Set manually"],
            key="onset_mode",
            help="The deformation at which the plates start to feel the nucleus. "
            "Below it the nucleus term is exactly zero, which is what keeps it "
            "distinguishable from the cytoskeleton term.",
        )
        st.slider("ε₀", 0.0, 0.6, step=0.01, key="nucleus_onset")
        if st.session_state["onset_mode"] == "Scan for best":
            found = st.session_state.get("_scanned_onset")
            st.caption(
                f"The slider is ignored while scanning. Last scan found "
                f"ε₀ = {found:.3f}." if found is not None
                else "The slider is ignored while scanning; the fit finds ε₀ itself."
            )

    with st.expander("📈 Fitting", expanded=True):
        hint(
            "Which terms to fit, parallel or series, and the ε windows are all set "
            "in the main panel under **Model and fit windows**."
        )
        st.selectbox(
            "Weighting",
            ["uniform", "relative"],
            key="weighting",
            help="uniform = minimise absolute residuals (the high-force end dominates). "
            "relative = weight each point by 1/|F| so the small-ε region counts too. "
            "Switch to relative if the fitted line ignores the low-deformation points.",
        )
        st.checkbox(
            "Fit a constant force offset",
            key="fit_offset",
            help="Absorbs a residual baseline or a slightly wrong contact point. "
            "Turn on if the fit misses at small ε.",
        )
        st.checkbox("Refit live as settings change", key="live_fit")

    with st.expander("🎨 Display"):
        st.selectbox(
            "Force unit",
            ["auto"] + list(FORCE_UNITS.keys()),
            key="force_unit",
            help="'auto' picks the unit that keeps the peak force between 1 and 1000.",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.color_picker("Data", key="data_color")
        with c2:
            st.color_picker("Fit", key="fit_color")
        st.slider("Marker size", 2, 14, key="marker_size")
        st.slider("Line width", 1, 8, key="line_width")
        st.slider("Plot height (px)", 320, 900, step=20, key="plot_height")
        st.slider(
            "Plot width",
            1.0,
            4.0,
            step=0.1,
            key="plot_width",
            help="Width of the chart relative to the panels beside it. Lower it "
            "if the plot feels too wide for the page.",
        )
        st.slider("Axis title size", 12, 44, key="axis_title_size")
        st.slider("Tick label size", 10, 36, key="tick_size")
        st.slider("Axis line thickness", 1, 8, key="axis_width")
        st.checkbox("Bold axis titles and ticks", key="bold_axes")
        st.checkbox("Show grid", key="show_grid")
        st.checkbox("Log-log axes", key="log_scale", help="A power law is a straight line here.")
        st.checkbox("Show model components", key="show_components")
        st.checkbox("Shade the fit windows", key="show_fit_window")
        st.checkbox(
            "Show the cell diagram beside the plot",
            key="show_schematic",
            help="A side-on sketch of the membrane, cytoskeleton and nucleus at "
            "the deformation you select.",
        )
        st.checkbox(
            "Show the video frame beside the plot",
            key="video_show_panel",
            help="Only appears once a video is loaded in the Compression video tab.",
        )

    with st.expander("🗄️ Google Sheets database"):
        if SHEETS_IMPORT_ERROR:
            st.info("Database module unavailable in this environment.")
            st.caption(SHEETS_IMPORT_ERROR)
        else:
            st.checkbox("Enable database", key="db_enabled")
            if st.session_state["db_enabled"]:
                if st.button("🔗 Connect", **STRETCH):
                    manager = initialize_sheets_manager()
                    st.session_state["gs_manager"] = manager
                    if manager:
                        st.success("Connected.")
                    else:
                        st.error("Connection failed. Check credentials below.")
                if st.session_state["gs_manager"]:
                    st.caption("Connected ✓")
                st.caption(
                    "Needs a service-account key in `.streamlit/secrets.toml` under "
                    "`[google_sheets_credentials]`, and the sheet shared with that "
                    "service-account email."
                )

    st.divider()
    if st.button("↩️ Reset settings to defaults", **STRETCH):
        st.session_state["_reset_requested"] = True
        st.rerun()


# ================================================================= header ==

head_left, head_right = st.columns([4, 1])
with head_left:
    st.markdown('<p class="app-title">🔬 AFM Cell Analyzer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-sub">Lulevich two-term compression model · '
        "F(ε) = Aₘ·Eₘ·ε³ + Aᵢ·Eᵢ·ε³ᐟ²</p>",
        unsafe_allow_html=True,
    )
with head_right:
    st.markdown("<div style='text-align:right;color:#6b7785'>v6.0</div>", unsafe_allow_html=True)

tab_analysis, tab_video, tab_igor, tab_db, tab_results, tab_export = st.tabs(
    [
        "📊 Force curve analysis",
        "🎥 Compression video",
        "🔧 Create curve (Igor)",
        "📋 Database",
        "📈 Results",
        "💾 Export",
    ]
)


# ================================================ TAB 1: analysis workflow ==

with tab_analysis:
    section("1 · Cell information")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Cell name / ID", placeholder="C2C12_001", key="cell_name")
    with c2:
        date_acquired = st.date_input(
            "Date acquired", value=datetime.now().date(), key="date_acquired"
        )
    with c3:
        st.number_input(
            "Spring constant (N/m, optional)",
            min_value=0.0,
            max_value=100.0,
            step=0.001,
            format="%.3f",
            key="spring_constant",
        )
    hint(
        f"Cell height h₀ = {st.session_state['cell_height_um']:.2f} μm — set it in the "
        "sidebar under **Cell geometry**, it directly scales both moduli."
    )

    section("2 · Load force curve")
    uploaded = st.file_uploader(
        "Force vs relative deformation (.csv or .xlsx)",
        type=["csv", "xlsx", "xls"],
        key="force_curve_file",
    )

    if uploaded is not None:
        try:
            df = load_table(uploaded.getvalue(), uploaded.name)
        except Exception as exc:
            df = None
            st.error(f"Could not read the file: {exc}")

        if df is not None and df.shape[1] >= 2:
            columns = df.columns.tolist()
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                eps_col = st.selectbox(
                    "Relative deformation column",
                    columns,
                    index=guess_column(columns, ("deform", "eps", "ε", "strain"), 0),
                    key="eps_col",
                )
            with c2:
                force_col = st.selectbox(
                    "Force column",
                    columns,
                    index=guess_column(columns, ("force", "f (", "f["), 1),
                    key="force_col",
                )
            with c3:
                input_unit = st.selectbox(
                    "Force unit in the file",
                    list(INPUT_FORCE_UNITS.keys()),
                    index=3,
                    key="input_force_unit",
                )

            raw_eps = pd.to_numeric(df[eps_col], errors="coerce").to_numpy(dtype=float)
            raw_force = pd.to_numeric(df[force_col], errors="coerce").to_numpy(dtype=float)
            force_N = to_newtons(raw_force, input_unit)

            finite = np.isfinite(raw_eps) & np.isfinite(force_N)
            order = np.argsort(raw_eps[finite], kind="stable")
            epsilon = raw_eps[finite][order]
            force_N = force_N[finite][order]

            st.session_state["data"] = {
                "epsilon": epsilon,
                "force_N": force_N,
                "source": uploaded.name,
                "n_dropped": int((~finite).sum()),
            }
        elif df is not None:
            st.error("The file needs at least two columns (deformation and force).")

    data = st.session_state.get("data")

    if not data:
        st.info(
            "Upload a force curve to begin, or build one from Igor files in the "
            "**Create curve (Igor)** tab."
        )
    else:
        epsilon = data["epsilon"]
        force_N = data["force_N"]
        style = current_style(force_N)
        peak_display, unit_label = from_newtons(np.nanmax(force_N), style.force_unit)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Points", f"{epsilon.size:,}")
        c2.metric("ε range", f"{epsilon.min():.3f} to {epsilon.max():.3f}")
        c3.metric("Peak force", f"{float(peak_display):.4g} {unit_label}")
        c4.metric("Dropped rows", data["n_dropped"])
        if data["n_dropped"]:
            st.caption(f"{data['n_dropped']} non-numeric or blank rows were skipped.")

        # Relative deformation must be a fraction. A column in percent looks
        # plausible but silently scales both moduli, so catch it here.
        if epsilon.max() > 1.5:
            st.warning(
                f"ε reaches {epsilon.max():.1f}. Relative deformation should be a "
                "fraction between 0 and 1 — this column looks like percent.",
                icon="⚠️",
            )
            if st.checkbox("Divide the deformation column by 100", key="eps_percent_fix"):
                epsilon = epsilon / 100.0
                st.session_state["data"]["epsilon"] = epsilon
                st.caption(f"Now ε ∈ [{epsilon.min():.3f}, {epsilon.max():.3f}].")
        if epsilon.min() < 0:
            st.info(
                f"{int((epsilon < 0).sum())} points have ε < 0 (pre-contact). They are "
                "kept for the plot but excluded from any fit window starting at ε ≥ 0."
            )
        # ---------------------------------------------------- fit window ---        section("3 · Model and fit windows")

        model = build_model(epsilon, force_N)
        auto_range = model.auto_detect_elastic_range()
        rupture = model.results.get("rupture", {})

        eps_lo_data = float(max(epsilon.min(), 0.0))
        eps_hi_data = float(epsilon.max())
        step = max((eps_hi_data - eps_lo_data) / 200.0, 1e-4)
        auto_window = (
            float(np.clip(auto_range["elastic_epsilon_min"], eps_lo_data, eps_hi_data)),
            float(np.clip(auto_range["elastic_epsilon_max"], eps_lo_data, eps_hi_data)),
        )

        def clamp_range(pair, fallback):
            """Keep a stored window inside the current curve's bounds."""
            try:
                lo, hi = float(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError):
                return fallback
            lo = float(np.clip(lo, eps_lo_data, eps_hi_data))
            hi = float(np.clip(hi, eps_lo_data, eps_hi_data))
            return (lo, hi) if lo < hi else fallback

        term_col, model_col = st.columns([1, 1.4])
        with term_col:
            st.markdown("**Elements**")
            st.checkbox("Membrane · Eₘ", key="use_membrane")
            st.checkbox("Cytoskeleton · Ec", key="use_interior")
            st.checkbox("Nucleus · Eₙ", key="use_nucleus")
        with model_col:
            st.markdown("**How the elements share the load**")
            st.radio(
                "Coupling",
                list(COUPLINGS.keys()),
                key="coupling",
                label_visibility="collapsed",
                help="Parallel: every element is squashed by the same amount and "
                "their forces add, so the stiffest one dominates. Series: every "
                "element carries the same force and their deformations add, a "
                "spring stacked on a spring, so the softest one dominates. Hybrid "
                "switches between the two at a crossover deformation. Auto fits "
                "all of them and reports which the data supports.",
            )
            st.caption(COUPLINGS[st.session_state["coupling"]])

        active = active_terms()
        coupling = COUPLING_KEYS[st.session_state["coupling"]]

        if coupling in ("hybrid_ps", "hybrid_sp"):
            h1, h2 = st.columns([1, 2])
            with h1:
                st.radio(
                    "Crossover ε",
                    ["Scan for best", "Set manually"],
                    key="crossover_mode",
                    horizontal=True,
                )
            with h2:
                st.slider(
                    "ε at which the load path changes",
                    0.0, 1.0, step=0.01, key="crossover",
                    disabled=st.session_state["crossover_mode"] == "Scan for best",
                )

        # Stage-by-stage only makes sense for the parallel coupling: the series
        # fit is a single exact linear solve over the whole window, so there is
        # nothing for stages to buy, and the hybrid is one joint optimisation.
        staged_available = coupling == "parallel"
        if staged_available:
            proc_col, stage_col = st.columns([1, 2])
            with proc_col:
                st.radio(
                    "Fitting procedure",
                    ["All at once", "Stage by stage"],
                    key="procedure",
                    help="All at once solves for every modulus together on the "
                    "combined window. Stage by stage measures each group on its "
                    "own window, which helps when the moduli come out correlated.",
                )
            staged = st.session_state["procedure"] == "Stage by stage"
            if staged:
                with stage_col:
                    st.caption("Stage each element is fitted in. Same number = fitted together.")
                    stage_cols = st.columns(3)
                    slot = 0
                    for term in TERM_ORDER:
                        if st.session_state.get(f"use_{term}"):
                            with stage_cols[slot % 3]:
                                st.selectbox(
                                    TERM_LABELS[term], [1, 2, 3], key=f"stage_of_{term}"
                                )
                            slot += 1
                    r1, r2 = st.columns([2, 1.4])
                    with r1:
                        st.slider("Refinement passes", 1, 8, key="refine_iterations")
                    with r2:
                        st.checkbox("Seed from all-at-once", key="seed_parallel")
        else:
            staged = False
            st.caption(
                "Stage-by-stage fitting applies to parallel coupling. The series "
                "fit is one exact solve over the whole window and the hybrid is a "
                "single joint optimisation, so neither is split into stages."
            )

        # ------------------------------------------------------- windows ---
        st.markdown("**Deformation windows**")
        st.caption(
            f"Auto-detected usable region: ε ∈ [{auto_window[0]:.3f}, "
            f"{auto_window[1]:.3f}] ({auto_range['n_points']} points) · "
            f"rupture: {auto_range['rupture_method']}"
        )

        # Every window is always on screen: the combined one, and one per
        # element. Which of them the fit uses depends on the procedure, and the
        # caption under each says so, but none of them is ever hidden.
        st.session_state["window_combined"] = clamp_range(
            st.session_state.get("window_combined"), auto_window
        )
        combined_lo, combined_hi = st.slider(
            "Combined window (all elements together)",
            min_value=eps_lo_data, max_value=eps_hi_data, step=step,
            key="window_combined",
        )
        n_combined = int(((epsilon >= combined_lo) & (epsilon <= combined_hi)).sum())
        st.caption(
            f"{n_combined} points · "
            + ("used for this fit" if not staged else "not used while fitting stage by stage")
        )

        term_windows = {}
        window_cols = st.columns(max(1, len(active))) if active else [st]
        for i, term in enumerate(active):
            key = f"window_term_{term}"
            st.session_state[key] = clamp_range(
                st.session_state.get(key),
                default_window_for((term,), auto_window, eps_lo_data, eps_hi_data),
            )
            with window_cols[i % len(window_cols)]:
                lo, hi = st.slider(
                    TERM_LABELS[term],
                    min_value=eps_lo_data, max_value=eps_hi_data, step=step,
                    key=key,
                )
                term_windows[term] = (lo, hi)
                st.caption(f"{int(((epsilon >= lo) & (epsilon <= hi)).sum())} points")

        # A stage's window is the span of the windows of the elements in it.
        stage_plan = []
        if staged:
            for stage_no, terms in stage_groups(active):
                spans = [term_windows[t] for t in terms if t in term_windows]
                if not spans:
                    continue
                stage_plan.append(
                    {
                        "terms": terms,
                        "range": (min(s[0] for s in spans), max(s[1] for s in spans)),
                    }
                )
            fit_lo = min((s["range"][0] for s in stage_plan), default=combined_lo)
            fit_hi = max((s["range"][1] for s in stage_plan), default=combined_hi)
            st.caption(
                "Stages: "
                + " · ".join(
                    " + ".join(TERM_LABELS[t] for t in s["terms"])
                    + f" over {s['range'][0]:.3f} to {s['range'][1]:.3f}"
                    for s in stage_plan
                )
            )
        else:
            fit_lo, fit_hi = combined_lo, combined_hi
            stage_plan = [{"terms": active, "range": (fit_lo, fit_hi)}]

        w1, w2, w3 = st.columns(3)
        with w1:
            if st.button("↺ Reset every window to auto", **STRETCH):
                st.session_state["_pending_clear_windows"] = True
                st.rerun()
        with w2:
            if st.button("↺ Suggested split per element", **STRETCH):
                suggestion = model.suggest_sequential_windows()
                mapping = {
                    "interior": suggestion["interior_range"],
                    "membrane": suggestion["membrane_range"],
                    "nucleus": (
                        float(min(suggestion["membrane_range"][0] + 0.05, auto_window[1])),
                        float(auto_window[1]),
                    ),
                }
                st.session_state["_pending_settings"] = {
                    f"window_term_{term}": clamp_range(window, auto_window)
                    for term, window in mapping.items()
                }
                st.rerun()
        with w3:
            targets = ["(off)", "Combined window"] + [TERM_LABELS[t] for t in active]
            st.session_state["_drag_keys"] = {"Combined window": "window_combined"}
            st.session_state["_drag_keys"].update(
                {TERM_LABELS[t]: f"window_term_{t}" for t in active}
            )
            st.selectbox(
                "Drag on the plot to set",
                targets,
                key="drag_target",
                help="Pick a window, then box-select across the chart to set its "
                "ε limits from the drag.",
            )

        # ------------------------------------------------- saved presets ---
        with st.expander("💾 Saved windows", expanded=False):
            p1, p2 = st.columns([2, 1])
            with p1:
                preset_name = st.text_input(
                    "Preset name",
                    placeholder="e.g. C2C12 standard",
                    key="preset_name",
                    label_visibility="collapsed",
                )
            with p2:
                if st.button("Save current", **STRETCH):
                    name = (preset_name or "").strip()
                    if not name:
                        st.warning("Give the preset a name first.")
                    else:
                        st.session_state["range_presets"][name] = {
                            "coupling": st.session_state["coupling"],
                            "procedure": st.session_state["procedure"],
                            "combined_window": [float(combined_lo), float(combined_hi)],
                            "term_windows": {
                                t: [float(w[0]), float(w[1])]
                                for t, w in term_windows.items()
                            },
                            "crossover": float(st.session_state["crossover"]),
                            "cell_type": st.session_state["cell_type"],
                            "terms": list(active),
                            "stages": [
                                {"terms": list(s["terms"]), "range": list(s["range"])}
                                for s in stage_plan
                            ],
                            "stage_of": {t: st.session_state[f"stage_of_{t}"] for t in active},
                            "nucleus_onset": st.session_state["nucleus_onset"],
                            "saved_at": datetime.now().isoformat(timespec="seconds"),
                        }
                        st.success(f"Saved “{name}”.")

            presets = st.session_state["range_presets"]
            if presets:
                a1, a2, a3 = st.columns([2, 1, 1])
                with a1:
                    chosen = st.selectbox(
                        "Preset", list(presets.keys()), key="preset_choice",
                        label_visibility="collapsed",
                    )
                with a2:
                    if st.button("Apply", **STRETCH):
                        apply_preset(presets[chosen], eps_lo_data, eps_hi_data)
                        st.rerun()
                with a3:
                    if st.button("Delete", **STRETCH):
                        presets.pop(chosen, None)
                        st.rerun()

                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "preset": name,
                                "coupling": pre.get("coupling", "?"),
                                "cell type": pre.get("cell_type", "?"),
                                "windows": " | ".join(
                                    " + ".join(TERM_LABELS.get(t, t) for t in st_["terms"])
                                    + f" {st_['range'][0]:.3f} to {st_['range'][1]:.3f}"
                                    for st_ in pre.get("stages", [])
                                ),
                                "saved": pre.get("saved_at", ""),
                            }
                            for name, pre in presets.items()
                        ]
                    ),
                    hide_index=True,
                    **STRETCH,
                )

            e1, e2 = st.columns(2)
            with e1:
                st.download_button(
                    "📥 Export presets",
                    data=json.dumps(presets, indent=2),
                    file_name="afm_fit_windows.json",
                    mime="application/json",
                    disabled=not presets,
                    **STRETCH,
                )
            with e2:
                imported = st.file_uploader(
                    "Import presets (.json)", type=["json"], key="preset_upload"
                )
                if imported is not None:
                    try:
                        incoming = json.loads(imported.getvalue().decode("utf-8"))
                        st.session_state["range_presets"].update(incoming)
                        st.success(f"Imported {len(incoming)} preset(s).")
                    except Exception as exc:
                        st.error(f"Could not read that preset file: {exc}")
            hint(
                "Presets live in this browser session. Export them to a file to keep "
                "them between visits or share them with the rest of the lab."
            )

        # ----------------------------------------------------------- fit ---
        section("4 · Fit")

        if (
            "nucleus" in active
            and coupling == "parallel"
            and st.session_state["onset_mode"] == "Scan for best"
        ):
            scan = model.scan_nucleus_onset(
                fit_lo, fit_hi, terms=active,
                weighting=st.session_state["weighting"],
                fit_offset=st.session_state["fit_offset"],
            )
            if scan.get("success"):
                # Set it on the model only. Writing back to the slider's key
                # would raise, because the widget has already been built this
                # run; the scanned value is kept separately for display.
                model.nucleus_onset = scan["best_onset"]
                st.session_state["_scanned_onset"] = float(scan["best_onset"])
                if scan["well_determined"]:
                    st.success(
                        f"Best nucleus onset ε₀ = {scan['best_onset']:.3f} "
                        f"(R² = {scan['best_r_squared']:.4f})"
                    )
                else:
                    st.warning(
                        f"The onset scan is flat: every ε₀ between "
                        f"{scan['trials'][0]['onset']:.3f} and "
                        f"{scan['trials'][-1]['onset']:.3f} fits about equally well, so "
                        f"this curve does not locate the nucleus. Treat Eₙ as "
                        f"unidentified rather than measured."
                    )
        else:
            scan = None

        run = st.session_state["live_fit"] or st.button("🚀 Fit curve", type="primary")

        fit = None
        comparison = None
        if run and not active:
            st.warning("Select at least one element above.")
        elif run:
            if coupling == "auto":
                with st.spinner("Fitting every coupling and comparing…"):
                    comparison = compare_couplings(
                        model, fit_lo, fit_hi, terms=active
                    )
                if comparison.get("success"):
                    winner = comparison["best"]["coupling"]
                    fit = comparison["fits"][winner]
                    st.session_state["_auto_choice"] = winner
                else:
                    st.error(comparison.get("error", "Could not compare couplings."))
            elif coupling == "series":
                fit = model.fit_series(
                    fit_lo, fit_hi, terms=active,
                    weighting=st.session_state["weighting"],
                )
            elif coupling in ("hybrid_ps", "hybrid_sp"):
                order = (
                    "parallel-then-series" if coupling == "hybrid_ps"
                    else "series-then-parallel"
                )
                if st.session_state["crossover_mode"] == "Scan for best":
                    scan_x = model.scan_crossover(fit_lo, fit_hi, terms=active, order=order)
                    if scan_x.get("success"):
                        fit = scan_x["best"]
                        st.success(
                            f"Best crossover ε = {scan_x['best_crossover']:.3f} "
                            f"(R² = {fit['r_squared']:.4f})"
                        )
                    else:
                        st.error(scan_x.get("error", "Hybrid scan failed."))
                else:
                    crossover = float(
                        np.clip(st.session_state["crossover"], fit_lo + 1e-4, fit_hi - 1e-4)
                    )
                    fit = model.fit_hybrid(
                        fit_lo, fit_hi, crossover, terms=active, order=order
                    )
            elif staged and len(stage_plan) > 1:
                fit = model.fit_staged(
                    stage_plan,
                    weighting=st.session_state["weighting"],
                    fit_offset=st.session_state["fit_offset"],
                    refine_iterations=st.session_state["refine_iterations"],
                    seed_parallel=st.session_state["seed_parallel"],
                )
            else:
                fit = model.fit(
                    epsilon_min=fit_lo, epsilon_max=fit_hi, terms=active,
                    fit_offset=st.session_state["fit_offset"],
                    weighting=st.session_state["weighting"],
                )

        if comparison and comparison.get("success"):
            st.info(comparison["verdict"])
            table = pd.DataFrame(
                [
                    {
                        "coupling": row["label"],
                        "R²": round(row["r_squared"], 5),
                        "ΔAICc": round(row["delta_aicc"], 1),
                        "weight": round(row["weight"], 3),
                        "CV RMSE": f"{row['cv_rmse']:.3g}",
                        "params": row["n_params"],
                        "Eₘ (MPa)": (
                            f"{row['Em_MPa']:.4g}" if row.get("Em_MPa") is not None else "—"
                        ),
                        "Ec (kPa)": (
                            f"{row['Ei_kPa']:.4g}" if row.get("Ei_kPa") is not None else "—"
                        ),
                    }
                    for row in comparison["candidates"]
                ]
            )
            st.dataframe(table, hide_index=True, **STRETCH)
            st.caption(
                "ΔAICc is the penalty against the best model; a gap under 2 means "
                "the curve cannot tell them apart. Weight is the relative "
                "likelihood of each. CV RMSE is the error on points held out of "
                "the fit, which is the criterion to trust when the two disagree. "
                "Note that a wrong coupling can still reach R² > 0.99 with badly "
                "wrong moduli, which is exactly why this table exists."
            )

        if fit is None:
            st.info("Press **Fit curve**, or turn on live refitting in the sidebar.")
        elif not fit.get("success"):
            st.error(fit.get("error", "Fit failed."))
        else:
            En_value = fit.get("En", 0.0)
            fitted_coupling = fit.get("coupling", "parallel")
            params = (fit.get("Em", 0.0), fit.get("Ei", 0.0), En_value)
            params = tuple(0.0 if not np.isfinite(v) else v for v in params)

            if fitted_coupling == "parallel":
                fitted = model.combined_model(
                    epsilon, params[0], params[1],
                    fit.get("force_offset", 0.0), En=params[2],
                )
                # In parallel the elements share the deformation, so each one's
                # force is a separate curve that adds up to the total.
                membrane = model.balloon_model_cubic(epsilon, params[0])
                interior = model.hertzian_contact_model(epsilon, params[1])
                nucleus = model.nucleus_model(epsilon, params[2]) if params[2] else None
            else:
                base_coupling = "series" if fitted_coupling == "series" else "hybrid"
                fitted = model.predict(
                    epsilon, params, base_coupling,
                    fit.get("crossover"),
                    fit.get("order", "parallel-then-series"),
                )
                # In series every element carries the whole force, so there are
                # no separate force curves to draw; what differs between them is
                # how much of the deformation each one takes.
                membrane = interior = nucleus = None

            deformation_shares = None
            if fitted_coupling in ("series", "hybrid"):
                peak = float(np.nanmax(np.abs(force_N))) if force_N.size else 0.0
                pieces = {}
                if params[0] > 0:
                    pieces["membrane"] = (peak / (model.Am * params[0])) ** (1.0 / 3.0)
                if params[1] > 0:
                    pieces["interior"] = (peak / (model.Ai * params[1])) ** (2.0 / 3.0)
                if params[2] > 0:
                    onset = fit.get("nucleus_force_onset", 0.0)
                    pieces["nucleus"] = (
                        max(peak - onset, 0.0) / (model.An * params[2])
                    ) ** (2.0 / 3.0)
                total = sum(pieces.values())
                if total > 0:
                    deformation_shares = {k: v / total for k, v in pieces.items()}

            windows_for_plot = [
                {
                    "range": tuple(s["range"]),
                    "label": " + ".join(TERM_LABELS[t] for t in s["terms"]),
                    "color": STAGE_COLORS[i % len(STAGE_COLORS)],
                }
                for i, s in enumerate(stage_plan)
            ]

            st.session_state["results"] = {
                "cell_name": st.session_state["cell_name"],
                "cell_type": st.session_state["cell_type"],
                "date_acquired": str(date_acquired),
                "cell_height_um": st.session_state["cell_height_um"],
                "spring_constant": st.session_state["spring_constant"],
                "video_link": st.session_state["video_link"],
                "epsilon": epsilon,
                "force_N": force_N,
                "fitted_N": fitted,
                "membrane_N": membrane,
                "interior_N": interior,
                "nucleus_N": nucleus,
                "fit": fit,
                "fit_windows": windows_for_plot,
                "source": data["source"],
                "timestamp": datetime.now(),
            }

            metric_cols = st.columns(3 + (1 if "nucleus" in active else 0) + 1)
            metric_cols[0].metric(
                "Eₘ membrane",
                f"{fit['Em_MPa']:.3g} MPa",
                delta=f"± {fit['Em_MPa_std']:.2g}"
                if np.isfinite(fit.get("Em_MPa_std", np.nan))
                else None,
                delta_color="off",
            )
            metric_cols[1].metric(
                "Ec cytoskeleton",
                f"{fit['Ei_kPa']:.3g} kPa",
                delta=f"± {fit['Ei_kPa_std']:.2g}"
                if np.isfinite(fit.get("Ei_kPa_std", np.nan))
                else None,
                delta_color="off",
            )
            index = 2
            if "nucleus" in active:
                metric_cols[index].metric(
                    "Eₙ nucleus",
                    f"{fit.get('En_kPa', 0.0):.3g} kPa",
                    delta=f"onset ε₀ = {model.nucleus_onset:.3f}",
                    delta_color="off",
                )
                index += 1
            metric_cols[index].metric("R²", f"{fit['r_squared']:.4f}")
            rmse_disp, rmse_unit = from_newtons(fit["rmse"], style.force_unit)
            metric_cols[index + 1].metric("RMSE", f"{float(rmse_disp):.3g} {rmse_unit}")

            st.caption(
                f"Membrane areal modulus Eₘ·h = "
                f"{fit.get('membrane_areal_modulus', 0.0) * 1e3:.4g} mN/m, which is what "
                f"the ε³ term actually determines. Eₘ itself is that divided by the "
                f"assumed bilayer thickness of "
                f"{st.session_state['membrane_thickness_nm']:.1f} nm, so halving the "
                f"thickness doubles Eₘ while the measurement is unchanged."
            )

            if fitted_coupling != "parallel" and st.session_state["show_components"]:
                st.caption(
                    "Element curves are not drawn for series or hybrid coupling: "
                    "every element carries the same force there, so they would be "
                    "three copies of the total. The deformation share each one "
                    "takes is in the diagram beside the plot."
                )

            for message in fit["warnings"]:
                st.warning(message)

            # ------------------------------------------------ plot + panel
            video_ready = (
                not VIDEO_IMPORT_ERROR
                and st.session_state.get("video_path")
                and st.session_state.get("video_info")
                and os.path.exists(st.session_state["video_path"])
                and st.session_state["video_show_panel"]
            )
            show_schematic = st.session_state["show_schematic"]

            selected_eps = float(np.clip(fit_hi, eps_lo_data, eps_hi_data))
            if video_ready or show_schematic:
                selected_eps = st.slider(
                    "Show the cell at ε =",
                    min_value=eps_lo_data,
                    max_value=eps_hi_data,
                    step=step,
                    key="sync_eps",
                )
            nearest = int(np.argmin(np.abs(epsilon - selected_eps)))
            highlight = (float(epsilon[nearest]), float(force_N[nearest]))

            side_panels = int(bool(video_ready)) + int(bool(show_schematic))
            plot_weight = float(st.session_state["plot_width"])
            if side_panels:
                widths = [plot_weight] + [1.0] * side_panels
                columns = st.columns(widths)
                plot_col = columns[0]
                panel_cols = columns[1:]
            else:
                # Even with no side panel, keep the chart from spanning the
                # whole page; the spare column is left empty on purpose.
                plot_col, spare = st.columns([plot_weight, max(0.01, 4.0 - plot_weight)])
                panel_cols = []

            with plot_col:
                st.plotly_chart(
                    force_curve_figure(
                        epsilon,
                        force_N,
                        style,
                        title=st.session_state["cell_name"] or "Force vs relative deformation",
                        fit_force_N=fitted,
                        membrane_N=membrane,
                        interior_N=interior,
                        nucleus_N=nucleus,
                        fit_window=windows_for_plot,
                        rupture_epsilon=rupture.get("epsilon")
                        if rupture.get("method") == "force-drop"
                        else None,
                        highlight=highlight if (video_ready or show_schematic) else None,
                    ),
                    key="main_fit_plot",
                    **plot_selection_kwargs(),
                    **STRETCH,
                )
                apply_plot_drag("main_fit_plot", eps_lo_data, eps_hi_data)

            panel_index = 0
            if show_schematic and panel_index < len(panel_cols):
                with panel_cols[panel_index]:
                    st.plotly_chart(
                        cell_schematic(
                            style,
                            coupling=(
                                "series" if fitted_coupling == "series"
                                else "hybrid" if fitted_coupling == "hybrid"
                                else "parallel"
                            ),
                            shares=deformation_shares,
                            epsilon=selected_eps,
                            cell_height_um=st.session_state["cell_height_um"],
                            cell_radius_um=fit["R0"] * 1e6,
                            nucleus_radius_um=fit.get("R_nucleus", fit["R0"] * 0.35) * 1e6,
                            membrane_thickness_nm=st.session_state["membrane_thickness_nm"],
                            nucleus_onset=model.nucleus_onset if "nucleus" in active else None,
                            Em_MPa=fit["Em_MPa"],
                            Ei_kPa=fit["Ei_kPa"],
                            En_kPa=fit.get("En_kPa") if "nucleus" in active else None,
                            show_nucleus="nucleus" in active,
                        ),
                        key="schematic_plot",
                        **STRETCH,
                    )
                panel_index += 1

            if video_ready and panel_index < len(panel_cols):
                with panel_cols[panel_index]:
                    vinfo = st.session_state["video_info"]
                    frame_index = va.frame_for_epsilon(
                        selected_eps,
                        st.session_state["video_contact_frame"],
                        st.session_state["video_end_frame"] or (vinfo["n_frames"] - 1),
                        float(epsilon.max()),
                    )
                    vframe, vdet = cached_detection(
                        st.session_state["video_path"],
                        video_signature(),
                        int(frame_index),
                        st.session_state["video_roi"],
                        float(st.session_state["video_sensitivity"]),
                        bool(st.session_state["video_strip_lines"]),
                    )
                    if vframe is None:
                        st.info("Frame unavailable.")
                    else:
                        force_here, unit_here = from_newtons(highlight[1], style.force_unit)
                        snap = va.annotate(vframe, vdet, label=f"ε = {highlight[0]:.3f}")
                        st.image(
                            va.crop(snap, vdet) if vdet and vdet.get("found") else snap,
                            caption=f"Frame {frame_index} · ε = {highlight[0]:.3f} · "
                            f"F = {float(force_here):.3g} {unit_here}",
                            **STRETCH,
                        )
                        if vdet and vdet.get("found"):
                            st.caption(f"Cell height {vdet['height_px']:.0f} px")
                        st.download_button(
                            "📷 Save screenshot",
                            data=png_bytes(snap),
                            file_name=(
                                f"{st.session_state['cell_name'] or 'cell'}"
                                f"_eps{highlight[0]:.3f}.png"
                            ),
                            mime="image/png",
                            **STRETCH,
                        )

            # ------------------------------------------------- diagnostics
            with st.expander("🔍 Fit diagnostics"):
                share_cols = st.columns(4)
                share_cols[0].metric(
                    "Membrane share at ε_max",
                    f"{100 * fit['membrane_fraction_at_max']:.1f} %"
                    if np.isfinite(fit.get("membrane_fraction_at_max", np.nan))
                    else "n/a",
                )
                share_cols[1].metric(
                    "Cytoskeleton share",
                    f"{100 * fit['interior_fraction_at_max']:.1f} %"
                    if np.isfinite(fit.get("interior_fraction_at_max", np.nan))
                    else "n/a",
                )
                share_cols[2].metric(
                    "Nucleus share",
                    f"{100 * fit['nucleus_fraction_at_max']:.1f} %"
                    if np.isfinite(fit.get("nucleus_fraction_at_max", np.nan))
                    else "n/a",
                )
                if fit.get("mode") == "staged":
                    share_cols[3].metric("Refinement passes", fit["n_iterations"])
                else:
                    share_cols[3].metric(
                        "Condition number",
                        f"{fit['condition_number']:.1f}"
                        if np.isfinite(fit.get("condition_number", np.nan))
                        else "n/a",
                        help="How separable the terms are over this window. Above ~30 "
                        "the split between them is unreliable even though their sum "
                        "is well determined.",
                    )

                mask = fit["mask"]
                st.plotly_chart(
                    residual_figure(epsilon[mask], (force_N - fitted)[mask], style),
                    key="residual_plot",
                    **STRETCH,
                )

                if fit.get("mode") == "staged":
                    st.markdown("**Convergence across passes**")
                    st.caption(
                        "Each pass refits every stage with the other terms held at "
                        "their current values. "
                        + (
                            "The sequence started from a parallel fit over the whole "
                            "region, which stops the first stage from absorbing the "
                            "force that belongs to the later ones."
                            if fit.get("seeded")
                            else "Seeding is off, so pass 1 is the plain unseeded "
                            "staged result."
                        )
                    )
                    st.dataframe(
                        pd.DataFrame(fit["iterations"]).round(
                            {"Em_MPa": 4, "Ei_kPa": 4, "En_kPa": 4}
                        ),
                        hide_index=True,
                        **STRETCH,
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "stage": i + 1,
                                    "terms": ", ".join(
                                        TERM_LABELS[t] for t in plan["terms"]
                                    ),
                                    "ε window": f"{plan['range'][0]:.3f} to {plan['range'][1]:.3f}",
                                    "points": res["n_points"],
                                    "R² in window": round(float(res["r_squared"]), 4),
                                }
                                for i, (plan, res) in enumerate(
                                    zip(fit["stage_plan"], fit["stages"])
                                )
                            ]
                        ),
                        hide_index=True,
                        **STRETCH,
                    )
                else:
                    st.markdown("**How much does the answer depend on the window?**")
                    sens = model.range_sensitivity(
                        fit_lo,
                        fit_hi,
                        terms=active,
                        fit_offset=st.session_state["fit_offset"],
                        weighting=st.session_state["weighting"],
                    )
                    s1, s2 = st.columns(2)
                    s1.metric(
                        "Eₘ spread",
                        f"{100 * sens['Em_relative_spread']:.1f} %"
                        if np.isfinite(sens["Em_relative_spread"])
                        else "n/a",
                    )
                    s2.metric(
                        "Ec spread",
                        f"{100 * sens['Ei_relative_spread']:.1f} %"
                        if np.isfinite(sens["Ei_relative_spread"])
                        else "n/a",
                    )
                    st.caption(
                        "Range of each modulus across shrinking upper bounds, as a "
                        "fraction of its mean. Under ~10 % the fit is robust; much "
                        "more means the curve does not constrain the terms separately."
                    )
                    figure = sensitivity_figure(sens["trials"], style)
                    if figure is not None:
                        st.plotly_chart(figure, key="sensitivity_plot", **STRETCH)

                if scan and scan.get("success"):
                    st.markdown("**Nucleus onset scan**")
                    st.caption(
                        "R² against the assumed onset. A sharp peak means the curve "
                        "locates the nucleus; a flat line means it does not."
                    )
                    st.line_chart(
                        pd.DataFrame(scan["trials"]).set_index("onset")[["r_squared"]],
                        height=240,
                    )

                st.markdown("**Geometry prefactors actually used**")
                st.code(
                    f"h0 = {fit['cell_height'] * 1e6:.3f} um\n"
                    f"R0 = {fit['R0'] * 1e6:.3f} um\n"
                    f"R_nucleus = {fit.get('R_nucleus', float('nan')) * 1e6:.3f} um\n"
                    f"h_membrane = {st.session_state['membrane_thickness_nm']:.2f} nm\n"
                    f"Am = {fit['Am']:.4e} N/Pa   (F_membrane = Am*Em*eps^3)\n"
                    f"Ai = {fit['Ai']:.4e} N/Pa   (F_cyto = Ai*Ec*eps^1.5)\n"
                    f"An = {fit.get('An', float('nan')):.4e} N/Pa   "
                    f"(F_nucleus = An*En*<eps-eps0>^1.5)",
                    language="text",
                )

            section("5 · Save")
            c1, c2 = st.columns([2, 1])
            with c1:
                if st.session_state["video_link"]:
                    st.caption(f"Video link on record: {st.session_state['video_link']}")
                else:
                    st.caption(
                        "No video linked. Add one in the **Compression video** tab to "
                        "store it with this cell."
                    )
            with c2:
                st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
                disabled = not (st.session_state["db_enabled"] and st.session_state["gs_manager"])
                if st.button(
                    "💾 Save to database",
                    **STRETCH,
                    disabled=disabled,
                    help="Enable and connect the database in the sidebar first."
                    if disabled
                    else None,
                ):
                    if not st.session_state["cell_name"]:
                        st.error("Give the cell a name first.")
                    else:
                        ok, msg = st.session_state["gs_manager"].append_cell_data(
                            {
                                "cell_id": st.session_state["cell_name"],
                                "date_analyzed": str(date_acquired),
                                "cell_height": st.session_state["cell_height_um"],
                                "cantilever_constant": (
                                    f"{st.session_state['spring_constant']} N/m"
                                    if st.session_state["spring_constant"] > 0
                                    else "N/A"
                                ),
                                "Em": round(fit["Em_MPa"], 6),
                                "Ei": round(fit["Ei_kPa"], 6),
                                "video_link": st.session_state["video_link"],
                                "force_curve_created": "Yes",
                                "fit_quality": round(float(fit["r_squared"]), 4),
                                "notes": f"ε ∈ [{fit_lo:.3f}, {fit_hi:.3f}], "
                                f"{fit['n_points']} pts, {fit['weighting']} weighting",
                                "analysis_status": "Complete",
                            }
                        )
                        (st.success if ok else st.warning)(msg)


# ==================================================== TAB 2: video analysis ==

with tab_video:
    section("Compression video")

    if VIDEO_IMPORT_ERROR:
        st.error(
            f"Video analysis unavailable: {VIDEO_IMPORT_ERROR}. "
            f"Add `opencv-python-headless` to requirements.txt."
        )
    else:
        st.markdown(
            "Load the compression video to put a picture of the cell next to the "
            "curve, and to derive deformation from the cell's own shape as a check "
            "on the contact point and cell height."
        )

        src1, src2 = st.columns([1, 1])
        with src1:
            uploaded_video = st.file_uploader(
                "Upload video", type=["mp4", "avi", "mov", "wmv", "mkv"], key="video_file"
            )
        with src2:
            st.text_input(
                "…or a Google Drive link",
                placeholder="https://drive.google.com/file/d/...",
                key="video_link",
                help="Only works for files shared with 'anyone with the link'.",
            )
            if st.button("⬇️ Fetch from Drive", **STRETCH):
                try:
                    with st.spinner("Downloading…"):
                        dest = os.path.join(tempfile.gettempdir(), "afm_drive_video.mp4")
                        va.download_drive_video(st.session_state["video_link"], dest)
                    st.session_state["video_path"] = dest
                    st.session_state["video_name"] = "Google Drive video"
                    st.session_state["video_info"] = va.probe(dest)
                    st.session_state["video_track"] = None
                    st.rerun()
                except Exception as exc:
                    st.error(f"{exc}")

        if uploaded_video is not None and st.session_state["video_name"] != uploaded_video.name:
            path = os.path.join(tempfile.gettempdir(), f"afm_video_{uploaded_video.name}")
            with open(path, "wb") as handle:
                handle.write(uploaded_video.getvalue())
            st.session_state["video_path"] = path
            st.session_state["video_name"] = uploaded_video.name
            st.session_state["video_track"] = None
            try:
                st.session_state["video_info"] = va.probe(path)
            except Exception as exc:
                st.session_state["video_info"] = None
                st.error(f"Could not open that video: {exc}")

        info = st.session_state.get("video_info")
        path = st.session_state.get("video_path")

        if not (info and path and os.path.exists(path)):
            st.info("No video loaded yet.")
        else:
            signature = video_signature()
            n_frames = int(info["n_frames"])
            last = max(0, n_frames - 1)

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Frames", f"{n_frames:,}")
            v2.metric("FPS", f"{info['fps']:.1f}" if info["fps"] else "n/a")
            v3.metric("Size", f"{info['width']}×{info['height']}")
            v4.metric(
                "Duration",
                f"{info['duration_s']:.1f} s" if np.isfinite(info["duration_s"]) else "n/a",
            )

            # ------------------------------------------------ detection setup
            section("Detection")
            d1, d2 = st.columns([1, 1])
            with d1:
                st.slider(
                    "Edge sensitivity",
                    0.3,
                    3.0,
                    step=0.1,
                    key="video_sensitivity",
                    help="Raise it if the cell is faint; lower it if background "
                    "texture is being picked up instead.",
                )
                st.checkbox(
                    "Ignore long horizontal structures",
                    key="video_strip_lines",
                    help="Removes the substrate line and the cantilever, which "
                    "otherwise merge with the cell into one shapeless blob.",
                )
            with d2:
                st.caption("Search region (fraction of the frame)")
                rx = st.slider("Horizontal", 0.0, 1.0, step=0.01, key="video_roi_x")
                ry = st.slider("Vertical", 0.0, 1.0, step=0.01, key="video_roi_y")
                st.session_state["video_roi"] = (rx[0], ry[0], rx[1], ry[1])

            roi = st.session_state["video_roi"]
            # Widget state must be clamped to this video before the widgets are
            # built, or a shorter clip than the previous one raises.
            st.session_state.setdefault("video_preview_frame", last // 2)
            st.session_state["video_preview_frame"] = int(
                np.clip(st.session_state["video_preview_frame"], 0, last)
            )
            st.session_state["video_contact_frame"] = int(
                np.clip(st.session_state["video_contact_frame"], 0, last)
            )
            if not st.session_state["video_end_frame"]:
                st.session_state["video_end_frame"] = last
            st.session_state["video_end_frame"] = int(
                np.clip(st.session_state["video_end_frame"], 0, last)
            )

            preview_frame = st.slider("Preview frame", 0, last, key="video_preview_frame")
            frame, det = cached_detection(
                path,
                signature,
                int(preview_frame),
                roi,
                float(st.session_state["video_sensitivity"]),
                bool(st.session_state["video_strip_lines"]),
            )

            p1, p2 = st.columns([2, 1])
            with p1:
                if frame is None:
                    st.error("Could not read that frame.")
                else:
                    label = (
                        f"h = {det['height_px']:.0f} px" if det and det.get("found") else "not found"
                    )
                    st.image(
                        va.annotate(frame, det, label=label),
                        caption=f"Frame {preview_frame}",
                        **STRETCH,
                    )
            with p2:
                if det and det.get("found"):
                    st.success("Cell detected")
                    st.metric("Height", f"{det['height_px']:.0f} px")
                    st.metric("Width", f"{det['width_px']:.0f} px")
                    st.caption(
                        f"circularity {det['circularity']:.2f} · "
                        f"solidity {det['solidity']:.2f}"
                    )
                    st.download_button(
                        "📷 Save this frame",
                        data=png_bytes(va.annotate(frame, det, label=label)),
                        file_name=f"frame_{preview_frame}.png",
                        mime="image/png",
                        **STRETCH,
                    )
                elif det:
                    st.warning(f"No cell found: {det.get('reason', 'unknown')}")
                    st.caption(
                        "Try narrowing the search region to exclude the cantilever, "
                        "or adjusting the edge sensitivity."
                    )

            # ------------------------------------------------ curve alignment
            section("Line the video up with the curve")
            st.caption(
                "Mark the frame where the cantilever first touches the cell and the "
                "frame at the end of the ramp. Deformation is then assumed to grow "
                "linearly with frame number between them, which holds for a "
                "constant-speed approach."
            )
            a1, a2 = st.columns(2)
            with a1:
                contact_frame = st.number_input(
                    "Contact frame (ε = 0)", 0, last, key="video_contact_frame"
                )
            with a2:
                end_frame = st.number_input(
                    "End-of-ramp frame", 0, last, key="video_end_frame"
                )

            if st.button("🔬 Track the cell through the video", type="primary", **STRETCH):
                try:
                    st.session_state["video_track"] = cached_track(
                        path,
                        signature,
                        60,
                        roi,
                        float(st.session_state["video_sensitivity"]),
                        bool(st.session_state["video_strip_lines"]),
                        int(contact_frame),
                        int(end_frame),
                    )
                except Exception as exc:
                    st.error(f"Tracking failed: {exc}")

            track = st.session_state.get("video_track")
            if track:
                found = int(np.sum(track["found"]))
                total = len(track["frames"])
                eps_video, h_ref = va.deformation_from_track(track, reference="max")

                t1, t2, t3 = st.columns(3)
                t1.metric("Frames with a cell", f"{found}/{total}")
                t2.metric("Reference height", f"{h_ref:.0f} px" if np.isfinite(h_ref) else "n/a")
                t3.metric(
                    "Max deformation seen",
                    f"{np.nanmax(eps_video):.3f}" if np.isfinite(eps_video).any() else "n/a",
                )

                if found < total * 0.5:
                    st.warning(
                        "The cell was found in fewer than half the sampled frames. "
                        "Narrow the search region or adjust the sensitivity before "
                        "trusting the comparison below."
                    )

                data_now = st.session_state.get("data")
                if data_now is not None and np.isfinite(eps_video).any():
                    eps_curve_at_frames = np.array(
                        [
                            va.epsilon_for_frame(
                                f, contact_frame, end_frame, float(np.nanmax(data_now["epsilon"]))
                            )
                            for f in track["frames"]
                        ]
                    )
                    scale, r2 = va.align_scale(eps_video, eps_curve_at_frames)

                    st.markdown("**Video deformation vs the curve's deformation axis**")
                    comparison = pd.DataFrame(
                        {
                            "frame": track["frames"],
                            "ε from video": eps_video,
                            "ε from curve": eps_curve_at_frames,
                        }
                    ).dropna()
                    st.line_chart(comparison.set_index("frame"), height=280)

                    c1, c2 = st.columns(2)
                    c1.metric(
                        "Scale factor",
                        f"{scale:.3f}" if np.isfinite(scale) else "n/a",
                        help="Video deformation divided by curve deformation. 1.0 means "
                        "they agree.",
                    )
                    c2.metric("Agreement R²", f"{r2:.3f}" if np.isfinite(r2) else "n/a")

                    if np.isfinite(scale) and abs(scale - 1.0) > 0.15:
                        suggested = st.session_state["cell_height_um"] * scale
                        st.warning(
                            f"The video says the cell deformed {scale:.2f}× as much as the "
                            f"curve's ε axis claims. The usual cause is the cell height: "
                            f"{st.session_state['cell_height_um']:.2f} μm would need to be "
                            f"about {suggested:.2f} μm for the two to agree. A wrong contact "
                            f"point does the same thing."
                        )
                    elif np.isfinite(scale):
                        st.success(
                            "Video and force curve agree on how much the cell deformed."
                        )

# =================================================== TAB 3: Igor generator ==

with tab_igor:
    section("Build a force curve from Igor .ibw files")

    if IGOR_IMPORT_ERROR:
        st.error(f"Igor tools unavailable: {IGOR_IMPORT_ERROR}")
    else:
        st.info(
            "Upload the hard-surface reference and the cell compression curve. "
            "The result is a CSV of force vs relative deformation that you can "
            "load in the analysis tab."
        )
        st.warning(
            "The .ibw reader is a heuristic byte scanner, and the Z axis is "
            "reconstructed from the ramp settings below rather than read from the "
            "file. Check the generated curve against a known measurement before "
            "trusting the moduli it leads to.",
            icon="⚠️",
        )

        c1, c2 = st.columns(2)
        with c1:
            igor_surface = st.file_uploader(
                "Surface reference (.ibw)", type=["ibw"], key="igor_surface"
            )
        with c2:
            igor_cell = st.file_uploader(
                "Cell compression (.ibw)", type=["ibw"], key="igor_cell"
            )

        def parse_ibw(upload):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ibw") as tmp:
                tmp.write(upload.getvalue())
                path = tmp.name
            try:
                return IgorParser(path).parse().get("data")
            finally:
                os.unlink(path)

        if igor_surface is not None and igor_cell is not None:
            data_surface = parse_ibw(igor_surface)
            data_cell = parse_ibw(igor_cell)

            c1, c2 = st.columns(2)
            c1.metric("Surface points", f"{0 if data_surface is None else len(data_surface):,}")
            c2.metric("Cell points", f"{0 if data_cell is None else len(data_cell):,}")

            if data_cell is None:
                st.error("Could not extract wave data from the cell file.")
            else:
                st.markdown("**Acquisition parameters**")
                p1, p2, p3 = st.columns(3)
                with p1:
                    k_cantilever = st.number_input(
                        "Cantilever spring constant (N/m)",
                        min_value=0.0001,
                        max_value=1000.0,
                        value=0.05,
                        step=0.001,
                        format="%.4f",
                    )
                with p2:
                    z_total_um = st.number_input(
                        "Total Z ramp (μm)",
                        min_value=0.01,
                        max_value=200.0,
                        value=10.0,
                        step=0.1,
                        help="Piezo travel covered by the wave, used to build the Z axis.",
                    )
                with p3:
                    height_igor_um = st.number_input(
                        "Cell height h₀ (μm)",
                        min_value=0.1,
                        max_value=100.0,
                        value=float(st.session_state["cell_height_um"]),
                        step=0.1,
                    )

                d1, d2 = st.columns(2)
                with d1:
                    defl_unit = st.selectbox(
                        "Deflection wave units", ["metres", "volts"], index=0
                    )
                with d2:
                    invols_nm_v = st.number_input(
                        "InvOLS (nm/V)",
                        min_value=0.1,
                        max_value=1000.0,
                        value=50.0,
                        step=1.0,
                        disabled=defl_unit == "metres",
                        help="Only used when the deflection wave is in volts.",
                    )

                subtract_deflection = st.checkbox(
                    "Indentation = Δz − Δd (subtract cantilever bending)",
                    value=True,
                    help="Piezo travel overstates the indentation by the amount the "
                    "cantilever itself bends. Leave on unless your wave is already "
                    "a true indentation.",
                )

                if st.button("⚙️ Generate force curve", type="primary"):
                    try:
                        deflection = np.asarray(data_cell, dtype=float)
                        if defl_unit == "volts":
                            deflection = deflection * invols_nm_v * 1e-9  # V -> m

                        corrector = BaselineCorrector(deflection, np.arange(deflection.size))
                        corrector.auto_detect_baseline(method="flat")
                        deflection = corrector.correct_baseline()

                        force_N_gen = deflection * k_cantilever  # N/m · m = N

                        z_m = np.linspace(0.0, z_total_um * 1e-6, deflection.size)
                        contact_idx = int(corrector.estimate_contact_point(deflection))
                        delta = np.abs(z_m - z_m[contact_idx])
                        if subtract_deflection:
                            delta = delta - np.abs(deflection - deflection[contact_idx])
                        delta = np.clip(delta, 0.0, None)
                        delta[:contact_idx] = 0.0

                        eps_gen = delta / (height_igor_um * 1e-6)

                        out = pd.DataFrame(
                            {
                                "Relative Deformation": eps_gen,
                                "Force (nN)": force_N_gen * 1e9,
                                "Indentation (um)": delta * 1e6,
                                "Z (um)": z_m * 1e6,
                            }
                        )

                        st.success(
                            f"Generated {len(out):,} points · contact at index {contact_idx} "
                            f"· peak force {force_N_gen.max() * 1e9:.3g} nN "
                            f"· ε up to {eps_gen.max():.3f}"
                        )

                        gen_style = current_style(force_N_gen)
                        st.plotly_chart(
                            force_curve_figure(
                                eps_gen,
                                force_N_gen,
                                gen_style,
                                title="Generated force curve",
                            ),
                            **STRETCH,
                            key="igor_plot",
                        )
                        st.dataframe(out.head(25), hide_index=True, **STRETCH)

                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button(
                                "📥 Download CSV",
                                data=out.to_csv(index=False),
                                file_name="force_curve_generated.csv",
                                mime="text/csv",
                                **STRETCH,
                            )
                        with c2:
                            if st.button("➡️ Use this curve now", **STRETCH):
                                st.session_state["data"] = {
                                    "epsilon": eps_gen,
                                    "force_N": force_N_gen,
                                    "source": "Igor-generated",
                                    "n_dropped": 0,
                                }
                                st.success("Loaded into the analysis tab.")
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")


# ======================================================== TAB 3: database ==

with tab_db:
    section("Database browser")
    manager = st.session_state.get("gs_manager")
    if not (st.session_state["db_enabled"] and manager):
        st.info("Enable and connect the Google Sheets database in the sidebar.")
    else:
        df_all = manager.get_all_cells()
        if df_all.empty:
            st.info("No cells stored yet.")
        else:
            c1, c2 = st.columns([3, 1])
            with c1:
                search = st.text_input("Search cell ID", placeholder="C2C12_001")
            with c2:
                limit = st.number_input(
                    "Rows", min_value=5, max_value=max(5, len(df_all)), value=min(25, len(df_all))
                )
            shown = manager.search_cells(search) if search else df_all
            st.dataframe(shown.head(int(limit)), hide_index=True, **STRETCH)

            stats = manager.get_statistics()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total cells", stats.get("total_cells", 0))
            s2.metric(
                "Mean Eₘ",
                f"{stats.get('avg_em', 0):.3g} MPa" if stats.get("avg_em", 0) else "n/a",
            )
            s3.metric(
                "Mean Eᵢ",
                f"{stats.get('avg_ei', 0):.3g} kPa" if stats.get("avg_ei", 0) else "n/a",
            )
            s4.metric("With video", stats.get("cells_with_video", 0))


# ========================================================= TAB 4: results ==

with tab_results:
    section("Latest result")
    results = st.session_state.get("results")
    if not results:
        st.info("Fit a curve in the analysis tab first.")
    else:
        fit = results["fit"]
        style = current_style(results["force_N"])

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Cell**")
            st.write(f"Name: {results['cell_name'] or '—'}")
            st.write(f"Date: {results['date_acquired']}")
            st.write(f"Height: {results['cell_height_um']:.2f} μm")
            st.write(f"Source: {results['source']}")
            st.write(f"Fitted: {results['timestamp']:%Y-%m-%d %H:%M}")
        with c2:
            st.markdown("**Mechanics**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Eₘ", f"{fit['Em_MPa']:.3g} MPa")
            m2.metric("Eᵢ", f"{fit['Ei_kPa']:.3g} kPa")
            m3.metric("R²", f"{fit['r_squared']:.4f}")
            st.caption(
                f"Window ε ∈ [{fit['epsilon_range'][0]:.3f}, {fit['epsilon_range'][1]:.3f}] · "
                f"{fit['n_points']} points · {fit['weighting']} weighting · "
                f"bending constant Kₘ = {fit['Km_kT']:.3g} k_BT"
            )

        st.plotly_chart(
            force_curve_figure(
                results["epsilon"],
                results["force_N"],
                style,
                title=results["cell_name"] or "Force vs relative deformation",
                fit_force_N=results["fitted_N"],
                membrane_N=results["membrane_N"],
                interior_N=results["interior_N"],
                fit_window=tuple(fit["epsilon_range"]),
            ),
            **STRETCH,
            key="results_tab_plot",
        )


# ========================================================== TAB 5: export ==

with tab_export:
    section("Export")
    results = st.session_state.get("results")

    st.markdown("**This analysis**")
    if not results:
        st.info("Nothing to export yet.")
    else:
        fit = results["fit"]
        curve = pd.DataFrame(
            {
                "relative_deformation": results["epsilon"],
                "force_N": results["force_N"],
                "fit_N": results["fitted_N"],
                "membrane_term_N": results["membrane_N"],
                "interior_term_N": results["interior_N"],
                "residual_N": results["force_N"] - results["fitted_N"],
            }
        )
        summary = {
            "cell_name": results["cell_name"],
            "date_acquired": results["date_acquired"],
            "cell_height_um": results["cell_height_um"],
            "R0_um": fit["R0"] * 1e6,
            "spring_constant_N_per_m": results["spring_constant"],
            "Em_MPa": fit["Em_MPa"],
            "Em_MPa_std": fit["Em_MPa_std"],
            "Ei_kPa": fit["Ei_kPa"],
            "Ei_kPa_std": fit["Ei_kPa_std"],
            "coupling": fit.get("coupling", "parallel"),
            "crossover": fit.get("crossover"),
            "force_offset_N": fit.get("force_offset", 0.0),
            "r_squared": fit["r_squared"],
            "adj_r_squared": fit.get("adj_r_squared", float("nan")),
            "rmse_N": fit["rmse"],
            "epsilon_min": fit["epsilon_range"][0],
            "epsilon_max": fit["epsilon_range"][1],
            "n_points": fit["n_points"],
            "weighting": fit["weighting"],
            "fit_offset_enabled": fit.get("fit_offset", False),
            "condition_number": fit.get("condition_number", float("nan")),
            "membrane_fraction_at_max": fit.get("membrane_fraction_at_max", float("nan")),
            "warnings": fit["warnings"],
            "source": results["source"],
            "analysed_at": results["timestamp"].isoformat(),
        }

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "📥 Curve + fit (CSV)",
                data=curve.to_csv(index=False),
                file_name=f"{results['cell_name'] or 'cell'}_fit.csv",
                mime="text/csv",
                **STRETCH,
            )
        with c2:
            st.download_button(
                "📥 Parameters (JSON)",
                data=json.dumps(summary, indent=2, default=str),
                file_name=f"{results['cell_name'] or 'cell'}_parameters.json",
                mime="application/json",
                **STRETCH,
            )
        with c3:
            st.download_button(
                "📥 Parameters (CSV)",
                data=pd.DataFrame([{k: v for k, v in summary.items() if k != "warnings"}]).to_csv(
                    index=False
                ),
                file_name=f"{results['cell_name'] or 'cell'}_parameters.csv",
                mime="text/csv",
                **STRETCH,
            )

    st.divider()
    st.markdown("**Whole database**")
    manager = st.session_state.get("gs_manager")
    if not (st.session_state["db_enabled"] and manager):
        st.info("Connect the database in the sidebar to export all cells.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📥 All cells (CSV)",
                data=manager.export_to_csv() or "",
                file_name=f"afm_cells_{datetime.now():%Y%m%d}.csv",
                mime="text/csv",
                **STRETCH,
            )
        with c2:
            st.download_button(
                "📥 All cells (JSON)",
                data=manager.export_to_json() or "",
                file_name=f"afm_cells_{datetime.now():%Y%m%d}.json",
                mime="application/json",
                **STRETCH,
            )
