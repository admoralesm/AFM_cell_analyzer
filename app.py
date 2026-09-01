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
    exponent_profile_figure,
    force_curve_figure,
    from_newtons,
    residual_figure,
    sensitivity_figure,
    to_newtons,
)

# --------------------------------------------------- companion file check ---
#
# These files are updated together, and a deploy that picks up one but not
# another fails somewhere deep inside a call with a TypeError that Streamlit
# Cloud redacts, which tells you nothing. So check up front for the pieces
# this version of app.py needs, name the file that is behind, and carry on
# without the missing feature rather than crashing.

def _missing_pieces():
    """Which companion files are older than this app.py expects."""
    import inspect

    stale = []

    try:
        params = inspect.signature(force_curve_figure).parameters
        if "highlight_window" not in params:
            stale.append(
                ("plot_utils.py", "the highlighted segment band on the curve")
            )
    except (TypeError, ValueError):  # pragma: no cover - builtin or C function
        pass

    try:
        schematic_params = inspect.signature(cell_schematic).parameters
        if "membrane_mode" not in schematic_params:
            stale.append(
                ("plot_utils.py", "the diagram following the chosen combination")
            )
    except (TypeError, ValueError):  # pragma: no cover
        pass

    for method, feature in (
        ("fit_composition", "the segmented fit"),
        ("search_compositions", "the “Try every combination” search"),
        ("composition_terms", "the fitted components on the plot"),
    ):
        if not hasattr(LulevichModel, method):
            stale.append(("lulevich_model.py", feature))

    return stale


STALE_FILES = _missing_pieces()


def figure_kwargs(function, **kwargs):
    """
    Drop keyword arguments the installed version of a function cannot take.

    Without this, one file left behind in a deploy takes the whole app down.
    With it, the feature that file carries is simply missing until it is
    updated, and the banner at the top says which file to update.
    """
    import inspect

    try:
        accepted = inspect.signature(function).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in accepted}


# Optional dependencies: the app must still run without Google credentials
# or the Igor toolchain installed.
try:
    from google_sheets_manager import initialize_sheets_manager

    SHEETS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local install
    initialize_sheets_manager = None
    SHEETS_IMPORT_ERROR = str(exc)

try:
    import box_store
    from box_store import BoxStore, BoxError

    BOX_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    box_store = None
    BoxStore = None
    BoxError = Exception
    BOX_IMPORT_ERROR = str(exc)

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


if STALE_FILES:
    files = sorted({name for name, _ in STALE_FILES})
    features = "\n".join(f"- {feature} (from `{name}`)" for name, feature in STALE_FILES)
    st.error(
        "**Some files are older than this version of `app.py`.**\n\n"
        "Copy the current " + " and ".join(f"`{f}`" for f in files)
        + " into the repository, then reboot the app. Until then these are "
        "switched off:\n\n" + features
    )


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
    # Off by default: the fit is one Lulevich curve, and the per-element
    # curves beside it invite reading three fits into a plot that has one.
    "show_components": False,
    "bare_plot": False,
    "show_legend": True,
    "show_fit_window": True,
    "show_video_marker": True,
    "show_rupture_marker": True,
    "show_schematic": True,
    "show_schematic_moduli": True,
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
    "model_kind": "Segmented (membrane → cytoskeleton → nucleus)",
    "segment_break_1": 0.15,
    "segment_break_2": 0.40,
    # What each element does either side of the first boundary. These are the
    # two choices the physics leaves open, and the combination search below
    # settles them from the data when you ask it to.
    "membrane_after_break": "holds what it reached",
    "cyto_starts_at": "at ε₁",
    "highlight_segment": "(none)",
    "composition_search": None,
    # The segmented model always starts at zero, so only the far end is set.
    "window_end": 0.60,
    "procedure": "All at once",
    "crossover_mode": "Scan for best",
    "crossover": 0.18,
    "use_membrane": True,
    "use_interior": True,
    "use_nucleus": True,
    "regime_mode": False,
    "regime_split": 0.40,
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
    "invols_nm_per_V": 50.0,
    "operator": "",
    "cell_notes": "",
    "box_root_folder": "",
    "box_store": None,
    "box_index": None,
    "db_selection": [],
    "upload_video_with_cell": False,
    "db_search": "",
    "db_view": "Gallery",
    "exploration": None,
    "video_link": "",
    # The last successful fit, kept so a rerun (uploading a video, ticking a
    # box, changing tab) does not wipe the results off the page.
    "_last_fit": None,
    "_last_fit_signature": None,
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
    "video_clahe": 0.0,
    "video_gamma": 1.0,
    "video_brightness": 0,
    "video_contrast": 1.0,
    "video_cell_side": "anywhere",
    # In phase contrast the cell is usually the clear, bright object; saying
    # so stops a dark patch of debris winning on shape alone.
    "video_appearance": "clear",
    "video_reject_dark": True,
    "video_find_nucleus": True,
    "video_strip_lines": True,
    "video_show_panel": True,
    # data / results
    "data": None,
    "results": None,
    "gs_manager": None,
    "db_enabled": False,
    # The lab's existing sheet, used when the optional Sheets mirror is on.
    "sheet_id": "1FYnQGcaSiAAx1GUNqi_6sWGHhmf6n7vuS7l-bRceJxM",
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


# The extras that can be drawn on the force curve, all of which the single
# "Data and fit only" switch turns off together.
PLOT_EXTRAS = (
    "show_components",
    "show_fit_window",
    "show_video_marker",
    "show_rupture_marker",
    "show_legend",
)


def plot_flags(state):
    """
    Which plot extras are drawn.

    One switch beats five: "Data and fit only" overrides the individual boxes
    rather than merely unticking them, so a plot cannot end up half cleaned
    with no obvious reason why something is still on it.
    """
    def value(name):
        # Streamlit's session state is dict-like but has no .get, and a plain
        # dict is what the tests pass, so read both the same way.
        try:
            return bool(state[name])
        except (KeyError, TypeError):
            return False

    bare = value("bare_plot")
    return {name: (not bare) and value(name) for name in PLOT_EXTRAS}


def current_style(force_N=None) -> PlotStyle:
    """Build the PlotStyle from the sidebar settings, honouring auto units."""
    unit = st.session_state["force_unit"]
    if unit == "auto":
        unit = autoscale_unit(force_N) if force_N is not None and np.size(force_N) else "nN"

    on = plot_flags(st.session_state).get

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
        # One switch beats four: "Data and fit only" wins over the individual
        # ones so a person cannot half-clean a plot and wonder what is left.
        show_components=on("show_components"),
        show_fit_window=on("show_fit_window"),
        show_video_marker=on("show_video_marker"),
        show_rupture_marker=on("show_rupture_marker"),
        show_legend=on("show_legend"),
        show_schematic_moduli=st.session_state["show_schematic_moduli"],
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


@st.cache_data(show_spinner=False, max_entries=256)
def cached_thumbnail(root_folder, file_id):
    """Thumbnails are small and re-requested on every rerun, so cache them."""
    store = st.session_state.get("box_store")
    if store is None:
        return None
    return store.thumbnail_bytes(file_id)


@st.cache_data(show_spinner=False)
def cached_frame(path, signature, index):
    """`signature` (size + mtime) busts the cache when the file is replaced."""
    return va.read_frame(path, index)


@st.cache_data(show_spinner=False)
def cached_detection(
    path, signature, index, roi, sensitivity, strip_lines,
    enhance=None, cell_side="anywhere", reject_dark=True, find_nucleus=False,
    appearance="either",
):
    """Read one frame, enhance it, and find the probe, cell and nucleus."""
    frame = va.read_frame(path, index)
    if frame is None:
        return None, None, None, None
    if enhance:
        frame = va.enhance_frame(frame, **enhance)
    probe_box = None
    if reject_dark or cell_side != "anywhere":
        found = va.detect_probe(frame)
        probe_box = found if found.get("found") else None
    det = va.detect_cell(
        frame, roi=roi, sensitivity=sensitivity, strip_lines=strip_lines,
        probe=probe_box, cell_side=cell_side, reject_dark=reject_dark,
        appearance=appearance,
    )
    nucleus = va.detect_nucleus(frame, det) if (find_nucleus and det.get("found")) else None
    return frame, det, nucleus, probe_box


@st.cache_data(show_spinner="Tracking the cell through the video…")
def cached_track(path, signature, n_samples, roi, sensitivity, strip_lines, start, end,
                 enhance=None, cell_side="anywhere", reject_dark=True,
                 find_nucleus=False, appearance="either"):
    track = va.track_cell(
        path,
        n_samples=n_samples,
        roi=roi,
        sensitivity=sensitivity,
        start=start,
        end=end,
        enhance=enhance,
        cell_side=cell_side,
        reject_dark=reject_dark,
        track_nucleus=find_nucleus,
        appearance=appearance,
    )
    # Detections hold OpenCV contours; drop them so the cached value stays small.
    return {k: v for k, v in track.items() if k != "detections"}


def enhancement():
    """Frame adjustments as a hashable tuple-backed dict, or None if untouched."""
    settings = {
        "clahe_clip": float(st.session_state["video_clahe"]),
        "gamma": float(st.session_state["video_gamma"]),
        "brightness": int(st.session_state["video_brightness"]),
        "contrast": float(st.session_state["video_contrast"]),
    }
    untouched = (
        settings["clahe_clip"] == 0.0
        and settings["gamma"] == 1.0
        and settings["brightness"] == 0
        and settings["contrast"] == 1.0
    )
    return None if untouched else settings


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


def build_model(epsilon, force_N, active_windows=None) -> LulevichModel:
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
        active_windows=active_windows,
        segment_break_1=float(st.session_state["segment_break_1"]),
        segment_break_2=float(st.session_state["segment_break_2"]),
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
MODELS = {
    "Segmented (membrane → cytoskeleton → nucleus)":
        "Three stretches of deformation, each with different structures bearing "
        "the load. Continuous at the boundaries and linear in the moduli.",
    "Side by side (every element acts everywhere)":
        "All elements squashed by the same amount, their forces adding. The "
        "stiffest one dominates.",
    "Stacked (elements in line)":
        "All elements carrying the same force, their deformations adding. The "
        "softest one dominates.",
    "Side by side, then stacked":
        "Side by side up to a crossover deformation, stacked above it.",
    "Stacked, then side by side":
        "Stacked at small deformation, side by side once compressed.",
    "Compare these and rank them":
        "Fits the four above and ranks them by AICc and cross-validation.",
}
MODEL_KEYS = {
    "Segmented (membrane → cytoskeleton → nucleus)": "segmented",
    "Side by side (every element acts everywhere)": "parallel",
    "Stacked (elements in line)": "series",
    "Side by side, then stacked": "hybrid_ps",
    "Stacked, then side by side": "hybrid_sp",
    "Compare these and rank them": "auto",
}

# Older saved presets and stored database records name the model with the
# wording this app used before, including "parallel below, series above",
# which nobody could read. They are kept here only so that an old record still
# refits; nothing in the interface offers these names any more.
LEGACY_MODEL_NAMES = {
    "Parallel (forces add)": "parallel",
    "Series (deformations add)": "series",
    "Hybrid: parallel below, series above": "hybrid_ps",
    "Hybrid: series below, parallel above": "hybrid_sp",
    "Auto (let the data choose)": "auto",
}
# Any name the app has ever used, resolved to the key the fitter takes.
MODEL_KEYS_ANY = {**LEGACY_MODEL_NAMES, **MODEL_KEYS}
# The two things the physics does not settle for you, in plain words.
# Left of the arrow is what you pick in the interface, right of it is the
# argument the model takes.
MEMBRANE_CHOICES = {
    "holds what it reached": "freeze",
    "keeps stiffening": "continue",
}
CYTO_CHOICES = {
    "at ε₁": "break",
    "from the very start": "zero",
}
COMPOSITION_LABELS = {
    ("freeze", "break"): "Membrane alone, then it hands over to the cytoskeleton",
    ("freeze", "zero"): "Both from the start, membrane holds after ε₁",
    ("continue", "break"): "Membrane throughout, cytoskeleton joins at ε₁",
    ("continue", "zero"): "Membrane and cytoskeleton both throughout",
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
        # The membrane carries load from the very start of the compression, so
        # its window opens at zero rather than partway along.
        "windows": {"membrane": (0.0, 0.40), "interior": (0.0, 0.40),
                    "nucleus": (0.40, 1.0)},
    },
    "Cardiomyocyte": {
        "cell_height_um": 12.0,
        "radius_aspect": 0.60,
        "nucleus_fraction": 0.32,
        "membrane_thickness_nm": 4.5,
        "nucleus_onset": 0.18,
        "expected": {"Em": (5e5, 5e7), "Ei": (1e3, 1e5), "En": (2e3, 2e5)},
        "windows": {"membrane": (0.0, 0.35), "interior": (0.0, 0.35),
                    "nucleus": (0.35, 1.0)},
    },
    "Custom": {
        "cell_height_um": 8.09,
        "radius_aspect": 0.55,
        "nucleus_fraction": 0.35,
        "membrane_thickness_nm": 4.0,
        "nucleus_onset": 0.15,
        "expected": {"Em": (1e3, 1e9), "Ei": (1e0, 1e7), "En": (1e1, 1e7)},
        "windows": {"membrane": (0.0, 0.40), "interior": (0.0, 0.40),
                    "nucleus": (0.40, 1.0)},
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
    # A preset saved before the models were renamed still carries the old
    # wording; resolve it to a name the radio actually offers.
    stored_model = preset.get("coupling")
    if stored_model in MODEL_KEYS:
        pending["model_kind"] = stored_model
    elif stored_model in LEGACY_MODEL_NAMES:
        wanted = LEGACY_MODEL_NAMES[stored_model]
        for label, key in MODEL_KEYS.items():
            if key == wanted:
                pending["model_kind"] = label
                break
    for field, allowed in (
        ("membrane_after_break", MEMBRANE_CHOICES),
        ("cyto_starts_at", CYTO_CHOICES),
    ):
        if preset.get(field) in allowed:
            pending[field] = preset[field]
    for field in ("segment_break_1", "segment_break_2"):
        if preset.get(field) is not None:
            pending[field] = float(preset[field])
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
    # "window_end" is a single number, not a pair: the segmented range always
    # starts at zero, so a drag can only move where it ends.
    value = window[1] if key == "window_end" else window
    if st.session_state.get(key) != value:
        st.session_state["_pending_settings"] = {key: value}
        st.rerun()



# ======================================================== the Box database ==


def current_fit_settings():
    """Everything needed to reproduce the fit currently configured."""
    return {
        # "coupling" is the stored name of the model. It keeps its old key so
        # that records written by earlier versions still load.
        "coupling": st.session_state["model_kind"],
        "procedure": st.session_state["procedure"],
        "terms": list(active_terms()),
        "segment_break_1": float(st.session_state["segment_break_1"]),
        "segment_break_2": float(st.session_state["segment_break_2"]),
        "membrane_after_break": st.session_state["membrane_after_break"],
        "cyto_starts_at": st.session_state["cyto_starts_at"],
        "regime_mode": bool(st.session_state["regime_mode"]),
        "weighting": st.session_state["weighting"],
        "fit_offset": bool(st.session_state["fit_offset"]),
        "refine_iterations": int(st.session_state["refine_iterations"]),
        "nucleus_onset": float(st.session_state["nucleus_onset"]),
        "cell_type": st.session_state["cell_type"],
        "radius_aspect": float(st.session_state["radius_aspect"]),
        "nucleus_fraction": float(st.session_state["nucleus_fraction"]),
        "membrane_thickness_nm": float(st.session_state["membrane_thickness_nm"]),
        "poisson_membrane": float(st.session_state["poisson_membrane"]),
        "poisson_interior": float(st.session_state["poisson_interior"]),
        "term_windows": {
            term: list(st.session_state.get(f"window_term_{term}", (0.0, 1.0)))
            for term in TERM_ORDER
        },
        "combined_window": list(st.session_state.get("window_combined", (0.0, 1.0))),
    }


def morphology_frame_png():
    """A single frame showing the cell, for the database gallery."""
    if VIDEO_IMPORT_ERROR or not st.session_state.get("video_path"):
        return None
    path = st.session_state["video_path"]
    if not os.path.exists(path):
        return None
    try:
        index = int(st.session_state.get("video_preview_frame", 0))
        frame, det, nucleus, probe_box = cached_detection(
            path, video_signature(), index,
            st.session_state["video_roi"],
            float(st.session_state["video_sensitivity"]),
            bool(st.session_state["video_strip_lines"]),
            enhancement(),
            st.session_state["video_cell_side"],
            bool(st.session_state["video_reject_dark"]),
            bool(st.session_state["video_find_nucleus"]),
            st.session_state["video_appearance"],
        )
        if frame is None:
            return None
        annotated = va.annotate(frame, det, nucleus=nucleus, probe=probe_box)
        return png_bytes(va.crop(annotated, det) if det and det.get("found") else annotated)
    except Exception:
        return None


def cell_record(fit, epsilon, force_N, fitted, date_acquired, stage_plan):
    """
    The record and the curve, built once.

    Box and the download both go through here, so the file you keep on disk
    is the same record the database holds rather than a near-copy that drifts
    from it.
    """
    curve = pd.DataFrame(
        {
            "relative_deformation": epsilon,
            "force_N": force_N,
            "fit_N": fitted,
        }
    ).to_csv(index=False)

    record = {
        "cell_id": st.session_state["cell_name"],
        "date": str(date_acquired),
        "cell_type": st.session_state["cell_type"],
        "cell_height_um": float(st.session_state["cell_height_um"]),
        "spring_constant_N_per_m": float(st.session_state["spring_constant"]),
        "invols_nm_per_V": float(st.session_state["invols_nm_per_V"]),
        "operator": st.session_state["operator"],
        "notes": st.session_state["cell_notes"],
        "coupling": fit.get("coupling", "parallel"),
        "procedure": st.session_state["procedure"],
        "Em_MPa": float(fit.get("Em_MPa", 0.0)),
        "Ec_kPa": float(fit.get("Ei_kPa", 0.0)),
        "En_kPa": float(fit.get("En_kPa", 0.0)),
        "r_squared": float(fit.get("r_squared", float("nan"))),
        "rmse_N": float(fit.get("rmse", float("nan"))),
        "epsilon_min": float(fit["epsilon_range"][0]),
        "epsilon_max": float(fit["epsilon_range"][1]),
        "n_points": int(fit.get("n_points", 0)),
        "R0_um": float(fit.get("R0", 0.0)) * 1e6,
        "membrane_areal_modulus_mN_per_m": float(
            fit.get("membrane_areal_modulus", 0.0)
        ) * 1e3,
        "settings": current_fit_settings(),
        "stage_plan": [
            {"terms": list(stage["terms"]), "range": list(stage["range"])}
            for stage in stage_plan
        ],
        "warnings": list(fit.get("warnings", [])),
        "source_file": st.session_state["data"].get("source", ""),
        "video_url": st.session_state.get("video_link", ""),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }

    video_bytes, video_name = None, "video.mp4"
    if st.session_state.get("upload_video_with_cell") and st.session_state.get("video_path"):
        path = st.session_state["video_path"]
        if os.path.exists(path):
            with open(path, "rb") as handle:
                video_bytes = handle.read()
            video_name = os.path.basename(path) or "video.mp4"

    return record, curve, video_bytes, video_name


def send_cell_to_box(store, fit, epsilon, force_N, fitted, date_acquired, model, stage_plan):
    """Package this analysis and write it into the cell's Box folder."""
    record, curve, video_bytes, video_name = cell_record(
        fit, epsilon, force_N, fitted, date_acquired, stage_plan
    )
    return store.save_cell(
        record,
        curve_csv=curve,
        thumbnail_png=morphology_frame_png(),
        video_bytes=video_bytes,
        video_name=video_name,
    )


def cell_bundle_zip(fit, epsilon, force_N, fitted, date_acquired, stage_plan):
    """The same record as Box, as a zip the browser can download."""
    import zipfile

    record, curve, video_bytes, video_name = cell_record(
        fit, epsilon, force_N, fitted, date_acquired, stage_plan
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("record.json", json.dumps(record, indent=2, default=str))
        archive.writestr("curve.csv", curve)
        frame = morphology_frame_png()
        if frame:
            archive.writestr("morphology.png", frame)
        if video_bytes:
            archive.writestr(video_name, video_bytes)
    return buffer.getvalue()


def model_from_record(record, curve):
    """Rebuild a model from a stored record, using that cell's own geometry."""
    settings = record.get("settings") or {}
    height_m = float(record.get("cell_height_um", 8.0)) * 1e-6
    radius_m = height_m * float(settings.get("radius_aspect", 0.55))
    windows = settings.get("term_windows") if settings.get("regime_mode") else None
    return LulevichModel(
        curve["force_N"].to_numpy(dtype=float),
        curve["relative_deformation"].to_numpy(dtype=float),
        cell_height=height_m,
        cell_radius=radius_m,
        membrane_thickness=float(settings.get("membrane_thickness_nm", 4.0)) * 1e-9,
        poisson_membrane=float(settings.get("poisson_membrane", 0.5)),
        poisson_interior=float(settings.get("poisson_interior", 0.5)),
        nucleus_radius=radius_m * float(settings.get("nucleus_fraction", 0.35)),
        nucleus_onset=float(settings.get("nucleus_onset", 0.2)),
        active_windows={k: tuple(v) for k, v in (windows or {}).items()} or None,
    )


def refit_stored_cell(store, cell_id, settings):
    """
    Refit one stored cell with a given set of settings and save the result.

    The cell's own geometry is kept: height, spring constant and the rest were
    measured for that cell and are not something a batch should overwrite. Only
    the fitting choices come from the settings passed in.
    """
    record = store.load_cell(cell_id)
    if not record or not record.get("curve_csv"):
        raise BoxError(f"{cell_id}: no stored curve to refit.")
    curve = pd.read_csv(io.StringIO(record["curve_csv"]))

    merged = dict(record.get("settings") or {})
    merged.update(settings)
    record["settings"] = merged
    model = model_from_record(record, curve)

    terms = tuple(settings.get("terms") or ("membrane", "interior"))
    lo, hi = settings.get("combined_window", [0.02, 0.4])
    coupling = MODEL_KEYS_ANY.get(settings.get("coupling", ""), "segmented")

    if coupling == "segmented":
        # The default model, so bulk refits have to handle it. The composition
        # comes from the record when it has one and from the current defaults
        # when it was saved before compositions existed.
        membrane = MEMBRANE_CHOICES.get(
            settings.get("membrane_after_break", ""), "freeze"
        )
        cyto_start = CYTO_CHOICES.get(settings.get("cyto_starts_at", ""), "break")
        fit = model.fit_composition(
            lo, hi,
            e1=float(settings.get("segment_break_1", 0.15)),
            e2=float(settings.get("segment_break_2", 0.40)),
            membrane=membrane,
            cyto_start=cyto_start,
            use_membrane="membrane" in terms,
            use_interior="interior" in terms,
            use_nucleus="nucleus" in terms,
            weighting=settings.get("weighting", "uniform"),
        )
    elif coupling == "series":
        fit = model.fit_series(lo, hi, terms=terms, weighting=settings.get("weighting", "uniform"))
    elif coupling in ("hybrid_ps", "hybrid_sp"):
        order = "parallel-then-series" if coupling == "hybrid_ps" else "series-then-parallel"
        scan = model.scan_crossover(lo, hi, terms=terms, order=order)
        if not scan.get("success"):
            raise BoxError(f"{cell_id}: hybrid fit failed.")
        fit = scan["best"]
    elif coupling == "auto":
        comparison = compare_couplings(model, lo, hi, terms=terms)
        if not comparison.get("success"):
            raise BoxError(f"{cell_id}: {comparison.get('error')}")
        fit = comparison["fits"][comparison["best"]["coupling"]]
    elif settings.get("procedure") == "Stage by stage":
        windows = settings.get("term_windows", {})
        plan = [
            {"terms": (term,), "range": tuple(windows.get(term, (lo, hi)))}
            for term in terms
        ]
        fit = model.fit_staged(
            plan, weighting=settings.get("weighting", "uniform"),
            refine_iterations=int(settings.get("refine_iterations", 3)),
        )
    else:
        fit = model.fit(lo, hi, terms=terms,
                        weighting=settings.get("weighting", "uniform"),
                        fit_offset=bool(settings.get("fit_offset", False)))

    if not fit.get("success"):
        raise BoxError(f"{cell_id}: {fit.get('error', 'fit failed')}")

    record.update(
        {
            "coupling": fit.get("coupling", "parallel"),
            "procedure": settings.get("procedure", record.get("procedure", "")),
            "Em_MPa": float(fit.get("Em_MPa", 0.0)),
            "Ec_kPa": float(fit.get("Ei_kPa", 0.0)),
            "En_kPa": float(fit.get("En_kPa", 0.0)),
            "r_squared": float(fit.get("r_squared", float("nan"))),
            "epsilon_min": float(fit["epsilon_range"][0]),
            "epsilon_max": float(fit["epsilon_range"][1]),
            "n_points": int(fit.get("n_points", 0)),
            "refit_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    record.pop("curve_csv", None)
    store.save_cell(record)
    return record



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

    with st.expander("📈 Fitting", expanded=False):
        hint(
            "The model, the deformation ranges and the fitting options are all in "
            "the main panel, under **Model** and **Deformation ranges**."
        )

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
        st.markdown("**What is drawn on the curve**")
        bare = st.checkbox(
            "Data and fit only", key="bare_plot",
            help="Strips everything except the measured points and the one "
            "fitted Lulevich curve: no bands, no markers, no element curves, "
            "no legend. This is the figure version.",
        )
        st.caption(
            "Everything below is off while “Data and fit only” is on."
            if bare else "Or switch off individual pieces:"
        )
        st.checkbox(
            "Element curves", key="show_components", disabled=bare,
            help="The membrane, cytoskeleton and nucleus contributions drawn "
            "apart. They are parts of the one fit, not separate fits.",
        )
        st.checkbox(
            "Shaded segment bands", key="show_fit_window", disabled=bare,
            help="The coloured blocks behind the curve marking each segment. "
            "This also hides the highlighted segment.",
        )
        st.checkbox(
            "Video frame marker", key="show_video_marker", disabled=bare,
            help="The orange ring and dotted line showing where on the curve "
            "the displayed video frame sits.",
        )
        st.checkbox(
            "Rupture marker", key="show_rupture_marker", disabled=bare,
            help="The dash-dotted line where the force drops.",
        )
        st.checkbox("Legend", key="show_legend", disabled=bare)

        st.markdown("**Panels beside the curve**")
        st.checkbox(
            "Cell diagram",
            key="show_schematic",
            help="A side-on sketch of the membrane, cytoskeleton and nucleus at "
            "the deformation you select.",
        )
        st.checkbox(
            "Moduli under the diagram", key="show_schematic_moduli",
            help="The Eₘ, E_c and Eₙ values printed under the cell diagram. The "
            "same numbers are in the results table.",
        )
        st.checkbox(
            "Video frame",
            key="video_show_panel",
            help="Only appears once a video is loaded in the Compression video tab.",
        )

    with st.expander("📦 Box database", expanded=False):
        if BOX_IMPORT_ERROR:
            st.error(f"Box module unavailable: {BOX_IMPORT_ERROR}")
        else:
            st.text_input(
                "Root folder ID",
                key="box_root_folder",
                placeholder="0 for All Files, or the id from the folder URL",
                help="Open the folder in Box; the id is the number at the end of "
                "the URL. Everything is stored under it as cells/<cell name>/.",
            )
            if st.button("🔌 Connect to Box", **STRETCH):
                try:
                    store = box_store.store_from_secrets(
                        st, st.session_state["box_root_folder"] or None
                    )
                    if store is None:
                        st.error(
                            "No [box] section in secrets. See the setup notes below."
                        )
                    else:
                        info = store.check()
                        st.session_state["box_store"] = store
                        st.session_state["box_index"] = None
                        st.success(
                            f"Connected as {info['login']} · folder "
                            f"“{info['folder_name']}” · {info['auth_method']}"
                        )
                except Exception as exc:
                    st.session_state["box_store"] = None
                    st.error(str(exc))
            if st.session_state.get("box_store"):
                st.caption("Connected ✓")
            with st.expander("Setting up Box"):
                st.markdown(
                    "Create an app at **developer.box.com**, then put its "
                    "credentials in `.streamlit/secrets.toml`:\n\n"
                    "```toml\n[box]\nclient_id = \"...\"\n"
                    "client_secret = \"...\"\nenterprise_id = \"...\"\n"
                    "root_folder_id = \"0\"\n```\n\n"
                    "**Client Credentials Grant** is the one to ask for: its tokens "
                    "renew themselves, so nothing expires mid-session. On a UC Davis "
                    "account a Box admin has to authorise the app once, from Admin "
                    "Console → Enterprise Settings → Platform Apps.\n\n"
                    "To try it before that approval comes through, generate a "
                    "**developer token** in the console and use "
                    "`developer_token = \"...\"` instead. It works for 60 minutes.\n\n"
                    "With client credentials the app acts as its own service user, "
                    "not as you, so share the target folder with that user or point "
                    "`root_folder_id` at a folder it owns."
                )

    with st.expander("🗄️ Google Sheets (optional mirror)"):
        if SHEETS_IMPORT_ERROR:
            st.info("Database module unavailable in this environment.")
            st.caption(SHEETS_IMPORT_ERROR)
        else:
            st.checkbox("Enable database", key="db_enabled")
            if st.session_state["db_enabled"]:
                st.text_input(
                    "Spreadsheet ID or URL",
                    key="sheet_id",
                    placeholder="1AbC…  or the full /spreadsheets/d/… link",
                    help="The sheet must already exist in YOUR Drive and be shared "
                    "with the service account as an Editor. A service account has "
                    "no Drive storage of its own, so it cannot create one.",
                )
                if st.button("🔗 Connect", **STRETCH):
                    manager = initialize_sheets_manager(
                        spreadsheet_id=st.session_state["sheet_id"] or None
                    )
                    st.session_state["gs_manager"] = manager
                    if manager:
                        st.success("Connected.")
                if st.session_state["gs_manager"]:
                    st.caption("Connected ✓")
                    url = st.session_state["gs_manager"].get_spreadsheet_url()
                    if url:
                        st.caption(f"[Open the sheet]({url})")
                with st.expander("Setting this up"):
                    st.markdown(
                        "1. In Google Cloud, create a service account and download "
                        "its JSON key. Enable the Sheets and Drive APIs.\n"
                        "2. Put the key in `.streamlit/secrets.toml` under "
                        "`[google_sheets_credentials]`.\n"
                        "3. Create a blank Sheet in your own Drive, then share it "
                        "with the service account's `client_email` as an **Editor**. "
                        "This step is the one that is usually missed, and skipping "
                        "it produces a 403 about Drive storage quota, which sounds "
                        "like a different problem than it is.\n"
                        "4. Paste the sheet id above, or set it in secrets:\n\n"
                        "```toml\n[google_sheets]\nspreadsheet_id = \"...\"\n```"
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
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.text_input("Cell name / ID", placeholder="C2C12_001", key="cell_name")
    with c2:
        date_acquired = st.date_input(
            "Date acquired", value=datetime.now().date(), key="date_acquired"
        )
    with c3:
        st.number_input(
            "Spring constant k (N/m)",
            min_value=0.0, max_value=100.0, step=0.001, format="%.4f",
            key="spring_constant",
            help="Cantilever stiffness. Recorded with the cell; the force curve "
            "you upload is assumed to already be in force units.",
        )
    with c4:
        st.number_input(
            "InvOLS (nm/V)",
            min_value=0.0, max_value=1000.0, step=0.1, format="%.2f",
            key="invols_nm_per_V",
            help="Deflection sensitivity from the calibration ramp. Recorded so "
            "the numbers can be traced back to the calibration they came from.",
        )
    c5, c6 = st.columns([1, 3])
    with c5:
        st.text_input("Operator", placeholder="initials", key="operator")
    with c6:
        st.text_input("Notes", placeholder="passage, treatment, anything worth keeping",
                      key="cell_notes")
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
                    index=guess_column(
                        columns, ("reldef", "rel def", "rel_def", "deform", "eps", "ε",
                                  "strain"), 0,
                    ),
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
                    index=0,
                    key="input_force_unit",
                    help="Files exported by this app are in newtons.",
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
            )        # ----------------------------------------------------------- model ---
        section("3 · Model")

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

        element_col, model_col = st.columns([1, 1.5])
        with model_col:
            st.markdown("**2 · How they share the load**")
            st.radio(
                "How the cell is modelled",
                list(MODELS.keys()),
                key="model_kind",
                label_visibility="collapsed",
                help="Segmented treats the compression as three stretches with "
                "different structures bearing the load. The others assume every "
                "element acts across the whole curve, which is what makes them "
                "fail on a curve that changes character partway along.",
            )
            st.caption(MODELS[st.session_state["model_kind"]])
        with element_col:
            st.markdown("**1 · Which components**")
            st.checkbox("Membrane · Eₘ", key="use_membrane")
            st.checkbox("Cytoskeleton · Ec", key="use_interior")
            st.checkbox("Nucleus · Eₙ", key="use_nucleus")

        active = active_terms()
        kind = MODEL_KEYS[st.session_state["model_kind"]]
        segmented = kind == "segmented"
        coupling = kind

        if segmented:
            st.markdown("**3 · What each one does at the first boundary**")
            c1, c2 = st.columns(2)
            with c1:
                st.radio(
                    "After ε₁ the membrane…",
                    list(MEMBRANE_CHOICES.keys()),
                    key="membrane_after_break",
                    help="“Holds what it reached” means the membrane stops adding "
                    "force above ε₁ and keeps the force it had there, so the whole "
                    "of the extra load goes to the cytoskeleton. “Keeps stiffening” "
                    "means the ε³ term carries on rising underneath the others.",
                )
            with c2:
                st.radio(
                    "The cytoskeleton starts…",
                    list(CYTO_CHOICES.keys()),
                    key="cyto_starts_at",
                    help="“At ε₁” means the cytoskeleton only begins to bear load "
                    "once the membrane hands over. “From the very start” means both "
                    "carry load together from first contact.",
                )
            membrane_mode = MEMBRANE_CHOICES[st.session_state["membrane_after_break"]]
            cyto_mode = CYTO_CHOICES[st.session_state["cyto_starts_at"]]
            st.caption(
                f"→ {COMPOSITION_LABELS[(membrane_mode, cyto_mode)]}. "
                "The nucleus always joins at ε₂. If you are not sure which of the "
                "four is right, the combination search below tries them all."
            )
        else:
            membrane_mode, cyto_mode = "freeze", "break"

        # ---------------------------------------------------- exploration ---
        if segmented:
            section("4 · Explore the curve")
            e1_col, e2_col = st.columns([1, 2])
            with e1_col:
                if st.button("🔬 Find the segments", type="secondary", **STRETCH):
                    with st.spinner("Measuring the exponent along the curve…"):
                        st.session_state["exploration"] = model.explore_segments(
                            terms=active or ("membrane", "interior", "nucleus")
                        )
                st.caption(
                    "Scans for the two boundaries, then measures the power law each "
                    "stage actually follows: 3 for the membrane, 3/2 for a Hertzian "
                    "contact."
                )
            exploration = st.session_state.get("exploration")
            with e2_col:
                if exploration and exploration.get("success"):
                    if exploration["confident"]:
                        st.success(
                            f"ε₁ = {exploration['break_1']:.3f}, "
                            f"ε₂ = {exploration['break_2']:.3f}. Each stage follows "
                            f"the law the model assigns it."
                        )
                    else:
                        st.warning(
                            f"ε₁ = {exploration['break_1']:.3f}, "
                            f"ε₂ = {exploration['break_2']:.3f}, but the evidence is "
                            f"weak. See the notes below."
                        )
                    if st.button("✓ Use these breakpoints", type="primary", **STRETCH):
                        st.session_state["_pending_settings"] = {
                            "segment_break_1": round(float(exploration["break_1"]), 3),
                            "segment_break_2": round(float(exploration["break_2"]), 3),
                        }
                        st.rerun()
                elif exploration:
                    st.error(exploration.get("error", "Exploration failed."))

            if exploration and exploration.get("success"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "stage": row["stage"],
                                "ε from": round(row["range"][0], 3),
                                "ε to": round(row["range"][1], 3),
                                "points": row["n_points"],
                                "exponent measured": (
                                    round(row["measured_exponent"], 2)
                                    if np.isfinite(row["measured_exponent"])
                                    else None
                                ),
                                "expected": row["expected_exponent"],
                                "power-law R²": (
                                    round(row["power_law_r2"], 3)
                                    if np.isfinite(row["power_law_r2"]) else None
                                ),
                                "modulus": row["modulus_label"],
                            }
                            for row in exploration["stages"]
                        ]
                    ),
                    hide_index=True,
                    **STRETCH,
                )
                st.caption(
                    "A blank exponent means that stage does not rise far enough above "
                    "the noise for its power law to be measured. The breakpoint still "
                    "comes from the fit, but nothing independently confirms the shape."
                )
                for note in exploration["notes"]:
                    st.warning(note)
                st.plotly_chart(
                    exponent_profile_figure(
                        exploration["profile"], current_style(force_N),
                        exploration["break_1"], exploration["break_2"],
                    ),
                    key="exponent_profile",
                    **STRETCH,
                )

        # --------------------------------------------------------- ranges ---
        section("5 · Deformation ranges" if segmented else "4 · Deformation ranges")
        st.caption(
            f"Data spans ε = {eps_lo_data:.3f} to {eps_hi_data:.3f}. "
            f"Auto-detected usable region: {auto_window[0]:.3f} to "
            f"{auto_window[1]:.3f} ({auto_range['n_points']} points) · "
            f"rupture: {auto_range['rupture_method']}"
        )

        if segmented:
            # The segmented model starts at first contact by definition: the
            # membrane term is ε³ measured from ε = 0, so a range that starts
            # anywhere else is fitting a curve the model does not describe.
            # Only the far end is yours to choose.
            fit_lo = 0.0
            st.session_state["window_end"] = float(
                np.clip(
                    st.session_state.get("window_end", eps_hi_data),
                    max(step, 0.0), eps_hi_data,
                )
            )
            fit_hi = st.slider(
                "Fit up to ε =",
                min_value=float(step), max_value=eps_hi_data, step=step,
                key="window_end",
                help="The range always starts at zero. This sets where it ends, "
                "and the fitted curve is drawn only over that range.",
            )
            st.session_state["window_combined"] = (fit_lo, fit_hi)
        else:
            st.session_state["window_combined"] = clamp_range(
                st.session_state.get("window_combined"), auto_window,
            )
            fit_lo, fit_hi = st.slider(
                "Fitted range",
                min_value=eps_lo_data, max_value=eps_hi_data, step=step,
                key="window_combined",
                help="The stretch of the curve the fit is measured on.",
            )
        st.caption(f"{int(((epsilon >= fit_lo) & (epsilon <= fit_hi)).sum())} points")

        term_windows = {}
        stage_plan = [{"terms": active, "range": (fit_lo, fit_hi)}]
        staged = False
        break_1 = float(st.session_state["segment_break_1"])
        break_2 = float(st.session_state["segment_break_2"])

        highlight_window = None

        if segmented:
            st.markdown("**Segment table**")
            st.caption(
                "Type the boundaries directly. The segments are contiguous, so the "
                "end of one is the start of the next: editing a row's ε end moves "
                "that boundary."
            )
            # The row names follow the composition chosen above, so the table
            # always says what is actually carrying load in each stretch.
            seg_1_name = (
                "1 · membrane + cytoskeleton" if cyto_mode == "zero"
                else "1 · membrane, ε³"
            )
            seg_2_name = (
                "2 · membrane + cytoskeleton" if membrane_mode == "continue"
                else "2 · cytoskeleton, membrane holding"
            )
            seg_3_name = (
                "3 · " + ("membrane + " if membrane_mode == "continue" else "")
                + "cytoskeleton + nucleus"
            )
            table = pd.DataFrame(
                [
                    {
                        "segment": seg_1_name,
                        "ε start": round(fit_lo, 3),
                        "ε end": round(break_1, 3),
                        "points": int(((epsilon >= fit_lo) & (epsilon <= break_1)).sum()),
                    },
                    {
                        "segment": seg_2_name,
                        "ε start": round(break_1, 3),
                        "ε end": round(break_2, 3),
                        "points": int(((epsilon > break_1) & (epsilon <= break_2)).sum()),
                    },
                    {
                        "segment": seg_3_name,
                        "ε start": round(break_2, 3),
                        "ε end": round(fit_hi, 3),
                        "points": int(((epsilon > break_2) & (epsilon <= fit_hi)).sum()),
                    },
                ]
            )
            edited = st.data_editor(
                table,
                hide_index=True,
                key="segment_table",
                disabled=["segment", "ε start", "points"],
                column_config={
                    "ε end": st.column_config.NumberColumn(
                        "ε end", min_value=0.0, max_value=1.0, step=0.005, format="%.3f",
                    )
                },
                **STRETCH,
            )
            try:
                new_1 = float(edited.loc[0, "ε end"])
                new_2 = float(edited.loc[1, "ε end"])
            except Exception:
                new_1, new_2 = break_1, break_2
            if (
                abs(new_1 - break_1) > 1e-6 or abs(new_2 - break_2) > 1e-6
            ) and 0 <= new_1 < new_2 <= 1:
                st.session_state["_pending_settings"] = {
                    "segment_break_1": round(new_1, 4),
                    "segment_break_2": round(new_2, 4),
                }
                st.rerun()

            # Which segment to shade on the curve. Editing a boundary is much
            # easier when you can see the stretch of data it moves.
            st.radio(
                "Highlight on the plot",
                ["(none)", "Segment 1", "Segment 2", "Segment 3", "Whole fitted range"],
                key="highlight_segment",
                horizontal=True,
            )
            highlight_bounds = {
                "Segment 1": (fit_lo, break_1, seg_1_name),
                "Segment 2": (break_1, break_2, seg_2_name),
                "Segment 3": (break_2, fit_hi, seg_3_name),
                "Whole fitted range": (fit_lo, fit_hi, "fitted range"),
            }
            highlight_window = highlight_bounds.get(
                st.session_state["highlight_segment"]
            )

            b1, b2 = st.columns([1, 1])
            with b1:
                if st.button("🔎 Find the boundaries from the data", **STRETCH):
                    with st.spinner("Scanning boundaries…"):
                        scan_breaks = model.scan_segment_breaks(
                            fit_lo, fit_hi, terms=active or ("membrane", "interior"),
                            weighting=st.session_state["weighting"],
                        )
                    if scan_breaks.get("success"):
                        st.session_state["_pending_settings"] = {
                            "segment_break_1": round(float(scan_breaks["best_break_1"]), 3),
                            "segment_break_2": round(float(scan_breaks["best_break_2"]), 3),
                        }
                        st.rerun()
                    else:
                        st.error(scan_breaks.get("error", "Boundary scan failed."))
                st.caption(
                    "Moves the two boundaries only, keeping the combination you "
                    "picked above."
                )
            # Applying a winning combination means writing four widget keys,
            # which Streamlit only allows before those widgets exist. So both
            # the search button and the table's apply button stage the values
            # and rerun; the fit then happens with them already in place.
            apply_labels = {
                (MEMBRANE_CHOICES[m], CYTO_CHOICES[c]): (m, c)
                for m in MEMBRANE_CHOICES for c in CYTO_CHOICES
            }

            def stage_combination(row):
                m_label, c_label = apply_labels[(row["membrane"], row["cyto_start"])]
                return {
                    "segment_break_1": round(float(row["break_1"]), 3),
                    "segment_break_2": round(float(row["break_2"]), 3),
                    "membrane_after_break": m_label,
                    "cyto_starts_at": c_label,
                    "use_nucleus": bool(row["use_nucleus"]),
                }

            with b2:
                can_search = hasattr(model, "search_compositions")
                if st.button(
                    "🧩 Find the best combination and fit it", type="primary",
                    disabled=not can_search, **STRETCH,
                ) and can_search:
                    with st.spinner(
                        "Fitting all four combinations at their own best "
                        "boundaries and cross-validating each…"
                    ):
                        found = model.search_compositions(
                            fit_lo, fit_hi,
                            weighting=st.session_state["weighting"],
                        )
                    st.session_state["composition_search"] = found
                    if found.get("success"):
                        # Go straight to the answer: apply the winner and let
                        # the fit below run with it, so one press gives one
                        # fitted line rather than a table to act on.
                        st.session_state["_pending_settings"] = stage_combination(
                            found["best"]
                        )
                        st.rerun()
                st.caption(
                    "Searches all four ways the membrane and cytoskeleton can share "
                    "the first boundary, each with its own best ε₁ and ε₂, ranks "
                    "them on data they were not fitted to, and applies the winner."
                    if can_search else
                    "Needs an up to date `lulevich_model.py`."
                )

            search = st.session_state.get("composition_search")
            if search and search.get("success"):
                st.info(search["verdict"])
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "combination": row["label"],
                                "ε₁": round(row["break_1"], 3),
                                "ε₂": round(row["break_2"], 3),
                                "Eₘ (MPa)": float(f"{row['Em_MPa']:.4g}"),
                                "E_c (kPa)": float(f"{row['Ec_kPa']:.4g}"),
                                "Eₙ (kPa)": float(f"{row['En_kPa']:.4g}"),
                                "R²": round(row["r_squared"], 5),
                                "CV RMSE": f"{row['cv_rmse']:.4g}",
                                "ΔAICc": round(row["delta_aicc"], 1),
                                "note": " · ".join(
                                    part for part in (
                                        "picked" if row is search["best"] else "",
                                        "ties with the pick"
                                        if row.get("tied_with_best") else "",
                                        ", ".join(row["empty_terms"]) + " came out zero"
                                        if row.get("empty_terms") else "",
                                        ", ".join(row.get("idle_breaks", []))
                                        + " unused here"
                                        if row.get("idle_breaks") else "",
                                    ) if part
                                ),
                            }
                            for row in search["candidates"]
                        ]
                    ),
                    hide_index=True,
                    **STRETCH,
                )
                st.caption(
                    "Ranked by cross-validated error, which asks how well each "
                    "combination predicts points it was not fitted on, averaged "
                    "over several different fold splits. Candidates closer than "
                    "the amount that number moves between splits are called tied, "
                    "and the pick among tied candidates is the one with the fewest "
                    "free moduli. ΔAICc is shown but does not decide the order: on "
                    "these curves it is confident about differences the held-out "
                    "error says are not there."
                )
                best = search["best"]
                choice_names = [row["label"] for row in search["candidates"]]
                a1, a2 = st.columns([2, 1])
                with a1:
                    picked = st.selectbox(
                        "Override the pick", choice_names,
                        index=choice_names.index(best["label"])
                        if best["label"] in choice_names else 0,
                        key="composition_pick",
                        help="The winner is already applied. Use this only to try "
                        "one of the others.",
                    )
                with a2:
                    st.markdown("<div style='height:1.7rem'></div>",
                                unsafe_allow_html=True)
                    if st.button("✓ Use this one instead", **STRETCH):
                        row = next(
                            r for r in search["candidates"] if r["label"] == picked
                        )
                        st.session_state["_pending_settings"] = stage_combination(row)
                        st.rerun()
                st.caption(
                    f"Applied: ε₁ = {best['break_1']:.3f}, ε₂ = "
                    f"{best['break_2']:.3f}, "
                    f"{'with' if best['use_nucleus'] else 'without'} the nucleus."
                )
            elif search:
                st.error(search.get("error", "The combination search failed."))

            if break_2 <= break_1:
                st.error("Segment 2 must end after segment 1.")
            else:
                st.caption(
                    "The force is continuous across both boundaries by "
                    "construction, so moving one never puts a step in the curve."
                )

        elif coupling in ("hybrid_ps", "hybrid_sp"):
            h1, h2 = st.columns([1, 2])
            with h1:
                st.radio("Crossover ε", ["Scan for best", "Set manually"],
                         key="crossover_mode", horizontal=True)
            with h2:
                st.slider("ε at which the load path changes", 0.0, 1.0, step=0.01,
                          key="crossover",
                          disabled=st.session_state["crossover_mode"] == "Scan for best")

        elif coupling == "parallel":
            st.radio(
                "Fitting procedure", ["All at once", "Stage by stage"],
                key="procedure", horizontal=True,
                help="Stage by stage measures each element on its own window, "
                "which helps when the moduli come out correlated.",
            )
            staged = st.session_state["procedure"] == "Stage by stage"
            if staged:
                st.caption("A window per element; same stage number = fitted together.")
                window_cols = st.columns(max(1, len(active))) if active else [st]
                for i, term in enumerate(active):
                    key = f"window_term_{term}"
                    st.session_state[key] = clamp_range(
                        st.session_state.get(key),
                        default_window_for((term,), auto_window, eps_lo_data, eps_hi_data),
                    )
                    with window_cols[i % len(window_cols)]:
                        lo, hi = st.slider(
                            TERM_LABELS[term], min_value=eps_lo_data,
                            max_value=eps_hi_data, step=step, key=key,
                        )
                        term_windows[term] = (lo, hi)
                        st.selectbox(f"Stage for {TERM_LABELS[term]}", [1, 2, 3],
                                     key=f"stage_of_{term}", label_visibility="collapsed")
                stage_plan = []
                for stage_no, terms in stage_groups(active):
                    spans = [term_windows[t] for t in terms if t in term_windows]
                    if spans:
                        stage_plan.append(
                            {"terms": terms,
                             "range": (min(s[0] for s in spans), max(s[1] for s in spans))}
                        )
                if stage_plan:
                    fit_lo = min(s["range"][0] for s in stage_plan)
                    fit_hi = max(s["range"][1] for s in stage_plan)

        r1, r2 = st.columns([1, 3])
        with r1:
            if st.button("↺ Reset ranges", **STRETCH):
                st.session_state["_pending_clear_windows"] = True
                st.rerun()
        with r2:
            range_label = "End of the range" if segmented else "Fitted range"
            targets = ["(off)", range_label] + [
                TERM_LABELS[t] for t in (active if staged else [])
            ]
            st.session_state["_drag_keys"] = {
                range_label: "window_end" if segmented else "window_combined"
            }
            st.session_state["_drag_keys"].update(
                {TERM_LABELS[t]: f"window_term_{t}" for t in (active if staged else [])}
            )
            st.selectbox("Drag on the plot to set", targets, key="drag_target")

        with st.expander("⚙️ Advanced fitting options"):
            a1, a2 = st.columns(2)
            with a1:
                st.selectbox(
                    "Weighting", ["uniform", "relative"], key="weighting",
                    help="uniform minimises absolute residuals, so the high-force "
                    "end dominates. relative weights each point by 1/|F| so the "
                    "small-ε region counts too.",
                )
                st.checkbox("Fit a constant force offset", key="fit_offset")
            with a2:
                st.slider("Refinement passes (staged fits)", 1, 8,
                          key="refine_iterations")
                st.checkbox("Seed staged fits from all-at-once", key="seed_parallel")
            st.checkbox("Refit live as settings change", key="live_fit")

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
                            # Stored under its old key so presets saved by
                            # earlier versions still load.
                            "coupling": st.session_state["model_kind"],
                            "procedure": st.session_state["procedure"],
                            "segment_break_1": float(st.session_state["segment_break_1"]),
                            "segment_break_2": float(st.session_state["segment_break_2"]),
                            "membrane_after_break": st.session_state["membrane_after_break"],
                            "cyto_starts_at": st.session_state["cyto_starts_at"],
                            "combined_window": [float(fit_lo), float(fit_hi)],
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
                                "model": pre.get("coupling", "?"),
                                "ε₁": pre.get("segment_break_1"),
                                "ε₂": pre.get("segment_break_2"),
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

        # ----------------------------------------------------------- fit ---        # ------------------------------------------------------------- fit ---
        section("6 · Fit" if segmented else "5 · Fit")

        scan = None
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
                model.nucleus_onset = scan["best_onset"]
                st.session_state["_scanned_onset"] = float(scan["best_onset"])
                if not scan["well_determined"]:
                    st.warning(
                        "The nucleus onset scan is flat: every ε₀ fits about equally "
                        "well, so this curve does not locate the nucleus."
                    )

        run = st.session_state["live_fit"] or st.button("🚀 Fit curve", type="primary")

        # A fit has to survive a rerun. Uploading a video, ticking a checkbox
        # or opening a tab all rerun the script, and with live refitting off
        # nothing recomputes the fit, so it used to vanish and take the whole
        # database section with it. Keep the last good one and reuse it.
        fit_signature = repr(
            (
                data.get("source"),
                int(epsilon.size),
                float(force_N[0]) if force_N.size else 0.0,
                float(force_N[-1]) if force_N.size else 0.0,
                sorted(current_fit_settings().items(), key=lambda kv: kv[0]),
                round(fit_lo, 6), round(fit_hi, 6),
                round(break_1, 6), round(break_2, 6),
                kind,
            )
        )

        fit = None
        comparison = None
        if run and not active:
            st.warning("Select at least one element above.")
        elif run and segmented and break_2 <= break_1:
            st.error("Set ε₂ above ε₁ before fitting.")
        elif run:
            if segmented:
                # fit_composition covers what fit_segmented did and adds the
                # two choices about the first boundary, so it is the one path.
                # An older lulevich_model.py has only the fixed version.
                if hasattr(model, "fit_composition"):
                    fit = model.fit_composition(
                        fit_lo, fit_hi, e1=break_1, e2=break_2,
                        membrane=membrane_mode,
                        cyto_start=cyto_mode,
                        use_membrane="membrane" in active,
                        use_interior="interior" in active,
                        use_nucleus="nucleus" in active,
                        weighting=st.session_state["weighting"],
                    )
                else:
                    fit = model.fit_segmented(
                        fit_lo, fit_hi, e1=break_1, e2=break_2, terms=active,
                        weighting=st.session_state["weighting"],
                        fit_offset=st.session_state["fit_offset"],
                    )
            elif coupling == "auto":
                with st.spinner("Fitting every model and comparing…"):
                    comparison = compare_couplings(model, fit_lo, fit_hi, terms=active)
                if comparison.get("success"):
                    fit = comparison["fits"][comparison["best"]["coupling"]]
                else:
                    st.error(comparison.get("error", "Could not compare models."))
            elif coupling == "series":
                fit = model.fit_series(
                    fit_lo, fit_hi, terms=active,
                    weighting=st.session_state["weighting"],
                )
            elif coupling in ("hybrid_ps", "hybrid_sp"):
                order = ("parallel-then-series" if coupling == "hybrid_ps"
                         else "series-then-parallel")
                if st.session_state["crossover_mode"] == "Scan for best":
                    scan_x = model.scan_crossover(fit_lo, fit_hi, terms=active, order=order)
                    if scan_x.get("success"):
                        fit = scan_x["best"]
                        st.caption(f"Best crossover ε = {scan_x['best_crossover']:.3f}")
                    else:
                        st.error(scan_x.get("error", "Hybrid scan failed."))
                else:
                    crossover = float(np.clip(st.session_state["crossover"],
                                              fit_lo + 1e-4, fit_hi - 1e-4))
                    fit = model.fit_hybrid(fit_lo, fit_hi, crossover, terms=active,
                                           order=order)
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

        # The video and database section below runs whether or not there is a
        # fit, so the names it reads have to exist either way.
        fitted = membrane = interior = nucleus = None

        stale_fit = False
        if fit is not None and fit.get("success"):
            st.session_state["_last_fit"] = fit
            st.session_state["_last_fit_signature"] = fit_signature
        elif fit is None and st.session_state.get("_last_fit") is not None:
            # Nothing asked for a fit on this run, so show the last good one
            # rather than an empty page.
            fit = st.session_state["_last_fit"]
            stale_fit = (
                st.session_state.get("_last_fit_signature") != fit_signature
            )
            if stale_fit:
                st.info(
                    "Showing the previous fit. Something has changed since it "
                    "was made, so press **Fit curve** to bring it up to date, "
                    "or switch on live refitting in Advanced fitting options."
                )

        if comparison and comparison.get("success"):
            st.info(comparison["verdict"])
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "model": row["label"],
                            "R²": round(row["r_squared"], 5),
                            "ΔAICc": round(row["delta_aicc"], 1),
                            "weight": round(row["weight"], 3),
                            "CV RMSE": f"{row['cv_rmse']:.3g}",
                            "params": row["n_params"],
                        }
                        for row in comparison["candidates"]
                    ]
                ),
                hide_index=True,
                **STRETCH,
            )
            st.caption(
                "ΔAICc under 2 means the curve cannot tell those models apart. A "
                "wrong model can still reach R² > 0.99 with badly wrong moduli, "
                "which is why this table exists."
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

            if fitted_coupling == "segmented":
                # Draw the components with the same basis the fit used, or the
                # curves would not add up to the line through the data.
                if hasattr(model, "composition_terms"):
                    membrane_basis, cyto_basis, nucleus_basis = model.composition_terms(
                        epsilon, fit["break_1"], fit["break_2"],
                        fit.get("membrane", "freeze"), fit.get("cyto_start", "break"),
                    )
                else:
                    membrane_basis, cyto_basis, nucleus_basis = model.segment_terms(
                        epsilon, fit["break_1"], fit["break_2"]
                    )
                fitted = (
                    membrane_basis * params[0]
                    + cyto_basis * params[1]
                    + nucleus_basis * params[2]
                    + fit.get("force_offset", 0.0)
                )
                membrane = membrane_basis * params[0]
                interior = cyto_basis * params[1]
                nucleus = nucleus_basis * params[2] if params[2] else None
            elif fitted_coupling == "parallel":
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

            # The model is only claimed over the range it was fitted on.
            # Drawn past that it is extrapolation, and a power law far outside
            # its window flattens into a line that looks like a result. NaN
            # outside the window makes plotly stop the line at the last
            # fitted point instead.
            # Use the range the fit actually recorded, not the sliders, so a
            # slider moved after the fit cannot punch NaNs into the residuals.
            drawn_lo, drawn_hi = fit.get("epsilon_range", (fit_lo, fit_hi))

            def clip_to_window(values):
                if values is None:
                    return None
                out = np.array(values, dtype=float, copy=True)
                out[(epsilon < drawn_lo) | (epsilon > drawn_hi)] = np.nan
                return out

            fitted = clip_to_window(fitted)
            membrane = clip_to_window(membrane)
            interior = clip_to_window(interior)
            nucleus = clip_to_window(nucleus)

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

            if fitted_coupling == "segmented":
                windows_for_plot = [
                    {"range": (fit_lo, fit["break_1"]), "label": "membrane",
                     "color": "#2ca02c"},
                    {"range": (fit["break_1"], fit["break_2"]), "label": "cytoskeleton",
                     "color": "#9467bd"},
                    {"range": (fit["break_2"], fit_hi), "label": "cytoskeleton + nucleus",
                     "color": "#e377c2"},
                ]
            else:
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

            # All three moduli, always. A term that was not in the model reads
            # 0 and says so, rather than disappearing: a blank column in a
            # results table is ambiguous between "zero" and "not measured",
            # and the two mean very different things when you pool cells.
            fitted_terms = set(fit.get("terms") or active)
            metric_cols = st.columns(5)
            for slot, (label, value, unit, term) in enumerate(
                (
                    ("Eₘ membrane", fit.get("Em_MPa", 0.0), "MPa", "membrane"),
                    ("Ec cytoskeleton", fit.get("Ei_kPa", 0.0), "kPa", "interior"),
                    ("Eₙ nucleus", fit.get("En_kPa", 0.0), "kPa", "nucleus"),
                )
            ):
                used = term in fitted_terms
                std = fit.get(
                    {"membrane": "Em_MPa_std", "interior": "Ei_kPa_std",
                     "nucleus": "En_kPa_std"}[term], float("nan")
                )
                if not used:
                    note = "not in this model"
                elif value <= 0:
                    note = "came out at zero"
                elif np.isfinite(std):
                    note = f"± {std:.2g}"
                elif term == "nucleus":
                    note = f"engages at ε = {fit.get('break_2', model.nucleus_onset):.3f}"
                else:
                    note = " "
                metric_cols[slot].metric(
                    label,
                    f"{0.0 if not used else value:.3g} {unit}",
                    delta=note,
                    delta_color="off",
                )
            metric_cols[3].metric("R²", f"{fit['r_squared']:.4f}")
            rmse_disp, rmse_unit = from_newtons(fit["rmse"], style.force_unit)
            metric_cols[4].metric("RMSE", f"{float(rmse_disp):.3g} {rmse_unit}")

            st.caption(
                f"Membrane areal modulus Eₘ·h = "
                f"{fit.get('membrane_areal_modulus', 0.0) * 1e3:.4g} mN/m, which is what "
                f"the ε³ term actually determines. Eₘ itself is that divided by the "
                f"assumed bilayer thickness of "
                f"{st.session_state['membrane_thickness_nm']:.1f} nm, so halving the "
                f"thickness doubles Eₘ while the measurement is unchanged."
            )

            if fitted_coupling == "segmented" and fitted is not None:
                # What each element is doing in each stretch, and how much of
                # the force it carries there. This is the question the moduli
                # alone do not answer: a modulus says how stiff, not how much
                # of the load that stiffness actually took.
                st.markdown("**How the three share the load, range by range**")
                e1, e2 = fit["break_1"], fit["break_2"]
                membrane_mode_fit = fit.get("membrane", "freeze")
                cyto_mode_fit = fit.get("cyto_start", "break")

                def state_words(term, lo, hi):
                    if term == "membrane":
                        if hi <= e1 or membrane_mode_fit == "continue":
                            return "loading"
                        return "holding"
                    if term == "interior":
                        if cyto_mode_fit == "zero" or lo >= e1:
                            return "loading"
                        return "not yet"
                    return "loading" if lo >= e2 else "not yet"

                rows = []
                for name, lo, hi in (
                    ("1", fit["epsilon_range"][0], e1),
                    ("2", e1, e2),
                    ("3", e2, fit["epsilon_range"][1]),
                ):
                    inside = (epsilon >= lo) & (epsilon <= hi)
                    if not inside.any() or hi <= lo:
                        continue
                    share = {}
                    for term, curve in (
                        ("membrane", membrane), ("interior", interior),
                        ("nucleus", nucleus),
                    ):
                        if curve is None:
                            share[term] = 0.0
                            continue
                        at_top = float(np.nan_to_num(curve[inside][-1]))
                        share[term] = max(at_top, 0.0)
                    total = sum(share.values()) or 1.0
                    row = {
                        "range": f"{lo:.3f} to {hi:.3f}",
                        "points": int(inside.sum()),
                    }
                    for term, label in (
                        ("membrane", "membrane"), ("interior", "cytoskeleton"),
                        ("nucleus", "nucleus"),
                    ):
                        word = state_words(term, lo, hi)
                        if term not in fitted_terms:
                            row[label] = "not in model"
                        elif word == "not yet":
                            row[label] = "0 %  not yet"
                        else:
                            row[label] = f"{100 * share[term] / total:.0f} %  {word}"
                    rows.append(row)

                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
                    st.caption(
                        "Percentages are each element's share of the total force at "
                        "the top of that range, so they show who is carrying the "
                        "cell by the end of each stretch. “Holding” means the "
                        "element adds no further force but keeps what it had "
                        "reached, so its share falls while its force does not. A "
                        "term switched off in the model reads “not in model” rather "
                        "than zero, because those are different statements."
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
                        **figure_kwargs(
                            force_curve_figure,
                            title=st.session_state["cell_name"]
                            or "Force vs relative deformation",
                            fit_force_N=fitted,
                            membrane_N=membrane,
                            interior_N=interior,
                            nucleus_N=nucleus,
                            fit_window=windows_for_plot,
                            rupture_epsilon=rupture.get("epsilon")
                            if rupture.get("method") == "force-drop"
                            else None,
                            highlight=highlight
                            if (video_ready or show_schematic) else None,
                            highlight_window=highlight_window,
                        ),
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
                            **figure_kwargs(
                                cell_schematic,
                                coupling=(
                                    "series" if fitted_coupling == "series"
                                    else "hybrid" if fitted_coupling == "hybrid"
                                    else "parallel"
                                ),
                                shares=deformation_shares,
                                epsilon=selected_eps,
                                cell_height_um=st.session_state["cell_height_um"],
                                cell_radius_um=fit["R0"] * 1e6,
                                nucleus_radius_um=fit.get(
                                    "R_nucleus", fit["R0"] * 0.35
                                ) * 1e6,
                                membrane_thickness_nm=st.session_state[
                                    "membrane_thickness_nm"
                                ],
                                nucleus_onset=model.nucleus_onset
                                if "nucleus" in active else None,
                                break_1=fit.get("break_1"),
                                break_2=fit.get("break_2"),
                                membrane_mode=fit.get("membrane", "freeze"),
                                cyto_start=fit.get("cyto_start", "break"),
                                Em_MPa=fit["Em_MPa"],
                                Ei_kPa=fit["Ei_kPa"],
                                En_kPa=fit.get("En_kPa") if "nucleus" in active else None,
                                show_nucleus="nucleus" in active,
                            ),
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
                    vframe, vdet, vnuc, vprobe = cached_detection(
                        st.session_state["video_path"],
                        video_signature(),
                        int(frame_index),
                        st.session_state["video_roi"],
                        float(st.session_state["video_sensitivity"]),
                        bool(st.session_state["video_strip_lines"]),
                        enhancement(),
                        st.session_state["video_cell_side"],
                        bool(st.session_state["video_reject_dark"]),
                        bool(st.session_state["video_find_nucleus"]),
                        st.session_state["video_appearance"],
                    )
                    if vframe is None:
                        st.info("Frame unavailable.")
                    else:
                        force_here, unit_here = from_newtons(highlight[1], style.force_unit)
                        snap = va.annotate(
                            vframe, vdet, label=f"ε = {highlight[0]:.3f}",
                            nucleus=vnuc, probe=vprobe,
                        )
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

        section("7 · Video and database" if segmented else "6 · Video and database")
        store = st.session_state.get("box_store")
        ready = store is not None

        v1, v2 = st.columns([2, 1])
        with v1:
            main_video = st.file_uploader(
                "Compression video for this cell",
                type=["mp4", "avi", "mov", "wmv", "mkv"],
                key="video_file_main",
                help="Uploaded here it is stored with the cell. The "
                "**Compression video** tab has the detection controls if you "
                "want the cell outlined and measured.",
            )
            if (
                main_video is not None
                and st.session_state.get("video_name") != main_video.name
                and not VIDEO_IMPORT_ERROR
            ):
                destination = os.path.join(
                    tempfile.gettempdir(), f"afm_video_{main_video.name}"
                )
                with open(destination, "wb") as handle:
                    handle.write(main_video.getvalue())
                st.session_state["video_path"] = destination
                st.session_state["video_name"] = main_video.name
                st.session_state["video_track"] = None
                try:
                    st.session_state["video_info"] = va.probe(destination)
                except Exception as exc:
                    st.session_state["video_info"] = None
                    st.error(f"Could not open that video: {exc}")
        with v2:
            info = st.session_state.get("video_info")
            if info:
                st.metric("Video frames", f"{info['n_frames']:,}")
                st.caption(
                    f"{st.session_state.get('video_name', '')} · "
                    f"{info['width']}×{info['height']}"
                )
                thumb = morphology_frame_png()
                if thumb:
                    st.image(thumb, caption="frame stored with the cell", width=190)
            else:
                st.caption("No video loaded for this cell.")

        have_fit = bool(fit and fit.get("success"))
        named = bool(st.session_state["cell_name"].strip())

        # Say what is blocking the send, in the order you would fix it. A
        # greyed-out button with no explanation is the same as a broken one.
        blockers = []
        if not named:
            blockers.append("give the cell a name in section 1")
        if not have_fit:
            blockers.append("fit the curve above")
        if not ready:
            blockers.append("connect to Box in the sidebar")

        st.checkbox(
            "Also upload the video file itself",
            key="upload_video_with_cell",
            help="Off sends the cell without the video; the record, curve "
            "and fit go either way. On also copies the video into the "
            "cell's Box folder so it can be opened from the database.",
        )

        will_send = ["the record, the curve and the fit"]
        if st.session_state.get("video_path") and os.path.exists(
            st.session_state["video_path"]
        ):
            if st.session_state["upload_video_with_cell"]:
                will_send.append("the video file")
            if not VIDEO_IMPORT_ERROR:
                will_send.append("a morphology frame")
        st.caption(
            "Either destination sends " + ", ".join(will_send)
            + ". The video is optional: a cell can be sent with none at all."
        )

        c1, c2 = st.columns(2)
        with c1:
            box_blockers = [b for b in blockers]
            if st.button(
                "📤 Send to Box", type="primary",
                disabled=bool(box_blockers), **STRETCH,
            ):
                try:
                    with st.spinner("Uploading to Box…"):
                        saved = send_cell_to_box(
                            store, fit, epsilon, force_N, fitted,
                            date_acquired, model, stage_plan,
                        )
                    st.success(
                        f"Saved **{saved['cell_id']}** to Box."
                        + (f" [Open the video]({saved['video_url']})"
                           if saved.get("video_url") else "")
                    )
                    st.session_state["box_index"] = None
                except Exception as exc:
                    st.error(f"Could not save: {exc}")
            st.caption(
                "Ready to send." if not box_blockers
                else "To enable: " + ", then ".join(box_blockers) + "."
            )
        with c2:
            # Box needs credentials that take a while to obtain. This does
            # not, and it writes exactly the same record, so the analysis is
            # never stuck inside the app waiting on an administrator.
            download_blockers = [b for b in blockers if "Box" not in b]
            if download_blockers:
                st.button("⬇️ Download this cell", disabled=True, **STRETCH)
            else:
                st.download_button(
                    "⬇️ Download this cell",
                    data=cell_bundle_zip(
                        fit, epsilon, force_N, fitted, date_acquired, stage_plan
                    ),
                    file_name=f"{st.session_state['cell_name'].strip()}.zip",
                    mime="application/zip",
                    **STRETCH,
                )
            st.caption(
                "A zip with record.json, curve.csv and the frame. Same record "
                "as Box, no account needed."
                if not download_blockers
                else "To enable: " + ", then ".join(download_blockers) + "."
            )


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
                "…or a Google Drive / Box link",
                placeholder="https://app.box.com/s/…  or  https://drive.google.com/file/d/…",
                key="video_link",
                help="Box links need either public sharing or an access token in "
                "secrets under [box]. Drive links must be shared with anyone "
                "who has the link.",
            )
            if st.button("⬇️ Fetch from link", **STRETCH):
                try:
                    token = None
                    try:
                        token = st.secrets.get("box", {}).get("access_token")
                    except Exception:
                        token = None
                    with st.spinner("Downloading…"):
                        dest = os.path.join(tempfile.gettempdir(), "afm_linked_video.mp4")
                        va.fetch_video(st.session_state["video_link"], dest, box_token=token)
                    st.session_state["video_path"] = dest
                    st.session_state["video_name"] = "linked video"
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
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("**Image**")
                st.slider(
                    "Local contrast (CLAHE)", 0.0, 6.0, step=0.5, key="video_clahe",
                    help="Brings out a faint cell without blowing out the bright "
                    "field. 0 leaves the frame alone.",
                )
                st.slider("Gamma", 0.3, 3.0, step=0.05, key="video_gamma",
                          help="Above 1 lifts the dark end, where a cell in the "
                          "probe's shadow lives.")
                st.slider("Brightness", -80, 80, step=2, key="video_brightness")
                st.slider("Contrast", 0.4, 3.0, step=0.05, key="video_contrast")
            with d2:
                st.markdown("**Finding the cell**")
                st.slider(
                    "Edge sensitivity", 0.3, 3.0, step=0.1, key="video_sensitivity",
                )
                st.checkbox(
                    "Ignore long horizontal structures", key="video_strip_lines",
                    help="Removes the substrate line and the cantilever body.",
                )
                st.checkbox(
                    "Ignore very dark objects (the probe)", key="video_reject_dark",
                    help="The cantilever is close to black. This paints it out "
                    "before segmenting, so it stops being picked as the cell.",
                )
                st.selectbox(
                    "The cell looks …",
                    ["clear", "dark", "either"],
                    key="video_appearance",
                    help="… compared with the background. In phase contrast "
                    "the cell is usually the clear, bright object, and saying so "
                    "stops a dark patch of debris of similar shape being picked "
                    "instead. Choose “either” if the cell matches the background "
                    "and shows only as an outline.",
                )
                st.selectbox(
                    "The cell sits …",
                    ["anywhere", "right", "left", "above", "below"],
                    key="video_cell_side",
                    help="… relative to the probe. Set this and the probe is "
                    "located first, then only that side is searched.",
                )
                st.checkbox("Also find the nucleus", key="video_find_nucleus")
            with d3:
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
            frame, det, nucleus, probe_box = cached_detection(
                path,
                signature,
                int(preview_frame),
                roi,
                float(st.session_state["video_sensitivity"]),
                bool(st.session_state["video_strip_lines"]),
                enhancement(),
                st.session_state["video_cell_side"],
                bool(st.session_state["video_reject_dark"]),
                bool(st.session_state["video_find_nucleus"]),
                st.session_state["video_appearance"],
            )

            p1, p2 = st.columns([2, 1])
            with p1:
                if frame is None:
                    st.error("Could not read that frame.")
                else:
                    label = (
                        f"h = {det['height_px']:.0f} px"
                        if det and det.get("found")
                        else "cell not found"
                    )
                    st.image(
                        va.annotate(frame, det, label=label, nucleus=nucleus,
                                    probe=probe_box),
                        caption=f"Frame {preview_frame} · red = cell, purple = "
                        f"nucleus, grey = probe",
                        **STRETCH,
                    )
            with p2:
                if det and det.get("found"):
                    st.success("Cell detected")
                    st.metric("Cell height", f"{det['height_px']:.0f} px")
                    st.metric("Cell width", f"{det['width_px']:.0f} px")
                    st.caption(
                        f"circularity {det['circularity']:.2f} · "
                        f"solidity {det['solidity']:.2f}"
                    )
                    if det.get("rejected_dark") or det.get("rejected_side"):
                        st.caption(
                            f"rejected {det.get('rejected_dark', 0)} dark and "
                            f"{det.get('rejected_side', 0)} wrong-side candidates"
                        )
                    if nucleus and nucleus.get("found"):
                        st.metric("Nucleus height", f"{nucleus['height_px']:.0f} px")
                        st.caption(
                            f"{100 * nucleus['area_fraction_of_cell']:.0f} % of the "
                            f"cell box · circularity {nucleus['circularity']:.2f}"
                        )
                    elif st.session_state["video_find_nucleus"]:
                        st.caption(f"Nucleus: {nucleus.get('reason', 'not found')}")
                    st.download_button(
                        "📷 Save this frame",
                        data=png_bytes(
                            va.annotate(frame, det, label=label, nucleus=nucleus,
                                        probe=probe_box)
                        ),
                        file_name=f"frame_{preview_frame}.png",
                        mime="image/png",
                        **STRETCH,
                    )
                elif det:
                    st.warning(f"No cell found: {det.get('reason', 'unknown')}")
                    st.caption(
                        "Try raising the local contrast, narrowing the search "
                        "region, or setting which side of the probe the cell is on."
                    )
                if probe_box and probe_box.get("found"):
                    st.caption(f"Probe found at x = {probe_box['bbox'][0]}")

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
                        enhancement(),
                        st.session_state["video_cell_side"],
                        bool(st.session_state["video_reject_dark"]),
                        bool(st.session_state["video_find_nucleus"]),
                        st.session_state["video_appearance"],
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
    section("Cell database")
    store = st.session_state.get("box_store")

    if BOX_IMPORT_ERROR:
        st.error(f"Box module unavailable: {BOX_IMPORT_ERROR}")
    elif store is None:
        st.info(
            "Not connected. Open **Box database** in the sidebar and connect, then "
            "every cell you send appears here."
        )
    else:
        head1, head2, head3 = st.columns([1, 1, 2])
        with head1:
            if st.button("🔄 Refresh", **STRETCH):
                st.session_state["box_index"] = None
        with head2:
            if st.button("🛠️ Rebuild index", **STRETCH,
                         help="Reads every cell's record.json and writes a fresh "
                              "index. Slower, but it recovers a stale or damaged one."):
                progress = st.progress(0.0, text="Reading cells…")
                try:
                    st.session_state["box_index"] = store.rebuild_index(
                        progress=lambda n, total, name: progress.progress(
                            n / max(total, 1), text=f"{n}/{total}  {name}"
                        )
                    )
                    progress.empty()
                    st.success("Index rebuilt.")
                except Exception as exc:
                    progress.empty()
                    st.error(str(exc))

        if st.session_state.get("box_index") is None:
            try:
                with st.spinner("Loading the index…"):
                    st.session_state["box_index"] = store.load_index()
            except Exception as exc:
                st.error(str(exc))
                st.session_state["box_index"] = pd.DataFrame()

        index = st.session_state.get("box_index")
        if index is None or index.empty:
            st.info("No cells stored yet. Fit a curve and press **Send to database**.")
        else:
            with head3:
                search = st.text_input(
                    "Filter", placeholder="cell name, operator, date or cell type",
                    key="db_search", label_visibility="collapsed",
                )
            shown = index
            if search:
                needle = search.lower()
                mask = index.apply(
                    lambda row: needle in " ".join(str(v).lower() for v in row.values),
                    axis=1,
                )
                shown = index[mask]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cells", len(index))
            m2.metric("Shown", len(shown))
            for column, label, target in (
                ("Em_MPa", "Median Eₘ", m3), ("Ec_kPa", "Median Ec", m4)
            ):
                values = pd.to_numeric(shown.get(column), errors="coerce").dropna()
                values = values[values > 0]
                unit = "MPa" if column == "Em_MPa" else "kPa"
                target.metric(
                    label, f"{values.median():.3g} {unit}" if len(values) else "n/a"
                )

            view = st.radio(
                "View", ["Gallery", "Table"], horizontal=True, key="db_view",
                label_visibility="collapsed",
            )

            if view == "Gallery":
                st.caption(
                    "Tick the cells you want to work on, then use the batch tools "
                    "below the gallery."
                )
                selected = []
                per_row = 4
                rows = list(shown.itertuples(index=False))
                for start in range(0, len(rows), per_row):
                    columns = st.columns(per_row)
                    for column, row in zip(columns, rows[start : start + per_row]):
                        record = row._asdict()
                        cell_id = str(record.get("cell_id", ""))
                        with column:
                            thumb_id = record.get("thumbnail_file_id")
                            image = None
                            if thumb_id and str(thumb_id) not in ("", "nan"):
                                image = cached_thumbnail(
                                    st.session_state["box_root_folder"], str(thumb_id)
                                )
                            if image:
                                st.image(image, **STRETCH)
                            else:
                                st.markdown(
                                    "<div style='height:110px;border:1px dashed #c7d0d8;"
                                    "border-radius:6px;display:flex;align-items:center;"
                                    "justify-content:center;color:#9aa7b2;font-size:0.8rem'>"
                                    "no frame</div>",
                                    unsafe_allow_html=True,
                                )
                            st.markdown(f"**{cell_id}**")
                            em = pd.to_numeric(record.get("Em_MPa"), errors="coerce")
                            ec = pd.to_numeric(record.get("Ec_kPa"), errors="coerce")
                            r2 = pd.to_numeric(record.get("r_squared"), errors="coerce")
                            st.caption(
                                f"Eₘ {em:.3g} MPa · Ec {ec:.3g} kPa"
                                if pd.notna(em) and pd.notna(ec)
                                else "not fitted"
                            )
                            st.caption(
                                f"R² {r2:.4f} · {record.get('date', '')}"
                                if pd.notna(r2) else str(record.get("date", ""))
                            )
                            url = str(record.get("video_url") or "")
                            if url and url != "nan":
                                st.markdown(f"[▶ video]({url})")
                            if st.checkbox("select", key=f"db_pick_{cell_id}"):
                                selected.append(cell_id)
                        # A little breathing room between rows.
                    st.markdown(
                        "<div style='height:0.6rem'></div>", unsafe_allow_html=True
                    )
            else:
                numeric = ["Em_MPa", "Ec_kPa", "En_kPa", "r_squared",
                           "epsilon_min", "epsilon_max", "cell_height_um"]
                table = shown.copy()
                for column in numeric:
                    if column in table.columns:
                        table[column] = pd.to_numeric(table[column], errors="coerce")
                st.dataframe(
                    table.round(
                        {c: 4 for c in numeric if c in table.columns}
                    ),
                    hide_index=True,
                    **STRETCH,
                )
                picks = st.multiselect(
                    "Select cells for the batch tools",
                    list(shown["cell_id"].astype(str)),
                    key="db_table_picks",
                )
                selected = list(picks)

            st.divider()
            section("Batch tools")
            if not selected:
                st.caption("Select one or more cells above.")
            else:
                st.caption(f"{len(selected)} selected: " + ", ".join(selected[:8])
                           + (" …" if len(selected) > 8 else ""))

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("🔁 Refit selected with current settings",
                             disabled=not selected, **STRETCH):
                    settings = current_fit_settings()
                    progress = st.progress(0.0, text="Refitting…")
                    done, failed = [], []
                    for n, cell_id in enumerate(selected, start=1):
                        progress.progress(n / len(selected), text=f"{cell_id} ({n}/{len(selected)})")
                        try:
                            done.append(refit_stored_cell(store, cell_id, settings))
                        except Exception as exc:
                            failed.append(f"{cell_id}: {exc}")
                    progress.empty()
                    st.session_state["box_index"] = None
                    if done:
                        st.success(f"Refitted {len(done)} cell(s).")
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "cell": r["cell_id"],
                                        "Eₘ (MPa)": round(r["Em_MPa"], 4),
                                        "Ec (kPa)": round(r["Ec_kPa"], 4),
                                        "Eₙ (kPa)": round(r["En_kPa"], 4),
                                        "R²": round(r["r_squared"], 4),
                                    }
                                    for r in done
                                ]
                            ),
                            hide_index=True,
                            **STRETCH,
                        )
                    for message in failed:
                        st.warning(message)
                    st.caption(
                        "Each cell keeps its own geometry: height, spring constant "
                        "and radius were measured for that cell and a batch does not "
                        "overwrite them. Only the fitting choices are applied."
                    )
            with b2:
                one = selected[0] if selected else None
                if st.button(f"✏️ Re-open {one}" if one else "✏️ Re-open selected",
                             disabled=len(selected) != 1, **STRETCH,
                             help="Loads the stored curve back into the analysis tab "
                                  "so you can change its windows and send it again."):
                    try:
                        record = store.load_cell(one)
                        if not record or not record.get("curve_csv"):
                            st.error(f"{one} has no stored curve.")
                        else:
                            curve = pd.read_csv(io.StringIO(record["curve_csv"]))
                            pending = {
                                "cell_name": record.get("cell_id", one),
                                "cell_height_um": float(record.get("cell_height_um", 8.0)),
                                "spring_constant": float(
                                    record.get("spring_constant_N_per_m", 0.0)
                                ),
                                "invols_nm_per_V": float(
                                    record.get("invols_nm_per_V", 50.0)
                                ),
                                "operator": record.get("operator", ""),
                                "cell_notes": record.get("notes", ""),
                            }
                            settings = record.get("settings") or {}
                            for key in ("coupling", "procedure", "weighting",
                                        "cell_type", "nucleus_onset", "regime_mode"):
                                if settings.get(key) is not None:
                                    pending[key] = settings[key]
                            for term, window in (settings.get("term_windows") or {}).items():
                                pending[f"window_term_{term}"] = tuple(window)
                            if settings.get("combined_window"):
                                pending["window_combined"] = tuple(settings["combined_window"])
                            for term in TERM_ORDER:
                                pending[f"use_{term}"] = term in (settings.get("terms") or [])
                            pending["_applied_cell_type"] = settings.get("cell_type")
                            st.session_state["_pending_settings"] = pending
                            st.session_state["data"] = {
                                "epsilon": curve["relative_deformation"].to_numpy(float),
                                "force_N": curve["force_N"].to_numpy(float),
                                "source": f"Box · {one}",
                                "n_dropped": 0,
                            }
                            st.success(
                                f"Loaded {one}. Open the **Force curve analysis** tab, "
                                f"adjust it, then send it again to overwrite the record."
                            )
                            st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            with b3:
                st.download_button(
                    "📥 Export the whole index (CSV)",
                    data=index.to_csv(index=False),
                    file_name=f"afm_cell_database_{datetime.now():%Y%m%d}.csv",
                    mime="text/csv",
                    **STRETCH,
                )


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
