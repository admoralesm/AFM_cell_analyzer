"""
AFM Cell Analyzer v6

Force-curve analysis for single-cell compression using the Lulevich two-term
model. Settings live in one place, the fit is live, and every result comes
with the diagnostics needed to tell a real measurement from a bad window.
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------ companion imports ---
#
# Nothing here is imported with a bare `from x import y`, and that is not
# style. app.py, lulevich_model.py and plot_utils.py are updated together,
# and a deploy that picks up one but not the others used to die on the
# import line itself, before a single line of the app had run. Streamlit
# Cloud shows that as a truncated traceback with the reason cut out of it,
# so the one thing the person needed to know, which file is behind and
# which piece of it is missing, was exactly what they could not see.
#
# So each name is fetched by hand, whatever is missing is remembered, and
# the app either carries on without an optional piece or stops with a
# message that names the file and the missing names.

MISSING_PIECES = []


def _companion(name):
    """Import a companion module, or remember that it could not be."""
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - only on a broken deploy
        MISSING_PIECES.append((f"{name}.py", [f"the file itself ({exc})"]))
        return None


_model_module = _companion("lulevich_model")
_plot_module = _companion("plot_utils")


def _pull(module, filename, names, required=True):
    """Bind ``names`` from ``module`` into this one, remembering any gaps."""
    if module is None:
        for name in names:
            globals()[name] = None
        return False
    missing = []
    for name in names:
        value = getattr(module, name, None)
        if value is None and not hasattr(module, name):
            missing.append(name)
        globals()[name] = value
    if missing and required:
        MISSING_PIECES.append((filename, missing))
    return not missing


_pull(_model_module, "lulevich_model.py", (
    "LulevichModel",
    "compare_couplings",
    "compare_hypotheses",
    "dropped_term_warning",
    "recommend_components",
    "search_arrangements",
))
_pull(_plot_module, "plot_utils.py", (
    "FORCE_UNITS",
    "INPUT_FORCE_UNITS",
    "PlotStyle",
    "autoscale_unit",
    "balloon_figure",
    "cell_schematic",
    "exponent_profile_figure",
    "force_curve_figure",
    "from_newtons",
    "residual_figure",
    "sensitivity_figure",
    "to_newtons",
))

# Optional: newer pieces the app can run without, each switched off by the
# companion file check below rather than taking the whole app down.
HAS_ORDERINGS = _pull(_model_module, "lulevich_model.py",
                      ("ORDERINGS", "compare_orderings", "ordering_of"),
                      required=False)
if not HAS_ORDERINGS:
    ORDERINGS, compare_orderings = [], None

    def ordering_of(*_args, **_kwargs):
        return None

HAS_ORDER_PLOTS = _pull(_plot_module, "plot_utils.py",
                        ("ordering_figure", "ordering_slope_figure"),
                        required=False)

# The per-element window search. Optional, so an older lulevich_model.py
# leaves the bars usable by hand and only the button missing.
HAS_WINDOW_SEARCH = _pull(_model_module, "lulevich_model.py",
                          ("search_term_windows",), required=False)
if not HAS_WINDOW_SEARCH:
    search_term_windows = None

# The four-regime, C0-anchored C2C12 fit. Optional in the same way: without
# piecewise_fit.py the C2C12 page falls back to the other models rather than
# refusing to start.
PIECEWISE_PROBLEM = ""


def _load_piecewise():
    """Import piecewise_fit, or a copy saved under a near-miss name.

    Files added through the GitHub web page or a download folder often
    arrive as "Piecewise fit.py" or "piecewise_fit (1).py", which Python
    cannot import by name. Such a copy is loaded from its path instead, and
    the reason is kept so the page can say exactly what to rename.
    """
    global PIECEWISE_PROBLEM
    try:
        return importlib.import_module("piecewise_fit")
    except ModuleNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - a broken copy
        PIECEWISE_PROBLEM = f"piecewise_fit.py failed to import: {exc}"
        return None
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        names = sorted(os.listdir(here))
    except OSError:
        names = []
    for name in names:
        stem, ext = os.path.splitext(name)
        key = re.sub(r"[^a-z]", "", stem.lower())
        if ext.lower() != ".py" or not key.startswith("piecewisefit"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                "piecewise_fit", os.path.join(here, name))
            module = importlib.util.module_from_spec(spec)
            sys.modules["piecewise_fit"] = module
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - a broken copy
            sys.modules.pop("piecewise_fit", None)
            PIECEWISE_PROBLEM = f"'{name}' failed to import: {exc}"
            return None
        PIECEWISE_PROBLEM = (
            f"The engine is saved as '{name}'. It was loaded anyway, but "
            "rename it to piecewise_fit.py so every tool can import it.")
        return module
    PIECEWISE_PROBLEM = ("piecewise_fit.py was not found in the folder "
                         "that holds app.py.")
    return None


_piecewise_module = _load_piecewise()
# Newer than the rest: without it the plot draws each component on its own
# rather than stacked, and nothing else changes.
component_force = getattr(_piecewise_module, "component_force", None)
HAS_PIECEWISE = _pull(_piecewise_module, "piecewise_fit.py", (
    "C2C12_BOUNDARIES_PCT",
    "C2C12_REGIMES",
    "Geometry",
    "SEARCH_BANDS_PCT",
    "component_curve",
    "component_ranges",
    "lamina_summary",
    "find_boundaries",
    "boundaries_from_power_law",
    "power_law_profile",
    "fit_piecewise",
    "piecewise_moduli",
    "predict_piecewise",
    "probe_correction",
    "regime_curves",
), required=False)

# The components the page fits: the spring network's four (membrane,
# cytoskeleton, nuclear envelope, inside the nucleus) plus the contact
# line, which can be switched off. The engine's specification also has a
# perinuclear cytoskeleton and a lamina lump; they are left out here so
# that every way of sharing the load fits the same four components.
PW_REMOVED = ("K_nuc_cyto", "A_lamina")
if HAS_PIECEWISE:
    import dataclasses as _dataclasses
    PW_REGIMES = tuple(
        _dataclasses.replace(
            regime,
            terms=tuple(t for t in regime.terms if t.name not in PW_REMOVED),
            title=("Nuclear envelope stretch" if regime.key == "R3"
                   else "Contact artefact (baseline C₀)" if regime.key == "R1"
                   else regime.title),
            equation=("F₃(x) = K_nucleus·(x−ε₂)³ + F(ε₂)" if regime.key == "R3"
                      else "F₁(x) = C₀" if regime.key == "R1"
                      else regime.equation),
        )
        for regime in C2C12_REGIMES
    )
else:
    PW_REGIMES = ()

if _piecewise_module is not None and not HAS_PIECEWISE:
    _stale = [name for name in (
        "C2C12_BOUNDARIES_PCT", "C2C12_REGIMES", "Geometry",
        "SEARCH_BANDS_PCT", "component_curve", "component_ranges",
        "lamina_summary", "find_boundaries", "boundaries_from_power_law",
        "power_law_profile", "fit_piecewise", "piecewise_moduli",
        "predict_piecewise", "probe_correction", "regime_curves",
    ) if not hasattr(_piecewise_module, name)]
    PIECEWISE_PROBLEM = (
        (PIECEWISE_PROBLEM + " ") if PIECEWISE_PROBLEM.startswith("The engine")
        else "") + (
        "piecewise_fit.py is older than app.py; it lacks "
        + ", ".join(_stale) + ". Replace it with the version shipped "
        "with this app.py.")


def piecewise_problem_note():
    """Say why the 4-regime fit is unavailable (or loaded from a stray name)."""
    if PIECEWISE_PROBLEM:
        (st.warning if HAS_PIECEWISE else st.error)(
            "4-regime engine: " + PIECEWISE_PROBLEM)

if MISSING_PIECES:
    st.set_page_config(page_title="AFM Cell Analyzer", layout="wide")
    st.error(
        "**This deploy is half updated, so the app cannot start.**\n\n"
        + "\n".join(
            f"`{name}` is missing " + ", ".join(f"`{p}`" for p in pieces)
            for name, pieces in MISSING_PIECES
        )
        + "\n\nCopy the current "
        + " and ".join(f"`{name}`" for name, _ in MISSING_PIECES)
        + " into the repository next to `app.py`, then reboot the app. "
        "These files are a set and have to travel together."
    )
    st.stop()


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
        ("composition_curve", "the component curves on the plot"),
    ):
        if not hasattr(LulevichModel, method):
            stale.append(("lulevich_model.py", feature))

    if not hasattr(LulevichModel, "local_exponent"):
        stale.append(("lulevich_model.py", "the log-slope reading"))
    if exponent_profile_figure is None:
        stale.append(("plot_utils.py", "the power-law chart"))

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


# Where this app is, and where its companion files should be. On a hosted
# deployment the working directory is not always the folder holding app.py,
# and the main file can be nested a level below the repository root while
# the companions sit at the top. Both folders go on the import path so that
# either arrangement works.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(APP_DIR)
for _folder in (APP_DIR, REPO_DIR, os.getcwd()):
    if _folder and _folder not in sys.path:
        sys.path.insert(0, _folder)


def companion_files():
    """Every .py file the app can see beside itself, for the error messages."""
    try:
        return sorted(
            name for name in os.listdir(APP_DIR) if name.endswith(".py")
        )
    except OSError:  # pragma: no cover - unreadable folder
        return []


def _find_companion_file(name, depth=2):
    """Look for `name`.py near the app: beside it, above it, one or two
    folders down from either. Returns the path, or None."""
    wanted = f"{name}.py"
    roots, seen = (APP_DIR, REPO_DIR, os.getcwd()), set()
    for root in roots:
        if not root or root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        base = root.rstrip(os.sep).count(os.sep)
        for folder, subfolders, files in os.walk(root):
            # Caches, virtual environments and hidden folders are not the
            # repository; walking into site-packages would take seconds.
            subfolders[:] = [
                d for d in subfolders
                if not d.startswith((".", "__"))
                and d not in ("site-packages", "node_modules", "venv", "env")
            ]
            if folder.rstrip(os.sep).count(os.sep) - base >= depth:
                subfolders[:] = []
            if wanted in files:
                return os.path.join(folder, wanted)
    return None


def import_companion(name):
    """
    Import one of this app's own modules, wherever in the repo it ended up.

    Returns (module, error_text). A plain import is tried first, and only
    if that fails is the file hunted for and loaded from its path. That
    second attempt is what makes a deployment work when the main file and
    its companions are not in the same folder, which is a mistake that
    otherwise shows up as "no module named ..." and nothing else.
    """
    import importlib.util

    try:
        return importlib.import_module(name), None
    except Exception as exc:  # pragma: no cover - depends on the deployment
        first = f"{type(exc).__name__}: {exc}"

    path = _find_companion_file(name)
    if path is None:
        return None, first
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module, None
    except Exception as exc:  # pragma: no cover
        return None, f"{first}; and loading {path} failed with {exc}"


# Optional dependencies: the app must still run without Google credentials
# or the Igor toolchain installed.
_sheets, SHEETS_IMPORT_ERROR = import_companion("google_sheets_manager")
initialize_sheets_manager = (
    getattr(_sheets, "initialize_sheets_manager", None) if _sheets else None
)
if _sheets is not None and initialize_sheets_manager is None:
    SHEETS_IMPORT_ERROR = (
        "google_sheets_manager.py loaded but has no initialize_sheets_manager; "
        "it is an older version than this app expects."
    )

onedrive_store, ONEDRIVE_IMPORT_ERROR = import_companion("onedrive_store")
OneDriveStore = getattr(onedrive_store, "OneDriveStore", None)
OneDriveError = getattr(onedrive_store, "OneDriveError", Exception)
if onedrive_store is not None and OneDriveStore is None:
    ONEDRIVE_IMPORT_ERROR = (
        "onedrive_store.py loaded but has no OneDriveStore in it; it is an "
        "older version than this app expects."
    )
    onedrive_store = None


def onedrive_load_problem():
    """
    Why OneDrive support did not load, in words that name the fix.

    Almost always the same thing: `onedrive_store.py` is not next to
    `app.py` in the deployed folder. The bare ImportError says "No module
    named 'onedrive_store'", which is true and tells nobody what to do, so
    it is turned into the instruction here.
    """
    if not ONEDRIVE_IMPORT_ERROR:
        return None
    text = str(ONEDRIVE_IMPORT_ERROR)
    if "onedrive_store" in text and "No module named" in text:
        # The app has already searched for the file beside itself, one
        # folder up and two folders down, so by the time this is reached it
        # is genuinely not there under that name. Saying which folder was
        # looked in, and what is in it, is the difference between a message
        # somebody can act on and one they read three times.
        seen = companion_files()
        return (
            "**`onedrive_store.py` is not in this app's folder.**\n\n"
            f"The app is running from `{APP_DIR}`, and it also looked in "
            f"`{REPO_DIR}` and in the folders under both. The Python files "
            "it can actually see beside itself are:\n\n"
            + ("\n".join(f"- `{name}`" for name in seen)
               if seen else "- (none)")
            + "\n\nIf `onedrive_store.py` is not in that list, the copy in "
            "your repository is not reaching the deployment. The usual "
            "causes, in the order they happen: the file was uploaded to a "
            "different branch from the one the app deploys; the name is not "
            "exactly `onedrive_store.py` (a browser can save it as "
            "`onedrive_store.py.txt`, and capitals matter here); or the app "
            "has not been rebooted since the commit. Check the file's page "
            "on GitHub shows Python, not plain text, and that it sits in "
            "the same folder as `app.py` on the branch the app deploys."
        )
    if "No module named" in text:
        missing = text.split("No module named", 1)[1].strip().strip("'\"")
        return (
            f"**A package this needs is not installed: `{missing}`.** Add "
            f"it to `requirements.txt` in the repository and reboot the "
            f"app."
        )
    return f"**OneDrive support could not load.** {text}"


# The archive is OneDrive. Box was here too and was removed: on a UC Davis
# tenant it cannot be authorised without an administrator, so it was a button
# that could never light up. OneDriveStore exposes the same interface, so the
# gallery did not have to change shape.
ArchiveError = RuntimeError

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
    /* Results tables. table-layout:fixed is the whole point: it divides the
       width that is actually there rather than asking for more and being
       clipped, and word-wrap lets a long cell grow downwards instead of
       sideways off the edge. */
    table.flat-table {
        width: 100%; table-layout: fixed; border-collapse: collapse;
        font-size: 0.86rem; margin: 0.35rem 0 0.15rem 0;
    }
    table.flat-table th, table.flat-table td {
        padding: 0.32rem 0.5rem; vertical-align: top;
        overflow-wrap: anywhere; word-break: normal; hyphens: auto;
        border-bottom: 1px solid rgba(128,128,128,0.25);
    }
    table.flat-table th {
        font-weight: 600; border-bottom: 2px solid rgba(128,128,128,0.45);
    }
    table.flat-table tbody tr:last-child td {border-bottom: none;}
    .flat-table-caption {
        font-size: 0.78rem; opacity: 0.72; margin-bottom: 0.6rem;
    }
    /* And where a real dataframe is still the right widget, stop it from
       spilling: let it scroll inside its own box rather than under the
       neighbouring column. */
    div[data-testid="stDataFrame"] {max-width: 100%; overflow-x: auto;}
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


def flat_table(frame, align_right=None, caption=None):
    """
    A results table that shows every column, always.

    ``st.dataframe`` lays its columns out in a scrolling grid sized before the
    surrounding column is measured. Inside a narrow column, or with more than
    about five columns, the last one is simply cut off the right-hand edge,
    and the scrollbar that would reveal it is easy to miss. Every table in
    this app is a small, finished result meant to be read at a glance, not
    browsed, so they are drawn as plain HTML instead: fixed layout, cells that
    wrap, nothing off the edge. Sorting is lost, and none of these tables is
    long enough for that to matter.

    ``align_right`` names the columns holding numbers.
    """
    columns = list(frame.columns)
    right = set(align_right or [])
    head = "".join(
        f'<th style="text-align:{"right" if c in right else "left"}">{c}</th>'
        for c in columns
    )
    body = []
    for _, row in frame.iterrows():
        cells = "".join(
            f'<td style="text-align:{"right" if c in right else "left"}">'
            f"{'' if pd.isna(row[c]) else row[c]}</td>"
            for c in columns
        )
        body.append(f"<tr>{cells}</tr>")
    st.markdown(
        '<table class="flat-table"><thead><tr>' + head + "</tr></thead><tbody>"
        + "".join(body) + "</tbody></table>"
        + (f'<div class="flat-table-caption">{caption}</div>' if caption else ""),
        unsafe_allow_html=True,
    )


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
    # A light blue field of points with no outline, and one dark dashed
    # line over it. The eye separates them by lightness and by the kind of
    # mark, which survives a greyscale print and colour blindness both.
    "data_color": "#79c2e8",
    "fit_color": "#000000",
    # A different colour for the fit every time the fitted range moves, so
    # two screenshots taken from different ranges cannot be mistaken for one
    # another. Untick it to keep the colour chosen above.
    "recolour_on_range": True,
    # Big enough that a dense curve reads as one continuous light band, so
    # the dashed model line has something to sit on rather than something
    # to hide in.
    "marker_size": 9,
    "line_width": 3,
    "plot_height": 520,
    "show_grid": False,
    # The line in the corner of the plot saying which stretch of the curve
    # the numbers came from. On, because a figure without it cannot be
    # checked once it has left the app.
    "show_range_note": True,
    "axis_title_size": 28,
    "tick_size": 22,
    "axis_width": 4,
    "bold_axes": True,
    "log_scale": False,
    # Off by default: the fit is one Lulevich curve, and the per-element
    # curves beside it invite reading three fits into a plot that has one.
    "show_components": True,
    "bare_plot": False,
    "show_legend": True,
    # Separate from the bare switch: sometimes you want the markings but not
    # the model, to look at the data on its own.
    # One switch, not two. The measured points and the model drawn over
    # them are the plot; wanting one without the other is rare enough that
    # it does not deserve half the checkbox list, and two boxes that are
    # both ticked in every real session is two boxes too many.
    "show_data_and_fit": True,
    "plot_layers": [],
    "show_component_heights": False,
    # Relative deformation runs 0 to 1 by definition, so that is the honest
    # default: two cells squashed to different depths then look different,
    # which they are.
    "x_axis_mode": "0 to 1 (full squash)",
    "y_axis_mode": "Fit to the data",
    "x_axis_min": 0.0,
    "x_axis_max": 1.0,
    "y_axis_min": 0.0,
    "y_axis_max": 1.0,
    # Off by default. After a fit the plot should be the data, the fitted
    # line and its parts; a shaded band across the whole chart and a marker
    # for a video nobody has loaded are decoration that hides the residual
    # you are trying to see. Both are one tick away under the plot.
    "show_fit_window": False,
    "show_video_marker": False,
    "show_rupture_marker": True,
    "show_schematic": True,
    "schematic_style": "Mechanics schematic",
    "sharing_preview_eps": 0.10,
    "show_schematic_moduli": True,
    "plot_width": 2.4,
    # geometry / model
    "radius_mode": "From height",
    "radius_aspect": 0.55,
    "cell_radius_um": 4.45,
    "membrane_thickness_nm": 4.0,
    # The sub-membranous protein layer, two orders thicker than the bilayer.
    # Only used to express a fitted tension as an equivalent modulus.
    "protein_coat_nm": 200.0,
    # Relaxed sarcomere length, and how freely the cell spreads sideways as
    # it is squashed. Neither enters the fit; they turn a deformation into a
    # sarcomere length that can be judged against what a sarcomere can do.
    "sarcomere_nm": 2100.0,
    "sarcomere_spread": 1.0,
    # Shape, and the stiffening that follows from it. q = 0 is the classic
    # small-strain Lulevich model; see confinement_factor in lulevich_model.
    "cell_shape": "Sphere (a rounded cell)",
    "membrane_protein": "Not known — test for it",
    "confinement": 0.0,
    "confinement_scan": None,
    "component_search": None,
    "hypothesis_search": None,
    # One fitted range for the whole app: guided mode and full
    # control set the same two numbers, so switching between them
    # never changes what is being fitted.
    "window_start": 0.0,
    # Acquisition. The probe diameter is also used by the four-regime C2C12
    # fit, which corrects its Hertzian terms for the curvature of the sphere.
    "probe_diameter_um": 40.0,
    "approach_speed_um_s": 2.0,
    "poisson_membrane": 0.50,
    "poisson_interior": 0.50,
    # cell type
    "cell_type": "Myoblast (C2C12)",
    # Off unless you say so: almost every curve is a living cell, and a tick
    # that quietly collapses the model to one material must be one you made.
    "fixed_cell": False,
    "nucleus_fraction": 0.35,
    "nucleus_radius_um": 1.55,
    "nucleus_radius_mode": "From cell radius",
    "poisson_nucleus": 0.50,
    "nucleus_onset": 0.20,
    "onset_mode": "Scan for best",
    "_scanned_onset": None,
    # fitting
    # The page opens on the carried-forward fit: one component to a
    # stretch, each modulus found in turn and then held while the next is
    # found. Fitting each component only inside its own stretch is the
    # other way, one line away in "How the components share the load".
    "c2c12_fit_mode": "4-regime piecewise (C2C12)",
    # Where regimes 2, 3 and 4 start and where the fit ends, in PERCENT
    # relative deformation. Regime 1 always starts at 0 %. These are the
    # C2C12 prior's (see piecewise_prior): contact under 5 %, the nucleus
    # met at about 50 % and its bump over about 25 % later. Each new curve
    # has its boundaries found inside those constraints straight away.
    "pw_b1": 5.0,
    "pw_b2": 50.0,
    "pw_b3": 75.0,
    "pw_end": 91.2,
    # Per-coefficient initial guesses and bounds the user has changed, as
    # {name: {"p0": .., "lower": .., "upper": ..}}. Empty means the spec's.
    "pw_settings": {},
    "pw_log_y": False,
    "pw_view": "stacked",
    # The cell shell keeps stiffening after regime 2: K_shell*(x - ε₁)^3,
    # with the K_shell regime 2 measured, is carried through regimes 3 and 4
    # and they fit what is left on top of it.
    "pw_membrane_throughout": True,
    # Where each component stops adding load when it is set by hand, in
    # percent, {name: until}. Absent means its regime's end (the fit's end
    # for the membrane acting throughout), so it follows the boundaries.
    "pw_until": {},
    # Each component in the model or not. Off holds its coefficient at 0.
    # The model is the four components: membrane, cytoskeleton, nuclear
    # envelope and the inside of the nucleus, all four on. There is no
    # contact / alignment component: the first stretch, [0, ε₁), is the
    # probe settling onto the cell rather than the cell itself, and it is
    # carried by the baseline C₀ alone.
    "pw_use_K_shell": True,
    "pw_use_K_cyto": True,
    "pw_use_K_nucleus": True,
    "pw_use_K_core": True,
    # The straight lines on the log-log plot: their exponents, the point of
    # the data they pivot on and how far they run.
    "log_guides": True,
    "log_guide_p1": 3.0,
    "log_guide_p2": 1.5,
    "log_guide_anchor_pct": 30.0,
    "log_guide_span": 0.4,
    # The cells fitted so far this session, each with its own boundaries,
    # moduli and curve, for the All cells tab.
    "pw_collection": {},
    # The C2C12 constraints the boundaries are found inside, in percent:
    # ε₁ (end of the contact artefact) under 5 %, ε₂ (nucleus met) in the
    # prior's 44 to 62 %, and ε₃ - ε₂ (how long the nuclear bump lasts)
    # 25 % give or take 10. Set from piecewise_prior; editable on the page.
    # ε₁ is where the second component joins, not the end of a contact
    # artefact, so it is allowed anywhere in the first third of the squash.
    "pw_band1_lo": 1.0, "pw_band1_hi": 35.0,
    "pw_band2_lo": 44.0, "pw_band2_hi": 62.0,
    "pw_span_lo": 15.0, "pw_span_hi": 35.0,
    # The last boundary search, with the curve and settings it was run on.
    "pw_boundary_search": None,
    # The order the four components are met in, first to last.
    "component_order": ["membrane", "interior", "nucleus_shell", "nucleus"],
    # Which of the three ways the four components are fitted with, and the
    # mixture the third of them last found.
    "pw_style": "carried",
    "pw_best_carry": None,
    "pw_style_note": None,
    # How the probe met the cell, recorded with it and written into the
    # spreadsheet row.
    "compression_type": "Head-on",
    # How the boundaries are placed, and the R² every fit has to reach.
    "pw_method": "refined",
    # Whether ▶ Fit & plot places the boundaries from the curve or fits at
    # the numbers on the board exactly as they are.
    "pw_eps_way": "found",
    "pw_target_r2": 0.999,
    # Every route's placement for this curve, scored, and the one in use.
    "pw_placements": None,
    "pw_selected": None,
    # What the last "reach the target" search had to do, in one line.
    "pw_reach_note": None,
    # Guided by default: most people opening this want a number, not a
    # spring network. Everything is still one expander away.
    "ui_mode": "Guided · plain language",
    "model_kind": "Segmented (each part takes over in turn)",
    "segment_break_1": 0.15,
    "segment_break_2": 0.40,
    # What each element does either side of the first boundary. These are the
    # two choices the physics leaves open, and the combination search below
    # settles them from the data when you ask it to.
    # The order the cell is met in: the membrane from first contact and
    # still stiffening after ε₁, the cytoskeleton joining it at ε₁, the
    # nuclear envelope at ε₂ and the inside of the nucleus after that. Each
    # component then has an onset of its own, which is what lets the fit
    # tell them apart instead of putting one of them at zero.
    "membrane_after_break": "holds what it reached",
    "cyto_starts_at": "at ε₁",
    "highlight_segment": "(none)",
    "composition_search": None,
    # Per-element windows: the stretch of the squash each element is allowed
    # to carry load over. Off unless asked for, because switching them on
    # changes what every modulus means, and that has to be a decision.
    "use_element_windows": False,
    "element_window_search": None,
    # Boundaries measured from a batch of curves, per cell type. They
    # replace the written-down defaults for every new curve of that type.
    "learned_boundaries": {},
    "arrangement_search": None,
    # The whole curve, until you say otherwise.
    "window_end": 1.00,
    "procedure": "All at once",
    "crossover_mode": "Scan for best",
    "crossover": 0.18,
    "use_membrane": True,
    "use_interior": True,
    "use_nucleus": True,
    # The nucleus is a balloon of its own: an envelope around a filling.
    # Offered wherever there is a nucleus at all, and on by default there,
    # because a nucleus without a skin is not a nucleus.
    "use_nucleus_shell": True,
    # The cortical layer under the membrane. Offered only where a cell type
    # has one; see OPTIONAL_TERMS.
    "use_cortex": True,
    # Only offered for cell types whose membrane is two springs, so off here
    # and switched on by the cell type's own defaults.
    "use_tension": False,
    "regime_mode": False,
    "regime_split": 0.40,
    "stage_of_membrane": 2,
    "stage_of_interior": 1,
    "stage_of_nucleus": 3,
    # Every term needs one, or saving a preset for a cell type that has the
    # envelope or a cortex reaches for a key that was never made.
    "stage_of_nucleus_shell": 3,
    "stage_of_cortex": 1,
    "stage_of_tension": 2,
    "refine_iterations": 3,
    "seed_parallel": True,
    "range_presets": {},
    "drag_target": "(off)",
    "weighting": "uniform",
    "fit_offset": False,
    "live_fit": False,
    "legacy_eps_mode": "as_set",
    # cell metadata
    "cell_name": "",
    "cell_height_um": 8.09,
    "spring_constant": 0.0,
    "invols_nm_per_V": 50.0,
    "operator": "",
    "cell_notes": "",
    "onedrive_store": None,
    # The sheet read back for the database tab, and which database that tab
    # is reading. Both are cleared when a row is written, so a cell just
    # sent shows up without a refresh by hand.
    "sheet_rows": None,
    "db_source": "Google Sheet",
    "onedrive_root": "AFM cells",
    "onedrive_account": "personal",
    "_device_login": None,
    "archive_index": None,
    "db_selection": [],
    "upload_video_with_cell": False,
    "db_search": "",
    "db_view": "Gallery",
    "exploration": None,
    "video_link": "",
    # Measured off the video frame, and what you noticed while watching it.
    # Both go into the spreadsheet beside the moduli. 0 means not measured,
    # which is not the same as a cell of zero height.
    "video_height_um": 0.0,
    "video_comment": "",
    # The last successful fit, kept so a rerun (uploading a video, ticking a
    # box, changing tab) does not wipe the results off the page.
    "_last_fit": None,
    # What the per-material bars were last drawn from, so a fit that moves a
    # boundary can redraw them once instead of leaving them showing the
    # placement before it.
    "_bars_drawn_with": None,
    "_bars_redrawn_for": None,
    "_plot_png": None,
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
    # Draw the cell yourself when the detector guesses wrong, and measure the
    # probe to turn pixels into micrometres.
    "video_manual_cell": False,
    "video_cell_box_x": (0.35, 0.65),
    "video_cell_box_y": (0.35, 0.75),
    "video_use_probe_scale": False,
    "video_probe_box_x": (0.10, 0.90),
    "video_probe_width_um": 60.0,
    "video_crop_pad": 0.35,
    "video_saved_frame": None,
    "video_saved_frame_index": None,
    "video_reject_dark": True,
    "video_find_nucleus": True,
    "video_strip_lines": True,
    # The frame beside the curve is off: the video is loaded to link a cell
    # to its record and to measure its shape, and a frame pinned next to the
    # force curve is neither.
    "video_show_panel": False,
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
# Everything that belongs to one cell rather than to the session. The
# connections, the geometry and the display settings are deliberately not in
# this list: they are the setup, and a person working through a plate sets
# them once. It is also the list the new-cell button tells the rerun not to
# carry, so the clear below is not undone a dozen lines later.
NEW_CELL_CLEARS = (
    "data", "results", "_last_fit", "_last_fit_signature", "_plot_png",
    "cell_name", "cell_notes", "exploration", "composition_search",
    "arrangement_search", "component_search", "confinement_scan",
    "hypothesis_search", "boundary_search", "boundary_candidates",
    "element_window_search", "plot_layers", "_auto_picked", "_auto_notes",
    "_suggested_window", "_slope_profile",
    "video_path", "video_info", "video_track",
    "video_name", "video_link", "eps_percent_fix",
    "video_saved_frame", "video_saved_frame_index", "pw_boundary_search",
    "_pw_auto_found", "pw_placements", "pw_selected", "pw_reach_note",
    # The way of fitting is a property of the cell type, not of the cell
    # before this one: a C2C12 starts on the carried-forward fit.
    "c2c12_fit_mode",
)

if st.session_state.pop("_start_new_cell", False):
    for key in NEW_CELL_CLEARS:
        st.session_state[key] = DEFAULTS.get(key)
    # What each upload box last handed over. Cleared with the rest, or
    # re-uploading the same file for the next cell would look like no change
    # at all and be silently ignored.
    st.session_state["_video_seen"] = {}
    for key in [k for k in st.session_state if k.startswith("window_")]:
        del st.session_state[key]
    # Merged, not replaced: the button staged the session's settings here so
    # they survive a run that never draws their widgets.
    staged = dict(st.session_state.get("_pending_settings") or {})
    staged.update({
        "window_start": DEFAULTS["window_start"],
        "window_end": DEFAULTS["window_end"],
    })
    st.session_state["_pending_settings"] = staged

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

# A session saved when "Fixed cell" was an entry in the cell-type dropdown.
# It is a tick beside the materials now, so the stored setting becomes that
# tick rather than being left pointing at an entry the dropdown no longer
# offers, which would throw before the page drew anything.
if st.session_state.get("cell_type") == "Fixed cell":
    st.session_state["cell_type"] = DEFAULTS["cell_type"]
    st.session_state["fixed_cell"] = True


# Colours for the fitted line, walked one step at a time as the range moves.
# Black first, because that is the one a figure wants. The rest are dark
# enough to read as a line over light blue points and far enough apart to be
# told from one another at a glance.
FIT_COLORS = (
    "#000000",  # black
    "#c0392b",  # brick
    "#1a7f37",  # forest
    "#6a3d9a",  # violet
    "#b35c00",  # burnt orange
    "#8b1a62",  # plum
)
# Deliberately no blue or teal in that list: the measured points are blue,
# and a model line in a neighbouring blue is the one pairing where the two
# cannot be told apart at a glance.


def fit_line_colour():
    """The colour the fitted curve is drawn in right now."""
    if not st.session_state.get("recolour_on_range", True):
        return st.session_state.get("fit_color", FIT_COLORS[0])
    index = int(st.session_state.get("_fit_colour_step", 0))
    return FIT_COLORS[index % len(FIT_COLORS)]


def note_fit_range(lo, hi):
    """
    Advance the fit colour when the fitted range has actually moved.

    Called once a pass, before anything is drawn, so every route into a new
    range gets the same treatment: the bar, the boxes, a drag on the plot, a
    preset, or the window a search settled on. The first range a curve is
    given is not a change, so it keeps black.
    """
    try:
        now = (round(float(lo), 4), round(float(hi), 4))
    except (TypeError, ValueError):
        return
    seen = st.session_state.get("_fit_colour_range")
    if seen is None:
        st.session_state["_fit_colour_range"] = now
        return
    if now != seen:
        st.session_state["_fit_colour_range"] = now
        st.session_state["_fit_colour_step"] = (
            int(st.session_state.get("_fit_colour_step", 0)) + 1
        )


note_fit_range(
    st.session_state.get("window_start", 0.0),
    st.session_state.get("window_end", 1.0),
)


def suggested_plot_name(cell_name, fit, date_acquired, extension):
    """
    A file name you would have typed yourself.

    Takes the cell name rather than reading it from session state, so it can
    be checked without a running app.
    """
    safe = "".join(
        ch if (ch.isalnum() or ch in "-_") else "_" for ch in (cell_name or "").strip()
    )
    parts = [safe or "cell", str(date_acquired)]
    if fit and fit.get("success"):
        parts.append(f"Em{fit.get('Em_MPa', 0.0):.3g}MPa")
        parts.append(f"Ec{fit.get('Ei_kPa', 0.0):.3g}kPa")
    return "_".join(parts) + extension


def save_plot_controls(figure, fit, date_acquired):
    """
    Offer the figure as a file, named after the cell.

    HTML is always available and costs nothing: it is the figure object
    serialised. PNG is not offered until asked for, because rendering one
    means starting a headless browser, which takes seconds. Doing that on
    every script run made the whole app feel slow, since Streamlit reruns
    the script on every widget touch.
    """
    st.markdown("**Save the plot**")
    st.download_button(
        "🖼️ Save as HTML",
        data=figure.to_html(include_plotlyjs="cdn"),
        file_name=suggested_plot_name(
            st.session_state["cell_name"], fit, date_acquired, ".html"
        ),
        mime="text/html",
        **STRETCH,
    )

    if st.button("🖼️ Prepare a PNG", **STRETCH):
        try:
            with st.spinner("Rendering…"):
                st.session_state["_plot_png"] = figure.to_image(
                    format="png", width=1400, height=900, scale=2
                )
        except Exception as exc:
            st.session_state["_plot_png"] = None
            st.caption(
                "PNG export is not available on this deployment "
                f"({type(exc).__name__}). The camera icon on the chart still "
                "saves one, and the HTML keeps the text sharp at any size."
            )

    png = st.session_state.get("_plot_png")
    if png:
        st.download_button(
            "⬇️ Download the PNG",
            data=png,
            file_name=suggested_plot_name(
                st.session_state["cell_name"], fit, date_acquired, ".png"
            ),
            mime="image/png",
            type="primary",
            **STRETCH,
        )
    st.caption("Named after the cell, the date and the moduli.")


def show_search_maths():
    """
    The mathematics behind the "Work it out for me" button.

    Someone is entitled to know what a button decided on their behalf. This
    is the whole of it: three candidate models, one scoring rule, one tie
    rule.
    """
    st.markdown("**1 · The three arrangements it compares**")
    st.caption(
        "These are different physics, not different settings. Each says "
        "something different about how the parts of the cell carry the load."
    )
    st.markdown("*Side by side* — every part squashed by the same ε, forces add:")
    st.latex(
        r"F(\varepsilon) = A_m E_m \varepsilon^{3} + A_i E_c \varepsilon^{3/2} "
        r"+ A_n E_n \langle \varepsilon - \varepsilon_0 \rangle^{3/2}"
    )
    st.markdown("*Stacked* — every part carries the same F, squashes add:")
    st.latex(
        r"\varepsilon(F) = a F^{1/3} + b F^{2/3} + c \langle F - F_0 \rangle^{2/3},"
        r"\qquad E_m = \frac{1}{A_m a^{3}}"
    )
    st.markdown(
        "*Segmented* — side by side, but each part switches on at its own "
        "boundary, which is the extra thing the other two cannot express:"
    )
    st.latex(
        r"F(\varepsilon) = A_m E_m\, g_m(\varepsilon) + A_i E_c\, g_c(\varepsilon) "
        r"+ A_n E_n \langle \varepsilon - \varepsilon_2 \rangle^{3/2}"
    )
    st.caption(
        "with g_m either min(ε, ε₁)³ or ε³, and g_c either ⟨ε − ε₁⟩³ᐟ² or "
        "ε³ᐟ², which is the four combinations it also searches."
    )

    st.markdown("**2 · How it scores them**")
    st.caption(
        "Not by R², which rises whenever a model is given more freedom, and "
        "not by residual sum, for the same reason. By how well each predicts "
        "points it was never fitted on."
    )
    st.latex(
        r"\mathrm{CV} = \frac{1}{R}\sum_{r=1}^{R} \frac{1}{K}\sum_{k=1}^{K} "
        r"\sqrt{\frac{1}{|S_k|}\sum_{i \in S_k} "
        r"\bigl(F_i - \hat{F}^{(-k)}(\varepsilon_i)\bigr)^{2}}"
    )
    st.caption(
        "The points are split into K folds; the model is fitted on everything "
        "except fold k and asked to predict fold k; that is repeated over R "
        "different random splits, because one split of a few hundred points "
        "is noisy enough to reorder candidates that are genuinely tied. "
        "Lower is better. Here K = 5 and R = 3."
    )

    st.markdown("**3 · How it breaks a tie**")
    st.latex(
        r"\tau = \max\bigl(0.05\,\mathrm{CV}_{\min},\; "
        r"s_{\min} + \max_j s_j\bigr)"
    )
    st.caption(
        "s is the spread of a candidate's score across those repeated "
        "splits. Anything within τ of the best is called tied, because a gap "
        "smaller than the amount the number moves when you redraw the folds "
        "is not evidence. Among tied candidates the one with the fewest free "
        "moduli wins: a part the curve cannot see gives a number that will "
        "wander from cell to cell."
    )

    st.markdown("**4 · Where the boundaries come from**")
    st.caption(
        "The moduli are linear once the boundaries are fixed, so they are one "
        "exact bounded least-squares solve. The boundaries are not: they sit "
        "inside min and ⟨ ⟩ where the residual has a kink at every point a "
        "boundary crosses, which gradient methods handle badly. So they are "
        "profiled out instead: a grid of (ε₁, ε₂) is tried, each with the "
        "exact solve, and the pair with the smallest residual wins. Two "
        "refinement rounds then re-grid inside one step either side of that "
        "winner, narrowing it by roughly 25 times."
    )


# Each fitted term: where its prefactor and its modulus are kept, and the
# shape of ε it multiplies. One table, so the equation on the page and the
# equation in the code cannot drift apart.
EQUATION_TERMS = {
    "tension": ("At", "T0", "A_t", "T_0"),
    "membrane": ("Am", "Em", "A_m", "E_m"),
    "cortex": ("Ai", "Ecx", "A_i", "E_{cx}"),
    "interior": ("Ai", "Ei", "A_i", "E_c"),
    "nucleus_shell": ("An_shell", "Ene", "A_{ne}", "E_{ne}"),
    "nucleus": ("An", "En", "A_n", "E_n"),
}


def _basis_latex(term, fit):
    """The shape of ε one term multiplies, as written in the fit."""
    mode = fit.get("membrane", "continue")
    from_break = fit.get("cyto_start") == "break"
    # The shell has three histories, not two: it can carry on stretching,
    # hold what it reached at ε₁, or only start stretching there.
    shell = {
        "freeze": r"\min(\varepsilon,\ \varepsilon_1)",
        "late": r"\langle \varepsilon - \varepsilon_1 \rangle",
    }.get(mode, r"\varepsilon")
    if term == "tension":
        return shell
    if term == "membrane":
        return shell + "^{3}"
    if term == "cortex":
        return r"\varepsilon^{3/2}"
    if term == "interior":
        return (r"\langle \varepsilon - \varepsilon_1 \rangle^{3/2}"
                if from_break else r"\varepsilon^{3/2}")
    if term == "nucleus_shell":
        return r"\langle \varepsilon - \varepsilon_2 \rangle^{3}"
    return r"\langle \varepsilon - \varepsilon_2 \rangle^{3/2}"


def equation_pieces(fit):
    """
    The fitted model, term by term: symbol, shape of ε, and coefficient.

    The coefficient is the prefactor times the modulus, in newtons, which is
    the number actually multiplying that shape of ε. Written this way the
    equation on the page can be evaluated by hand and checked against the
    drawn curve, which is the whole point of printing it.
    """
    if not (fit and fit.get("success")):
        return []
    pieces = []
    for term in ("tension", "membrane", "cortex", "interior",
                 "nucleus_shell", "nucleus"):
        if term not in (fit.get("terms") or ()):
            continue
        a_key, e_key, a_tex, e_tex = EQUATION_TERMS[term]
        prefactor = float(fit.get(a_key, float("nan")))
        modulus = float(fit.get(e_key, float("nan")))
        if not (np.isfinite(prefactor) and np.isfinite(modulus)):
            continue
        pieces.append({
            "term": term,
            "name": plain_name(term),
            "prefactor": prefactor,
            "modulus": modulus,
            "coefficient_N": prefactor * modulus,
            "symbols": f"{a_tex} {e_tex}",
            "basis": _basis_latex(term, fit),
        })
    return pieces


def _sci_latex(value, digits=3):
    """A number as LaTeX, in scientific notation where that reads better."""
    if not np.isfinite(value):
        return r"\mathrm{n/a}"
    if value == 0:
        return "0"
    power = int(np.floor(np.log10(abs(value))))
    if -2 <= power <= 3:
        return f"{value:.{digits}g}"
    mantissa = value / (10.0 ** power)
    return f"{mantissa:.{digits}g} \\times 10^{{{power}}}"


def _relative_error(piece, fit):
    """One coefficient's fractional uncertainty, or nan where there is none.

    A coefficient is a fixed geometric prefactor times a fitted modulus, so
    the fractional uncertainty of the two is the same number and the units
    the modulus is reported in cannot change it.
    """
    key, _unit, std_key = MODULUS_FIELDS[piece["term"]]
    try:
        value = float(fit.get(key))
        error = float(fit.get(std_key))
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(value) and np.isfinite(error)) or value == 0:
        return float("nan")
    return abs(error / value)


def _coefficient_latex(piece, fit, factor):
    """A coefficient written with its uncertainty, where the fit has one."""
    value = piece["coefficient_N"] * factor
    relative = _relative_error(piece, fit)
    if not np.isfinite(relative) or relative <= 0:
        return _sci_latex(value)
    return (r"(" + _sci_latex(value) + r" \pm "
            + _sci_latex(value * relative, digits=2) + r")")


def sharing_label(fit):
    """How the components shared the load in this fit, as the page names it."""
    if fit.get("coupling") == "piecewise":
        return PW_SHARE
    for label, key in MODEL_KEYS.items():
        if key == fit.get("coupling"):
            return label
    return str(fit.get("model_label") or st.session_state.get("model_kind", ""))


COMPRESSION_TYPES = ("Head-on", "Off-centre", "Edge-on", "Not recorded")

# The lab's own table, column for column. This is the row that gets pasted
# into the shared sheet, so its headers, its order and the way each number
# is written are fixed here and nowhere else. Values are written the way
# the sheet writes them, "3.608 MPa ± 0.026", because that is the column a
# person reads; every number as a plain float, for averaging, is the other
# block in the same panel.
LAB_SHEET_COLUMNS = (
    "Experiment Date", "Cell ID", "Cell Height, h (μm)",
    "Spring Constant, K (N/m)", "Compression Type",
    "Boundaries (ε1 and ε2)", "Young's Modulus, Em (MPa)",
    "Young's Modulus, Ec (kPa)", "Young's Modulus, Ene (MPa)",
    "Young's Modulus, En (kPa)", "En range (ε)", "Fit Quality (R²)",
    "Chi squared",
)
LAB_SHEET_MODULI = (("Young's Modulus, Em (MPa)", "Em_MPa", "MPa"),
                    ("Young's Modulus, Ec (kPa)", "Ei_kPa", "kPa"),
                    ("Young's Modulus, Ene (MPa)", "Ene_MPa", "MPa"),
                    ("Young's Modulus, En (kPa)", "En_kPa", "kPa"))


def _sheet_date(value):
    """The date the way the sheet writes it: M/D/YYYY."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parts = [int(p) for p in text.split("-")]
        if len(parts) == 3:
            return f"{parts[1]}/{parts[2]}/{parts[0]}"
    except ValueError:
        pass
    return text


def lab_sheet_row(fit):
    """
    This fit as one row of the lab's table, written as the table writes it.

    Thirteen columns, always the same thirteen and always in this order, so
    a row pasted under the last one lines up without anybody checking. A
    modulus the fit does not have is an empty cell, not a zero.
    """
    if not (fit and fit.get("success")):
        return {}
    edges = {name: value for value, name in fit_edges(fit)}
    lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
    chi = fit.get("chi_squared_reduced", float("nan"))
    row = {
        "Experiment Date": _sheet_date(st.session_state.get("date_acquired")),
        "Cell ID": st.session_state.get("cell_name", ""),
        "Cell Height, h (μm)": st.session_state.get("cell_height_um", ""),
        "Spring Constant, K (N/m)": st.session_state.get("spring_constant", ""),
        "Compression Type": st.session_state.get("compression_type", ""),
        "Boundaries (ε1 and ε2)": ", ".join(
            f"{name} = {value:.3f}" for value, name in fit_edges(fit)) or "—",
    }
    for column, key, unit_name in LAB_SHEET_MODULI:
        value = fit.get(key)
        error = fit.get(f"{key}_std")
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float("nan")
        if not np.isfinite(value):
            row[column] = ""
            continue
        text = f"{value:.4g} {unit_name}"
        try:
            error = float(error)
        except (TypeError, ValueError):
            error = float("nan")
        if np.isfinite(error):
            text += f" ± {error:.2g}"
        row[column] = text
    row["En range (ε)"] = f"{lo:.3f} to {hi:.3f}"
    row["Fit Quality (R²)"] = round(
        float(fit.get("r_squared", float("nan"))), 5)
    row["Chi squared"] = (f"χ²/dof = {float(chi):.3g}"
                          if np.isfinite(chi) else "")
    # Never reordered and never short: the sheet's columns are the contract.
    return {column: row.get(column, "") for column in LAB_SHEET_COLUMNS}


def fit_record(fit, unit="nN"):
    """
    Everything this fit measured, as one flat record of name → value.

    Flat and unformatted on purpose. This is what gets pasted into a
    spreadsheet, and a number that arrived as "2.01 MPa" is text that Excel
    will not average. Units live in the column names, values are plain
    floats, and the two rows are in the same order, so one paste is a row of
    a growing table.

    One format whichever way the components shared the load: the columns
    every fit has come first, in the same order, so rows from piecewise and
    segmented fits stack into one table; what only one way measures comes
    after them.
    """
    if not (fit and fit.get("success")):
        return {}
    factor, unit_label = FORCE_UNITS.get(unit, (1e9, "nN"))
    lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
    # The boundaries this fit actually has: the lines on the curve.
    edges = {name: value for value, name in fit_edges(fit)}
    record = {
        "Fit ID": fit.get("fit_id") or fit_id(fit),
        "Cell ID": st.session_state.get("cell_name", ""),
        "Cell type": st.session_state.get("cell_type", ""),
        "Experiment date": str(st.session_state.get("date_acquired", "")),
        "Load sharing": sharing_label(fit),
        "Cell height (um)": st.session_state.get("cell_height_um", ""),
        "Spring constant (N/m)": st.session_state.get("spring_constant", ""),
        "eps min": round(lo, 4),
        "eps max": round(hi, 4),
        "eps1": round(float(edges["ε₁"]), 4) if "ε₁" in edges else "",
        "eps2": round(float(edges["ε₂"]), 4) if "ε₂" in edges else "",
        "eps3": round(float(edges["ε₃"]), 4) if "ε₃" in edges else "",
        "q (confinement)": round(float(fit.get("confinement", 0.0) or 0.0), 4),
    }
    for term in ALL_TERMS:
        if term not in (fit.get("terms") or ()):
            continue
        key, unit_name, std_key = MODULUS_FIELDS[term]
        value = fit.get(key)
        error = fit.get(std_key)
        label = f"{plain_name(term)} {key.split('_')[0]} ({unit_name})"
        record[label] = (float(value)
                         if value is not None and np.isfinite(float(value))
                         else "")
        record[f"± {label}"] = (
            float(error) if error is not None and np.isfinite(float(error))
            else ""
        )
        # The interval as two plain numbers, so a spreadsheet can draw error
        # bars from them without anybody parsing "1.98 to 2.06" back apart.
        if (value is not None and np.isfinite(float(value))
                and error is not None and np.isfinite(float(error))):
            record[f"{label} 95% low"] = max(
                float(value) - 1.96 * float(error), 0.0
            )
            record[f"{label} 95% high"] = float(value) + 1.96 * float(error)
        else:
            record[f"{label} 95% low"] = ""
            record[f"{label} 95% high"] = ""
    record.update({
        "R2": float(fit.get("r_squared", float("nan"))),
        "adjusted R2": float(fit.get("adj_r_squared", float("nan"))),
        "chi2": float(fit.get("chi_squared", float("nan"))),
        "chi2 per dof": float(fit.get("chi_squared_reduced", float("nan"))),
        "RMSE (N)": float(fit.get("rmse", float("nan"))),
        "n points": int(fit.get("n_points", 0)),
        "free parameters": int(fit.get("n_params", 0)),
        "weighting": fit.get("weighting", "uniform"),
    })
    pw = fit.get("piecewise")
    if pw:
        # What only the regime-by-regime sharing measures.
        for term, symbol, key, unit_name in EXTRA_PIECEWISE_ROWS:
            label = f"{EXTRA_NAMES[term]} {symbol} ({unit_name})"
            value = float(fit.get(key, float("nan")))
            error = float(fit.get(f"{key}_std", float("nan")))
            record[label] = value
            record[f"± {label}"] = error
        b = pw.get("boundaries_pct") or ()
        for name, value in zip(("eps1 (%)", "eps2 (%)", "eps3 (%)", "x_end (%)"),
                               list(b)[1:5]):
            record[name] = round(float(value), 3)
        record["eps route"] = pw.get("boundary_source", "")
        record["membrane throughout"] = bool(pw.get("membrane_throughout"))
        flat = pw.get("flat") or {}
        for name in ("k_align", "K_shell", "K_cyto", "K_nucleus", "K_core"):
            if name in flat:
                record[f"{name} (N/%^p)"] = float(flat[name])
                record[f"± {name}"] = float(flat.get(f"{name}_se", float("nan")))
        record["C0 (N)"] = float(flat.get("C0_N", float("nan")))
        return record
    record.update({
        "membrane past eps1": fit.get("membrane", ""),
        "cytoskeleton starts": fit.get("cyto_start", ""),
        "condition number": (
            float(fit["condition_number"])
            if np.isfinite(fit.get("condition_number", float("nan"))) else ""
        ),
        "worst basis correlation": (
            abs(float(fit["worst_correlation"]))
            if fit.get("worst_correlation") is not None
            and np.isfinite(fit.get("worst_correlation", float("nan"))) else ""
        ),
    })
    for piece in equation_pieces(fit):
        label = f"coefficient {TERM_SYMBOLS.get(piece['term'], piece['term'])} ({unit_label})"
        record[label] = piece["coefficient_N"] * factor
        relative = _relative_error(piece, fit)
        record[f"± {label}"] = (
            piece["coefficient_N"] * factor * relative
            if np.isfinite(relative) else ""
        )
    return record


def _tsv(record):
    """A record as two tab-separated lines: headers, then values."""

    def cell(value):
        if isinstance(value, float):
            if not np.isfinite(value):
                return ""
            return f"{value:.6g}"
        return str(value)

    return ("\t".join(record.keys()) + "\n"
            + "\t".join(cell(v) for v in record.values()))


def _csv(record):
    """The same record as CSV, quoted properly, for the download button."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(list(record.keys()))
    writer.writerow(_tsv(record).split("\n")[1].split("\t"))
    return buffer.getvalue()


def fitted_equation(fit, unit="nN", heading=True):
    """
    The model that fits best, written out twice: as symbols and as numbers.

    The symbolic line says what was assumed. The numeric line is that same
    equation with this cell's numbers in it, so it can be typed into
    anything and evaluated, and so two cells can be compared as equations
    rather than as tables of moduli.
    """
    pieces = equation_pieces(fit)
    if not pieces:
        st.caption("Fit the curve and the equation appears here.")
        return
    q = float(fit.get("confinement", 0.0) or 0.0)
    factor, unit_label = FORCE_UNITS.get(unit, (1e9, "nN"))
    confine = (r"(1-\varepsilon)^{-" + f"{q:g}" + r"}\," ) if q else ""
    bracket_open, bracket_close = (r"\left[", r"\right]") if q and len(pieces) > 1 else ("", "")

    if heading:
        st.markdown("**The model that fits best, written out**")
    st.latex(
        r"F(\varepsilon) = " + confine + bracket_open
        + " + ".join(f"{p['symbols']}\\, {p['basis']}" for p in pieces)
        + bracket_close
    )
    st.latex(
        r"\frac{F(\varepsilon)}{\mathrm{" + unit_label + r"}} = " + confine
        + bracket_open
        + " + ".join(
            _coefficient_latex(p, fit, factor) + r"\," + p["basis"]
            for p in pieces
        )
        + bracket_close
    )
    # The same boundaries as the lines on the curve and the results rows.
    boundaries = [f"{name} = {value:.3f}" for value, name in fit_edges(fit)]
    lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
    st.caption(
        "⟨x⟩ is x where x is positive and zero before that, so each term "
        "contributes nothing until its boundary. "
        + ("The bracket is multiplied by the confinement factor, the cell "
           f"running out of room to spread, measured here as q = {q:g}. "
           if q else "")
        + ("Boundaries: " + ", ".join(boundaries) + ". " if boundaries else "")
        + f"Fitted over ε = {lo:.3f} to {hi:.3f}."
    )
    rows = []
    for piece in pieces:
        key, unit_name, std_key = MODULUS_FIELDS[piece["term"]]
        value = fit.get(key)
        error = fit.get(std_key)
        relative = _relative_error(piece, fit)
        coefficient = piece["coefficient_N"] * factor
        # The moduli themselves are not repeated here: they are quoted with
        # their uncertainties in the results above. What this table is for
        # is the arithmetic between a modulus and the number in front of the
        # shape of ε, which is nowhere else.
        rows.append({
            "element": piece["name"],
            "prefactor (N/Pa)": f"{piece['prefactor']:.5g}",
            f"coefficient ({unit_label})": (
                f"{coefficient:.5g}"
                + (f" ± {coefficient * relative:.2g}"
                   if np.isfinite(relative) else "")
            ),
            "± (%)": (f"{relative * 100:.1f}%"
                      if np.isfinite(relative) else "n/a"),
        })
    flat_table(
        pd.DataFrame(rows),
        align_right=["prefactor (N/Pa)", f"coefficient ({unit_label})",
                     "± (%)"],
        caption="Each coefficient is that element's geometric prefactor "
                "times its modulus. The prefactor is fixed before fitting, "
                "so the coefficient carries exactly the fractional "
                "uncertainty of the modulus, and the ± on both is the "
                "standard error from the covariance of the fit, σ²(XᵀWX)⁻¹.",
    )
    worst = fit.get("worst_pair")
    correlation = abs(fit.get("worst_correlation", 0.0) or 0.0)
    if worst and correlation > 0.97:
        st.caption(
            f"⚠️ Over this range the {term_name(worst[0]).lower()} and "
            f"{term_name(worst[1]).lower()} basis functions have correlation "
            f"{correlation:.4f}. Their sum is well determined; the split "
            f"between them is only as good as that number is below 1, which "
            f"is what the ± above is measuring. Widening the range, "
            f"especially towards ε = 0, is what separates them."
        )
    # R², χ²/dof and the RMSE are on the tiles above; what is not up there
    # is how many parameters bought them and how well conditioned the
    # design was.
    st.caption(
        f"{int(fit.get('n_points', 0))} points"
        + f" · {int(fit.get('n_params', 0))} free parameters"
        + f" · weighted {fit.get('weighting', 'uniform')}"
        + (f" · κ(X) = {float(fit['condition_number']):.4g}"
           if np.isfinite(fit.get("condition_number", np.nan)) else "")
        + (f" · worst basis correlation ρ = {correlation:.4f}"
           if correlation else "")
        + "."
    )
    copy_the_results(fit, unit)


def element_support(term, fit):
    """
    Where this element carries load, said so it cannot be misread.

    The upper edge of an element's window is where it stops taking *more*
    load, not where its contribution ends: past it the element holds what it
    reached, so the force it carries continues to the end of the curve. A
    bare interval said the opposite -- "membrane 0.000 to 0.277" reads as a
    sarcolemma that stops carrying at 0.277, which is not what the model
    does and not what a sarcolemma does.
    """
    if not (fit and fit.get("success")):
        return ""
    spans = (fit.get("piecewise") or {}).get("spans") or {}
    if term not in (fit.get("terms") or ()) and term not in spans:
        return "not in this model"
    window = spans.get(term) or (fit.get("term_windows") or {}).get(term)
    lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
    if not window:
        return f"from {eps_text(lo, fit)}, stiffening to the end"
    a, b = float(window[0]), float(window[1])
    if term == "lamina":
        return f"lump on [{eps_text(a, fit)}, {eps_text(b, fit)}]"
    if b >= hi - 1e-6:
        return f"from {eps_text(a, fit)}, stiffening to the end"
    return f"from {eps_text(a, fit)}, holding from {eps_text(b, fit, bare=True)}"


def eps_text(value, fit=None, bare=False):
    """
    A deformation written the way the page it is on writes it.

    The four-regime page works in percent and the others in ε as a
    fraction; the same value is written the same way on the inputs, on the
    curve, in the results and in this sentence, so it reads as one number.
    """
    value = float(value)
    if fit is not None and fit.get("coupling") == "piecewise":
        return f"{100.0 * value:.1f} %"
    return f"{value:.3f}" if bare else f"ε {value:.3f}"


def copy_the_results(fit, unit="nN"):
    """
    The whole fit as one block to copy into a spreadsheet.

    Two tab-separated lines, headers then values, which is exactly what a
    spreadsheet expects from the clipboard: paste it into A1 and it lands
    one number per cell. Every number the fit produced is here, each beside
    its own uncertainty, so a cell can be added to a growing table without
    anybody retyping a modulus and losing a digit.
    """
    record = fit_record(fit, unit)
    if not record:
        return
    copy_block(record, key="download_fit_record",
               file_stem=st.session_state.get("cell_name", "cell") or "cell",
               summary=(f"fit {record['Fit ID']} · R² = "
                        f"{float(fit.get('r_squared', float('nan'))):.5f}"),
               lab=lab_sheet_row(fit))


def copy_block(record, key, file_stem, summary="", lab=None):
    """The lab's row to paste, the whole record under it, and both as CSV."""
    with st.expander("📋 Copy these results into a spreadsheet"
                     + (f" · {summary}" if summary else ""), expanded=False):
        if lab:
            st.markdown("**The lab table** — paste straight into the sheet")
            st.code(_tsv(lab), language="text")
            st.caption(
                "The thirteen columns of the shared sheet, in its order and "
                "written the way it writes them. Copy both lines, click the "
                "first empty cell of the sheet and paste: the headers land "
                "on the header row and the values under them. Pasting only "
                "the second line adds a row to a sheet that already has its "
                "headers."
            )
            st.download_button(
                "⬇️ The lab row as a CSV",
                data=_csv(lab),
                file_name=f"{file_stem}_row.csv",
                mime="text/csv",
                key=f"{key}_lab",
                **STRETCH,
            )
            st.markdown("**Every number this fit produced**")
        st.code(_tsv(record), language="text")
        st.caption(
            "Tab separated, headers on the first line. Copy both lines, "
            "click a cell in Excel and paste: each value lands in its own "
            "column, in the same order every time, so cell after cell "
            "stacks into one table. Uncertainties are the ± columns and the "
            "moduli are plain numbers, with the units in the headings, so "
            "they can be averaged as they are."
        )
        st.download_button(
            "⬇️ Or download it as a CSV",
            data=_csv(record),
            file_name=f"{file_stem}_fit.csv",
            mime="text/csv",
            key=key,
            **STRETCH,
        )


def settings_from_hypothesis(winner, epsilon_max=None, q=None):
    """
    Turn a winning picture of the cell into the widget values that make it.

    Written as staged settings rather than applied directly, because these
    are widget keys and Streamlit rejects a write to one after its widget
    exists. The caller reruns and they land at the top of the next pass.
    """
    pending = {
        "model_kind": "Segmented (each part takes over in turn)",
        "membrane_after_break": next(
            k for k, v in MEMBRANE_CHOICES.items() if v == winner["membrane"]
        ),
        "cyto_starts_at": next(
            k for k, v in CYTO_CHOICES.items() if v == winner["cyto_start"]
        ),
        "segment_break_1": round(float(winner["break_1"]), 4),
        "segment_break_2": round(float(winner["break_2"]), 4),
    }
    for term in ALL_TERMS:
        pending[f"use_{term}"] = term in winner["terms"]
    if epsilon_max is not None:
        pending["window_end"] = round(float(epsilon_max), 4)
    if q is not None:
        pending["confinement"] = round(float(q), 2)
    return pending


def analyse_curve(model, lo, hi, picks, weighting, measure_q, terms_hint,
                  n_grid=8, cv_repeats=2, passes=2, band_1=None, band_2=None):
    """
    The one fitting routine. Everything that fits a curve comes through here.

    There used to be two: the curve was read one way when it loaded and
    another way when the button was pressed, they used different candidate
    sets, and on a real cardiomyocyte they disagreed by two orders of
    magnitude in chi-squared. Two answers on one page is not a feature, it
    is a bug that looks like a feature, so there is one routine and the
    button is a harder setting of it.

    What it does, in the order the quantities depend on each other:

    1. Every picture of the cell is fitted with its own boundaries **and its
       own confinement q**. q multiplies every basis function, so a picture
       fitted at another picture's q is not that picture; profiling it per
       candidate is also what stops the answer depending on which was fitted
       first.
    2. They are scored on points they were not fitted to, and the winner
       brings its q back with it.

    What it does not do is change the mixture. Every candidate carries the
    components that are ticked, so this decides the arrangement and the
    boundaries and nothing else.
    """
    out = {"q": None, "q_scan": None, "hypotheses": None, "components": None}
    # The deep boundary is searched inside the band this cell type is known
    # to put it in, so the fit button and the refine button cannot disagree
    # about where a nucleus can be met.
    if band_2 is None:
        band_2 = deep_onset_band(st.session_state.get("cell_type"), lo, hi)
        if band_2 == (float(lo), float(hi)):
            band_2 = None
    try:
        chosen = compare_hypotheses(
            model, lo, hi, picks, weighting=weighting, cv_repeats=cv_repeats,
            n_grid=n_grid, refine_rounds=max(1, int(passes) - 1),
            scan_q=bool(measure_q), band_1=band_1, band_2=band_2,
        )
    except Exception:  # pragma: no cover - defensive
        return out
    if not chosen.get("success"):
        return out

    out["hypotheses"] = chosen
    best = chosen["best"]
    e1, e2 = float(best["break_1"]), float(best["break_2"])
    if measure_q:
        out["q"] = float(best.get("confinement", model.confinement))
        model.confinement = out["q"]
        # One last pass over q, now at the boundaries the winner actually
        # chose rather than the rough ones it was profiled at. This is the
        # closing step of the same coordinate descent and it is the reason
        # the number the page reports is the number that fits: on the WT
        # curves it moves q by a few tenths and χ²/dof by a third. It also
        # produces the profile the page needs to say how well q is
        # determined, rather than only what it came out at.
        if hasattr(model, "scan_confinement"):
            try:
                again = model.scan_confinement(
                    lo, hi, e1=e1, e2=e2, membrane=best["membrane"],
                    cyto_start=best["cyto_start"],
                    use_nucleus="nucleus" in best["terms"],
                    use_tension="tension" in best["terms"],
                    use_nucleus_shell="nucleus_shell" in best["terms"],
                    use_cortex="cortex" in best["terms"],
                    weighting=weighting,
                )
            except Exception:  # pragma: no cover - defensive
                again = None
            if again and again.get("success"):
                out["q_scan"] = again
                out["q"] = float(again["q"])
                model.confinement = out["q"]
                # And the boundaries once more at that q, because they were
                # chosen at the old one.
                refit = model._best_breakpoints(
                    lo, hi, best["membrane"], best["cyto_start"],
                    "nucleus" in best["terms"], weighting, n_grid,
                    max(1, int(passes) - 1),
                    use_tension="tension" in best["terms"],
                    # Every term the winner carries, or this last pass
                    # places the boundaries for a different model from the
                    # one the page then fits at them.
                    use_nucleus_shell="nucleus_shell" in best["terms"],
                    use_cortex="cortex" in best["terms"],
                    band_1=band_1, band_2=band_2,
                )
                if refit and refit.get("success"):
                    e1, e2 = float(refit["break_1"]), float(refit["break_2"])
                    best["break_1"], best["break_2"] = e1, e2
                    best["r_squared"] = refit["r_squared"]
                    best["fit"] = refit

    # Which components belong in the model is not asked here. It is its own
    # question with its own criterion, it belongs to the search button, and
    # asking it as part of every fit meant a fit could silently untick a box
    # while the person watched the curve.
    return out


def settings_from_arrangement(best, epsilon_max):
    """
    Turn the winning arrangement into the widget values that reproduce it.

    Written as staged settings rather than applied directly, because these
    are widget keys and Streamlit rejects a write to one after its widget
    exists. The caller reruns and they land at the top of the next pass.
    """
    pending = {"window_end": round(float(epsilon_max), 4)}
    arrangement = best.get("arrangement")

    if arrangement == "segmented":
        composition = best.get("composition") or {}
        labels = {
            (MEMBRANE_CHOICES[m], CYTO_CHOICES[c]): (m, c)
            for m in MEMBRANE_CHOICES for c in CYTO_CHOICES
        }
        key = (composition.get("membrane", "freeze"),
               composition.get("cyto_start", "break"))
        membrane_label, cyto_label = labels.get(
            key, ("holds what it reached", "at ε₁")
        )
        pending.update(
            {
                "model_kind": "Segmented (each part takes over in turn)",
                "segment_break_1": round(float(composition.get("break_1", 0.15)), 3),
                "segment_break_2": round(float(composition.get("break_2", 0.40)), 3),
                "membrane_after_break": membrane_label,
                "cyto_starts_at": cyto_label,
                "use_nucleus": bool(composition.get("use_nucleus", True)),
                "use_tension": bool(composition.get("use_tension", False)),
            }
        )
    elif arrangement == "series":
        pending["model_kind"] = "Stacked (elements in line)"
    else:
        pending["model_kind"] = "Side by side (every element acts everywhere)"
    return pending


# What each material is, mechanically, and the law that follows from it.
# One row per term, and the last column is the one that matters: a fit can
# only separate materials whose force laws differ in shape or in when they
# start. Everything about how this model is built follows from that line.
MATERIAL_LAWS = {
    "tension": {
        "role": "a network already under tension, resisting the area the "
                "cell must gain as it flattens",
        "energy": "T₀·ΔA, and the area gained goes as ε²",
        "law": "F = A_t·T₀·ε",
        "exponent": "1",
        "separable": "the only term linear in ε, so it dominates at first "
                     "contact where the powers have all but vanished",
    },
    "membrane": {
        "role": "a thin shell resisting being stretched, the balloon",
        "energy": "elastic energy in the area strain, which goes as ε²",
        "law": "F = Aₘ·Eₘ·ε³",
        "exponent": "3",
        "separable": "the steepest law here, so it is negligible at contact "
                     "and dominant deep in",
    },
    "cortex": {
        "role": "the layer welded under the membrane, squeezed from the "
                "moment the plates touch the cell",
        "energy": "Hertz contact between a sphere and a plane",
        "law": "F = Aᵢ·E_cx·ε³ᐟ²",
        "exponent": "3/2",
        "separable": "the same law as the scaffolding under it, so **only** "
                     "the onset separates them: the cortex is loaded from "
                     "ε = 0 and the scaffolding from ε₁",
    },
    "interior": {
        "role": "the material filling the cell, squeezed between two flat "
                "plates: a Hertzian contact",
        "energy": "Hertz contact between a sphere and a plane",
        "law": "F = Aᵢ·E_c·⟨ε − s⟩³ᐟ²",
        "exponent": "3/2",
        "separable": "between the other two in steepness, which is why a "
                     "curve starting near 1.7 means this and the shell "
                     "together",
    },
    "nucleus_shell": {
        "role": "the envelope around the nucleus: a shell resisting being "
                "stretched, met only once the plates reach it",
        "energy": "elastic energy in the envelope's area strain",
        "law": "F = A_ne·E_ne·⟨ε − ε₂⟩³",
        "exponent": "3",
        "separable": "shares its onset with what it contains, so the "
                     "exponents have to differ, and they do: 3 against 3/2, "
                     "the same pairing as the cell's own membrane and "
                     "cytoskeleton",
    },
    "nucleus": {
        "role": "what the envelope contains, squeezed like any elastic "
                "filling once the plates reach it",
        "energy": "the same Hertzian contact, met later",
        "law": "F = A_n·E_n·⟨ε − ε₂⟩³ᐟ²",
        "exponent": "3/2",
        "separable": "the same law as the cell's interior, so **only** its "
                     "onset ε₂ tells them apart: no onset, no separation",
    },
}


# Where each material's number lives in a fit, and what it is called.
# One table, so that adding a material means adding a row here rather than
# remembering six places that enumerate moduli by hand. Forgetting one of
# those places is how a fitted modulus ends up invisible on the page.
MODULUS_FIELDS = {
    "tension": ("T0_mN_m", "mN/m", "T0_mN_m_std"),
    "membrane": ("Em_MPa", "MPa", "Em_MPa_std"),
    "cortex": ("Ecx_kPa", "kPa", "Ecx_kPa_std"),
    "interior": ("Ei_kPa", "kPa", "Ei_kPa_std"),
    "nucleus_shell": ("Ene_MPa", "MPa", "Ene_MPa_std"),
    "nucleus": ("En_kPa", "kPa", "En_kPa_std"),
}

# Pascals per reported unit, for turning a reported number back into SI.
MODULUS_SCALE = {"MPa": 1e6, "kPa": 1e3}


def materials_table(terms, membrane_mode="continue", cyto_start="zero",
                    e1=None, e2=None, caption=None):
    """
    The materials in this fit, the law each obeys, and when it engages.

    The point of the table is the last two columns. Where a material starts
    and how steeply it rises are the only two things that let a fit tell it
    from another one, so they are what a reader should be looking at when
    deciding whether to believe a modulus.
    """
    if not terms:
        return
    rows = []
    for term in ALL_TERMS:
        if term not in terms:
            continue
        law = MATERIAL_LAWS[term]
        if term == "tension":
            when = ("from first contact" if membrane_mode != "late"
                    else f"from first contact (ε₁ = {e1:.3f} for the shell)"
                    if e1 is not None else "from first contact")
        elif term == "membrane":
            when = (
                f"from ε₁ = {e1:.3f}" if membrane_mode == "late" and e1 is not None
                else "from ε₁" if membrane_mode == "late"
                else "from first contact"
            )
            if membrane_mode == "freeze":
                when += (f", frozen after ε₁ = {e1:.3f}" if e1 is not None
                         else ", frozen after ε₁")
        elif term == "interior":
            when = (
                "from first contact" if cyto_start == "zero"
                else f"from ε₁ = {e1:.3f}" if e1 is not None else "from ε₁"
            )
        elif term == "cortex":
            when = "from first contact"
        elif term == "nucleus_shell":
            when = (f"from ε₂ = {e2:.3f}, with what it contains"
                    if e2 is not None else "from ε₂, with what it contains")
        else:
            when = f"from ε₂ = {e2:.3f}" if e2 is not None else "from ε₂"
        rows.append({
            "Material": plain_name(term),
            "Symbol": TERM_SYMBOLS.get(term, term),
            "What it is, mechanically": law["role"],
            "Force law": law["law"],
            "Rises as": f"ε^{law['exponent']}",
            "Carries load": when,
            "What makes it separable": law["separable"],
        })
    flat_table(
        pd.DataFrame(rows),
        align_right=["Rises as"],
        caption=caption,
    )


def separation_rule(q=None):
    """The one sentence the whole model rests on, said plainly."""
    st.info(
        "**How materials are separated.** A fit cannot see materials. It "
        "sees one curve, and it can only split that curve between two terms "
        "if those terms have **different shapes**: a different power of ε, "
        "or a different deformation at which they start. Two terms with the "
        "same power and the same start are one material wearing two names, "
        "and the solver will divide them arbitrarily, giving two numbers "
        "that wander from cell to cell while their sum stays put.\n\n"
        "That rule is why this model looks the way it does. The membrane's "
        "two springs are given the two laws a taut shell really has, ε and "
        "ε³, rather than two cube laws. Where two networks share the "
        "Hertzian 3/2, one of them has to start later or they cannot be "
        "told apart. And nothing is added because it exists in the cell; it "
        "is added only if the curve has a shape for it."
        + (
            f"\n\nThe confinement (1−ε)^−q, here q = {float(q):.2f}, is not "
            f"a material and is not fitted as one. It multiplies every term, "
            f"because it is the cell running out of room rather than "
            f"anything pushing back."
            if q is not None else ""
        ),
        icon="🔍",
    )


def numeric_column(frame, column):
    """
    One column of a table as numbers, and an empty series when it is absent.

    A spreadsheet somebody keeps by hand does not always have the column the
    app is looking for. `frame.get(name)` answers None for a missing one,
    and `pd.to_numeric(None).dropna()` is an AttributeError that takes the
    whole page down while naming neither the column nor the sheet.
    """
    if frame is None or column not in getattr(frame, "columns", []):
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return values


def safe_frame(frame):
    """
    A table the widget can actually draw.

    Streamlit hands a DataFrame to Arrow, and Arrow refuses two columns with
    the same name — with an error that names none of them and takes the
    whole page down. A spreadsheet somebody keeps by hand acquires repeats
    and blank headings, so anything read from one comes through here first.
    Later copies are numbered rather than dropped: a column with data in it
    is somebody's data even when its name is a mistake.
    """
    if frame is None or not hasattr(frame, "columns"):
        return frame
    seen, names = {}, []
    for index, raw in enumerate(list(frame.columns)):
        name = str(raw).strip() or f"Column {index + 1}"
        while name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        seen[name] = 1
        names.append(name)
    if names == list(frame.columns):
        return frame
    out = frame.copy()
    out.columns = names
    return out


def sub_panel(title, flat=False, expanded=False):
    """
    An expander, or a bold line when one is already open around it.

    Streamlit refuses an expander inside an expander, and in guided mode
    every setting lives inside one. Used with ``with``, so the body reads
    the same either way.
    """
    if not flat:
        return st.expander(title, expanded=expanded)
    st.markdown(f"**{title}**")
    return st.container()


def open_panel(title, guided, expanded=False, parent=None, flat=False):
    """
    Start a section: a collapsed expander in Guided mode, a heading otherwise.

    Entered and exited by hand rather than with a `with` block, so the long
    bodies of these sections keep their current indentation. Streamlit
    containers support the context-manager protocol either way.

    ``parent`` puts the panel inside a container staked out earlier in the
    page. Streamlit draws things where they were created, so this is how a
    panel whose code has to run late can still appear early: the controls
    that change the fit belong under the button that runs it, not below the
    curve they changed.
    """
    if flat:
        # Already inside an expander, and Streamlit will not nest them. A
        # bold line does the same job of separating one group from the next.
        st.markdown(f"**{title}**")
        panel = st.container()
        panel.__enter__()
        return panel
    if guided:
        panel = (parent or st).expander(title, expanded=expanded)
        panel.__enter__()
        return panel
    section(title)
    panel = st.container()
    panel.__enter__()
    return panel


def close_panel(panel):
    panel.__exit__(None, None, None)


def stiffness_in_words(value_pa):
    """
    An everyday comparison for a modulus.

    Someone who does not work in mechanics has no feel for a pascal. The
    comparisons are order-of-magnitude only and hedged as such, because a
    cell is not a rubber band and the point is scale, not identity.
    """
    if not np.isfinite(value_pa) or value_pa <= 0:
        return "not measurable from this curve"
    if value_pa < 3e2:
        return "softer than loose jelly"
    if value_pa < 3e3:
        return "about as soft as a set jelly"
    if value_pa < 3e4:
        return "about as firm as a gummy sweet"
    if value_pa < 3e5:
        return "about as firm as a soft eraser"
    if value_pa < 5e6:
        return "about as firm as a rubber band"
    return "stiffer than rubber, which is unusual for a cell"


def quality_in_words(r_squared):
    """What R² means, without saying R²."""
    if not np.isfinite(r_squared):
        return "could not be judged"
    if r_squared > 0.995:
        return "the line goes through the points almost exactly"
    if r_squared > 0.98:
        return "the line follows the points closely"
    if r_squared > 0.9:
        return "the line follows the general shape but misses in places"
    return "the line does not really follow the data; something is wrong"


def range_maths(lo, hi, terms=None):
    """
    What the chosen range does to the fit, written as maths.

    A range is not only a pair of numbers: it decides which points enter
    the least-squares sum, and it decides the stretch of deformation each
    material is actually measured over, which is not the same interval for
    each of them. Printed here, under the control that sets it, so moving
    a handle and watching these change is one gesture.
    """
    terms = tuple(terms or active_terms())
    e1 = float(st.session_state["segment_break_1"])
    e2 = float(st.session_state["segment_break_2"])
    q = float(st.session_state.get("confinement", 0.0) or 0.0)
    membrane = MEMBRANE_CHOICES.get(
        st.session_state["membrane_after_break"], "freeze"
    )
    from_break = CYTO_CHOICES.get(
        st.session_state["cyto_starts_at"], "break"
    ) == "break"

    st.latex(
        r"\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta} \ge 0}"
        r"\sum_{i:\ " + f"{lo:.3f}" + r" \le \varepsilon_i \le "
        + f"{hi:.3f}" + r"} w_i \left( F_i - F(\varepsilon_i) \right)^{2}"
    )
    st.caption(
        f"Only points inside the range enter the sum, so the range is part "
        f"of the model, not a view of it. Moving either end changes every "
        f"modulus."
        + (f" Every term is multiplied by (1−ε)^−{q:g} across it."
           if q else "")
    )

    # Which stretch each material is actually measured over. Two materials
    # can share the range and still be measured on different parts of it,
    # and that is the whole of how a fit tells them apart.
    rows = []
    for term in ALL_TERMS:
        if term not in terms:
            continue
        if term in ("tension", "membrane"):
            begin, end = lo, (hi if membrane == "continue" else min(e1, hi))
            basis = (r"min(ε, ε₁)" if membrane == "freeze"
                     else "⟨ε − ε₁⟩" if membrane == "late" else "ε")
            basis += "³" if term == "membrane" else ""
        elif term == "cortex":
            begin, end, basis = lo, hi, "ε³ᐟ²"
        elif term == "interior":
            begin = max(lo, e1) if from_break else lo
            end, basis = hi, ("⟨ε − ε₁⟩³ᐟ²" if from_break else "ε³ᐟ²")
        else:
            begin, end = max(lo, e2), hi
            basis = ("⟨ε − ε₂⟩³" if term == "nucleus_shell"
                     else "⟨ε − ε₂⟩³ᐟ²")
        points = 0
        data = (st.session_state.get("data") or {}).get("epsilon")
        if data is not None and np.size(data):
            data = np.asarray(data, dtype=float)
            points = int(((data >= begin) & (data <= end)).sum())
        rows.append({
            "material": plain_name(term),
            "measured over ε": f"{begin:.3f} to {max(end, begin):.3f}",
            "points there": points,
            "what it multiplies": basis,
        })
    if rows:
        flat_table(
            pd.DataFrame(rows),
            align_right=["measured over ε", "points there"],
            caption="A material is only measured where its own term is "
                    "non-zero. Two of them sharing an interval and a power "
                    "of ε cannot be told apart at all, however wide the "
                    "range is made.",
        )


def power_law_notes(fit, model=None):
    """
    What each material's exponent means, and how to check it by hand.

    Every term here is one power of ε. That exponent is the part of the
    model a person can hold it to without trusting any of the fitting: the
    slope of their own curve on log-log paper near contact should land
    between the smallest and the largest listed, and if it does not, the
    materials in the model are not the materials in the cell.
    """
    terms = set(fit.get("terms") or ())
    if not terms:
        return
    q = float(fit.get("confinement", 0.0) or 0.0)
    rows = []
    for term, symbol, power, shape in (
        ("tension", "T₀", 1.0, "A_t T₀ ε"),
        ("membrane", "Eₘ", 3.0, "A_m Eₘ ε³"),
        ("cortex", "E_cx", 1.5, "A_i E_cx ε³ᐟ²"),
        ("interior", "Ec", 1.5, "A_i Ec ⟨ε − ε₁⟩³ᐟ²"),
        ("nucleus_shell", "E_ne", 3.0, "A_ne E_ne ⟨ε − ε₂⟩³"),
        ("nucleus", "Eₙ", 1.5, "A_n Eₙ ⟨ε − ε₂⟩³ᐟ²"),
    ):
        if term not in terms:
            continue
        rows.append({
            "material": plain_name(term),
            "contributes": shape + (f" × (1−ε)^−{q:g}" if q else ""),
            "slope near contact": f"{power:g}",
            "slope at ε = 0.6": (
                f"{power + q * 0.6 / 0.4:.2f}" if q else f"{power:g}"
            ),
        })
    if not rows:
        return
    flat_table(
        pd.DataFrame(rows),
        align_right=["slope near contact", "slope at ε = 0.6"],
    )
    st.caption(
        "The slope is d(ln F)/d(ln ε), which is what you read off a log-log "
        "plot, and it is the part of this you can check by hand: measure "
        "the slope of your own curve near contact and it should land "
        "between the smallest and the largest listed here. 3 is a membrane "
        "on its own, 3/2 anything Hertzian, and above 3 is the cell running "
        "out of room. "
        + (f"Confinement adds qε/(1−ε) to every one of them, and here "
           f"q = {q:g}, which is why the measured slope climbs along the "
           f"curve instead of sitting on a constant."
           if q else "Here q = 0, so every slope is constant.")
    )
    separation_rule(q=q if q else None)
    if model is not None and hasattr(model, "local_exponent"):
        try:
            grid, slope = model.local_exponent(window_frac=0.18)
            good = np.isfinite(slope)
            if good.sum() > 4:
                st.caption(
                    "Measured on this curve: slope "
                    f"{float(slope[good][0]):.2f} near contact, "
                    f"{float(slope[good][-1]):.2f} at the far end. If those "
                    "sit outside the table above, the model has materials "
                    "this cell does not."
                )
        except Exception:  # pragma: no cover - a curve too short to differentiate
            pass


def stiffness_table(fit):
    """
    What each material turned out to be, in its own units and in words.

    All that is left of what used to be a retelling of the fit. The story
    of the compression restated the boundaries already drawn on the curve,
    and the paragraph after it restated R² and chi-squared, which are two
    numbers standing beside it. This table is the part that said something
    the numbers did not: what the material is, and what that stiffness is
    like to hold.
    """
    if not (fit and fit.get("success")):
        return

    terms = set(fit.get("terms") or ())
    st.markdown("##### How stiff each material turned out to be")
    names = components_for(st.session_state["cell_type"])
    rows = []
    # The tension row is a tension, in newtons per metre, not a modulus. It
    # is listed with the others because it is one of the springs, and given
    # its own units rather than being dressed up as a stiffness it is not.
    coat_m = float(st.session_state["protein_coat_nm"]) * 1e-9
    for term in ALL_TERMS:
        if term not in terms_for(st.session_state["cell_type"]):
            continue
        key, unit, _ = MODULUS_FIELDS[term]
        pa_factor = (
            1e-3 / max(coat_m, 1e-12) if term == "tension"
            else MODULUS_SCALE[unit]
        )
        label, everyday = names[term]
        used = term in terms
        value = float(fit.get(key, 0.0)) if used else 0.0
        rows.append(
            {
                "Part of the cell": label,
                "What it is": everyday,
                "Stiffness": f"{value:.3g} {unit}" if used else f"0 {unit}",
                "Roughly": (
                    "not included in this model" if not used
                    else "the data did not need it" if value <= 0
                    # A tension divided by the layer it sits in is a modulus,
                    # which is the only way to put it on the same scale as
                    # the rest.
                    else stiffness_in_words(value * pa_factor)
                ),
            }
        )
    flat_table(pd.DataFrame(rows), align_right=["Stiffness"])
    if "tension" in terms_for(st.session_state["cell_type"]):
        st.caption(
            "The protein network is quoted as a tension, in mN/m, because "
            "that is what the curve measures: a taut sheet answers with a "
            "force per unit length, and turning it into a stiffness needs an "
            "assumed layer thickness (set in the sidebar). The everyday "
            "comparison in the last column does make that conversion."
        )


def range_maths(lo, hi, terms=None):
    """
    What the chosen range does to the fit, written as maths.

    A range is not only a pair of numbers: it decides which points enter
    the least-squares sum, and it decides the stretch of deformation each
    material is actually measured over, which is not the same interval for
    each of them. Printed here, under the control that sets it, so moving
    a handle and watching these change is one gesture.
    """
    terms = tuple(terms or active_terms())
    e1 = float(st.session_state["segment_break_1"])
    e2 = float(st.session_state["segment_break_2"])
    q = float(st.session_state.get("confinement", 0.0) or 0.0)
    membrane = MEMBRANE_CHOICES.get(
        st.session_state["membrane_after_break"], "freeze"
    )
    from_break = CYTO_CHOICES.get(
        st.session_state["cyto_starts_at"], "break"
    ) == "break"

    st.latex(
        r"\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta} \ge 0}"
        r"\sum_{i:\ " + f"{lo:.3f}" + r" \le \varepsilon_i \le "
        + f"{hi:.3f}" + r"} w_i \left( F_i - F(\varepsilon_i) \right)^{2}"
    )
    st.caption(
        f"Only points inside the range enter the sum, so the range is part "
        f"of the model, not a view of it. Moving either end changes every "
        f"modulus."
        + (f" Every term is multiplied by (1−ε)^−{q:g} across it."
           if q else "")
    )

    # Which stretch each material is actually measured over. Two materials
    # can share the range and still be measured on different parts of it,
    # and that is the whole of how a fit tells them apart.
    rows = []
    for term in ALL_TERMS:
        if term not in terms:
            continue
        if term in ("tension", "membrane"):
            begin, end = lo, (hi if membrane == "continue" else min(e1, hi))
            basis = (r"min(ε, ε₁)" if membrane == "freeze"
                     else "⟨ε − ε₁⟩" if membrane == "late" else "ε")
            basis += "³" if term == "membrane" else ""
        elif term == "cortex":
            begin, end, basis = lo, hi, "ε³ᐟ²"
        elif term == "interior":
            begin = max(lo, e1) if from_break else lo
            end, basis = hi, ("⟨ε − ε₁⟩³ᐟ²" if from_break else "ε³ᐟ²")
        else:
            begin, end = max(lo, e2), hi
            basis = ("⟨ε − ε₂⟩³" if term == "nucleus_shell"
                     else "⟨ε − ε₂⟩³ᐟ²")
        points = 0
        data = (st.session_state.get("data") or {}).get("epsilon")
        if data is not None and np.size(data):
            data = np.asarray(data, dtype=float)
            points = int(((data >= begin) & (data <= end)).sum())
        rows.append({
            "material": plain_name(term),
            "measured over ε": f"{begin:.3f} to {max(end, begin):.3f}",
            "points there": points,
            "what it multiplies": basis,
        })
    if rows:
        flat_table(
            pd.DataFrame(rows),
            align_right=["measured over ε", "points there"],
            caption="A material is only measured where its own term is "
                    "non-zero. Two of them sharing an interval and a power "
                    "of ε cannot be told apart at all, however wide the "
                    "range is made.",
        )


def power_law_notes(fit, model=None):
    """
    What each material's exponent means, and how to check it by hand.

    Every term here is one power of ε. That exponent is the part of the
    model a person can hold it to without trusting any of the fitting: the
    slope of their own curve on log-log paper near contact should land
    between the smallest and the largest listed, and if it does not, the
    materials in the model are not the materials in the cell.
    """
    terms = set(fit.get("terms") or ())
    if not terms:
        return
    q = float(fit.get("confinement", 0.0) or 0.0)
    rows = []
    for term, symbol, power, shape in (
        ("tension", "T₀", 1.0, "A_t T₀ ε"),
        ("membrane", "Eₘ", 3.0, "A_m Eₘ ε³"),
        ("cortex", "E_cx", 1.5, "A_i E_cx ε³ᐟ²"),
        ("interior", "Ec", 1.5, "A_i Ec ⟨ε − ε₁⟩³ᐟ²"),
        ("nucleus_shell", "E_ne", 3.0, "A_ne E_ne ⟨ε − ε₂⟩³"),
        ("nucleus", "Eₙ", 1.5, "A_n Eₙ ⟨ε − ε₂⟩³ᐟ²"),
    ):
        if term not in terms:
            continue
        rows.append({
            "material": plain_name(term),
            "contributes": shape + (f" × (1−ε)^−{q:g}" if q else ""),
            "slope near contact": f"{power:g}",
            "slope at ε = 0.6": (
                f"{power + q * 0.6 / 0.4:.2f}" if q else f"{power:g}"
            ),
        })
    if not rows:
        return
    flat_table(
        pd.DataFrame(rows),
        align_right=["slope near contact", "slope at ε = 0.6"],
    )
    st.caption(
        "The slope is d(ln F)/d(ln ε), which is what you read off a log-log "
        "plot, and it is the part of this you can check by hand: measure "
        "the slope of your own curve near contact and it should land "
        "between the smallest and the largest listed here. 3 is a membrane "
        "on its own, 3/2 anything Hertzian, and above 3 is the cell running "
        "out of room. "
        + (f"Confinement adds qε/(1−ε) to every one of them, and here "
           f"q = {q:g}, which is why the measured slope climbs along the "
           f"curve instead of sitting on a constant."
           if q else "Here q = 0, so every slope is constant.")
    )
    separation_rule(q=q if q else None)
    if model is not None and hasattr(model, "local_exponent"):
        try:
            grid, slope = model.local_exponent(window_frac=0.18)
            good = np.isfinite(slope)
            if good.sum() > 4:
                st.caption(
                    "Measured on this curve: slope "
                    f"{float(slope[good][0]):.2f} near contact, "
                    f"{float(slope[good][-1]):.2f} at the far end. If those "
                    "sit outside the table above, the model has materials "
                    "this cell does not."
                )
        except Exception:  # pragma: no cover - a curve too short to differentiate
            pass


def stiffness_table(fit):
    """
    What each material turned out to be, in its own units and in words.

    All that is left of what used to be a retelling of the fit. The story
    of the compression restated the boundaries already drawn on the curve,
    and the paragraph after it restated R² and chi-squared, which are two
    numbers standing beside it. This table is the part that said something
    the numbers did not: what the material is, and what that stiffness is
    like to hold.
    """
    if not (fit and fit.get("success")):
        return

    terms = set(fit.get("terms") or ())
    st.markdown("##### How stiff each material turned out to be")
    names = components_for(st.session_state["cell_type"])
    rows = []
    # The tension row is a tension, in newtons per metre, not a modulus. It
    # is listed with the others because it is one of the springs, and given
    # its own units rather than being dressed up as a stiffness it is not.
    coat_m = float(st.session_state["protein_coat_nm"]) * 1e-9
    for term in ALL_TERMS:
        if term not in terms_for(st.session_state["cell_type"]):
            continue
        key, unit, _ = MODULUS_FIELDS[term]
        pa_factor = (
            1e-3 / max(coat_m, 1e-12) if term == "tension"
            else MODULUS_SCALE[unit]
        )
        label, everyday = names[term]
        used = term in terms
        value = float(fit.get(key, 0.0)) if used else 0.0
        rows.append(
            {
                "Part of the cell": label,
                "What it is": everyday,
                "Stiffness": f"{value:.3g} {unit}" if used else f"0 {unit}",
                "Roughly": (
                    "not included in this model" if not used
                    else "the data did not need it" if value <= 0
                    # A tension divided by the layer it sits in is a modulus,
                    # which is the only way to put it on the same scale as
                    # the rest.
                    else stiffness_in_words(value * pa_factor)
                ),
            }
        )
    flat_table(pd.DataFrame(rows), align_right=["Stiffness"])
    if "tension" in terms_for(st.session_state["cell_type"]):
        st.caption(
            "The protein network is quoted as a tension, in mN/m, because "
            "that is what the curve measures: a taut sheet answers with a "
            "force per unit length, and turning it into a stiffness needs an "
            "assumed layer thickness (set in the sidebar). The everyday "
            "comparison in the last column does make that conversion."
        )


def final_summary(fit, model, date_acquired=None):
    """
    The last thing on the page: this cell in one block, ready to be quoted.

    Everything above it is how the answer was arrived at. This is the
    answer, in the form somebody would paste into a lab book or read out
    to a supervisor: which cell, over what stretch of curve, under which
    picture, what each material came out at, and how well it fitted.
    """
    if not (fit and fit.get("success")):
        return
    terms = tuple(fit.get("terms") or ())
    lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
    q = float(fit.get("confinement", 0.0) or 0.0)
    chi = float(fit.get("chi_squared_reduced", float("nan")))
    r2 = float(fit.get("r_squared", float("nan")))

    st.markdown("#### In one block")
    name = st.session_state.get("cell_name") or "this cell"
    head = f"**{name}**"
    if date_acquired is not None:
        head += f" · {date_acquired}"
    head += (
        f" · {st.session_state['cell_type']}"
        + (" · chemically fixed" if fixed_cell_on() else "")
    )
    st.markdown(head)

    lines = [
        f"Fitted over ε = {lo:.3f} to {hi:.3f} "
        f"({int(fit.get('n_points', 0))} points), "
        f"weighted {fit.get('weighting', 'uniform')}"
        + (f", confinement q = {q:.2f}" if q else "")
        + ".",
    ]
    boundaries = []
    if fit.get("break_1") is not None:
        boundaries.append(f"ε₁ = {float(fit['break_1']):.3f}")
    if fit.get("break_2") is not None and any(
        t in terms for t in ("nucleus", "nucleus_shell")
    ):
        boundaries.append(f"ε₂ = {float(fit['break_2']):.3f}")
    if boundaries:
        lines.append("Boundaries: " + ", ".join(boundaries) + ".")
    for term in ALL_TERMS:
        if term not in terms:
            continue
        key, unit, std_key = MODULUS_FIELDS[term]
        value = fit.get(key)
        error = fit.get(std_key)
        if value is None:
            continue
        piece = f"{plain_name(term)}: {float(value):.4g} {unit}"
        if error is not None and np.isfinite(float(error)):
            piece += f" ± {float(error):.2g}"
        lines.append(piece + ".")
    lines.append(
        f"R² = {r2:.5f}"
        + (f", χ²/dof = {chi:.3g}" if np.isfinite(chi) else "")
        + "."
    )
    for line in lines:
        st.markdown(f"- {line}")

    # One copyable line, because a lab book is not a screenshot.
    quoted = "; ".join(
        f"{plain_name(term)} {float(fit.get(MODULUS_FIELDS[term][0], 0.0)):.4g} "
        f"{MODULUS_FIELDS[term][1]}"
        for term in ALL_TERMS if term in terms
    )
    st.code(
        f"{name}\t{lo:.3f}-{hi:.3f}\t{quoted}\tR2={r2:.5f}"
        + (f"\tchi2/dof={chi:.3g}" if np.isfinite(chi) else ""),
        language="text",
    )
    if np.isfinite(chi) and chi > 5 and fit.get("weighting") in (None, "noise"):
        st.caption(
            f"χ²/dof of {chi:.3g} under noise weighting means the line "
            f"misses the points by about {np.sqrt(chi):.0f} times the "
            f"scatter in the measurement, which is usually a missing "
            f"material or a boundary in the wrong place."
        )


def plot_option_controls():
    """
    The switches that change what is drawn on the curve.

    These live directly under the plot, not in the sidebar. They are
    about the figure in front of you, and a control for the thing you
    are looking at belongs next to it: in a sidebar expander nobody
    finds it, and the plot keeps its markings while the person hunts.
    """
    # Two extras and no more. The data and the fitted curve are always
    # drawn -- a plot without them is not a plot of anything -- and
    # everything else that used to be a switch here is sent to the figure
    # from the place that computed it, which leaves a row saying so.
    st.markdown("**Extras drawn on the curve**")
    st.caption(
        "The measured points and the fitted curve are always on. Anything "
        "else comes from a 📤 Send to plot button and is listed under the "
        "figure."
    )
    st.checkbox(
        "Shaded segment bands", key="show_fit_window",
        help="The coloured blocks behind the curve marking each stretch "
        "between the boundaries.",
    )
    st.checkbox(
        "Note saying which range was fitted", key="show_range_note",
        help="A line in the corner with the stretch of the curve the "
        "numbers came from, how many points that was, and how well the "
        "model followed it. A figure that leaves the room without its "
        "range on it cannot be checked later.",
    )
    st.checkbox(
        "Element curves", key="show_components",
        help="Each element's own contribution drawn apart. They are parts "
        "of the one fit, not separate fits.",
    )
    st.checkbox("Legend", key="show_legend")

    st.markdown("**Axes**")
    ax1, ax2 = st.columns(2)
    with ax1:
        st.selectbox(
            "Deformation axis",
            ["0 to 1 (full squash)", "Fit to the data", "Set my own"],
            key="x_axis_mode",
            help="Relative deformation runs from 0 to 1 by definition, so "
            "showing all of it puts every cell on the same scale. Fitting to "
            "the data zooms in on the part you measured.",
        )
        if st.session_state["x_axis_mode"] == "Set my own":
            st.number_input("ε from", key="x_axis_min", step=0.05, format="%.3f")
            st.number_input("ε to", key="x_axis_max", step=0.05, format="%.3f")
    with ax2:
        st.selectbox(
            "Force axis",
            ["Fit to the data", "Set my own"],
            key="y_axis_mode",
            help="Set your own to put several cells on the same force scale. "
            "The numbers are in the display unit chosen in the sidebar.",
        )
        if st.session_state["y_axis_mode"] == "Set my own":
            st.number_input("Force from", key="y_axis_min", step=0.5, format="%.4g")
            st.number_input("Force to", key="y_axis_max", step=0.5, format="%.4g")



# The extras that can be drawn on the force curve, all of which the single
# "Data and fit only" switch turns off together.
PLOT_EXTRAS = (
    "show_components",
    "show_fit_window",
    "show_rupture_marker",
    "show_legend",
    "show_component_heights",
    "show_range_note",
)


def axis_range(axis):
    """The limits for one axis, or None to let the data decide."""
    mode = st.session_state[f"{axis}_axis_mode"]
    if mode.startswith("Fit to"):
        return None
    if axis == "x" and mode.startswith("0 to 1"):
        return (0.0, 1.0)
    return (
        float(st.session_state[f"{axis}_axis_min"]),
        float(st.session_state[f"{axis}_axis_max"]),
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


def view_token():
    """
    What the zoom belongs to.

    Plotly keeps the view across a redraw for as long as this does not
    change. It has to change when the numbers on the axes stop meaning what
    they meant: another curve, another force unit, a log axis, or a pinned
    range. It must not change when a fit is pressed, which is the whole
    point: a fit run while zoomed in used to throw the view back out to the
    whole curve.
    """
    data = st.session_state.get("data") or {}
    epsilon = data.get("epsilon")
    return "|".join(str(part) for part in (
        data.get("source", "none"),
        int(np.size(epsilon)) if epsilon is not None else 0,
        st.session_state.get("force_unit"),
        bool(st.session_state.get("log_scale")),
        st.session_state.get("x_axis_mode"),
        st.session_state.get("y_axis_mode"),
        # And the ranges. A view kept across a change of range is a view of
        # where the curve used to be: the fitted stretch moves, the axes
        # stay, and the panel looks empty until somebody double-clicks it.
        # Pressing Fit changes none of these, so a zoom still survives that,
        # which is the case the held view exists for.
        round(float(st.session_state.get("window_start", 0.0)), 4),
        round(float(st.session_state.get("window_end", 1.0)), 4),
        round(float(st.session_state.get("segment_break_1", 0.0)), 4),
        round(float(st.session_state.get("segment_break_2", 0.0)), 4),
        tuple(
            tuple(np.round(st.session_state.get(element_window_key(term))
                           or (0.0, 0.0), 4))
            for term in ALL_TERMS
        ),
    ))


def fit_range_note(fit=None):
    """
    The line printed in the corner of the curve: what was fitted, and how well.

    Read off the fit when there is one, because that is the range the
    numbers actually came from, and off the chosen range before that, so
    the plot says what is about to be fitted rather than nothing. Pass the
    fit being drawn, so the note and the curve are the same fit.
    """
    if fit is None:
        fit = st.session_state.get("_last_fit")
    epsilon = (st.session_state.get("data") or {}).get("epsilon")
    if fit and fit.get("success"):
        lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
        bits = [f"fitted ε = {lo:.3f} to {hi:.3f}"]
        if fit.get("fit_id"):
            bits.insert(0, f"fit {fit['fit_id']}")
        points = fit.get("n_points")
        if points:
            bits.append(f"{int(points)} points")
        r2 = fit.get("r_squared")
        if r2 is not None and np.isfinite(r2):
            bits.append(f"R² = {r2:.5f}")
        q = fit.get("confinement")
        if q:
            bits.append(f"q = {float(q):.2f}")
        return "  ·  ".join(bits)

    lo, hi = range_bounds(
        "window_start", "window_end", 0.0,
        float(np.max(epsilon)) if epsilon is not None and np.size(epsilon)
        else 1.0,
        0.005,
    )
    inside = (
        int(((epsilon >= lo) & (epsilon <= hi)).sum())
        if epsilon is not None and np.size(epsilon) else 0
    )
    return f"range to fit: ε = {lo:.3f} to {hi:.3f}  ·  {inside} points"


def replace_style(style, **changes):
    """A copy of a PlotStyle with some fields changed."""
    import dataclasses
    try:
        return dataclasses.replace(style, **changes)
    except TypeError:  # pragma: no cover - a PlotStyle without the field
        return style


def current_style(force_N=None) -> PlotStyle:
    """Build the PlotStyle from the sidebar settings, honouring auto units."""
    unit = st.session_state["force_unit"]
    if unit == "auto":
        unit = autoscale_unit(force_N) if force_N is not None and np.size(force_N) else "nN"

    on = plot_flags(st.session_state).get

    return PlotStyle(
        force_unit=unit,
        data_color=st.session_state["data_color"],
        fit_color=fit_line_colour(),
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
        # Not a plot marking any more: the video is for the database
        # record and for measuring the cell's shape, not for annotating the
        # force curve. Left in the style object so nothing downstream has to
        # change, and left off.
        show_video_marker=False,
        show_rupture_marker=on("show_rupture_marker"),
        show_legend=on("show_legend"),
        show_component_heights=on("show_component_heights"),
        range_note=fit_range_note() if on("show_range_note") else None,
        uirevision=view_token(),
        # Not a marking: the fitted curve is the result, so the bare switch
        # leaves it alone and only its own checkbox removes it.
        # Neither the data nor the fit is a "marking", so the bare switch
        # leaves both alone and each has its own box.
        # Neither the data nor the model is a "marking", so the bare switch
        # leaves both alone; they go together, under one box.
        show_data=bool(st.session_state["show_data_and_fit"]),
        show_fit_line=bool(st.session_state["show_data_and_fit"]),
        x_range=axis_range("x"),
        y_range=axis_range("y"),
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
    store = st.session_state.get("onedrive_store")
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


def build_model(epsilon, force_N, active_windows=None,
                height_um=None) -> LulevichModel:
    """Construct the model from the current geometry settings. Metres, always.

    ``height_um`` overrides the page's cell height, for fitting a cell of
    the collection with its own.
    """
    height_m = float(st.session_state["cell_height_um"]
                     if height_um is None else height_um) * 1e-6
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
        # The taut layer is the protein coat, not the bilayer, and the deep
        # element of a four-element cell is myofibrils running the length of
        # the cell rather than a compact body at its centre. Both change the
        # prefactors, so both follow the cell type.
        shell_thickness=float(st.session_state["protein_coat_nm"]) * 1e-9,
        deep_uses_cell_radius=st.session_state["cell_type"] in DEEP_USES_CELL_RADIUS,
        sarcomere_length=float(st.session_state["sarcomere_nm"]) * 1e-9,
        # Eq 6 is written without a confinement factor, and a fixed cell has
        # no fluid to displace, so the tick zeroes it rather than carrying
        # over the q left behind by the living cell type.
        confinement=0.0 if fixed_cell_on()
        else float(st.session_state["confinement"]),
        poisson_membrane=float(st.session_state["poisson_membrane"]),
        poisson_interior=float(st.session_state["poisson_interior"]),
        nucleus_radius=nucleus_m,
        poisson_nucleus=float(st.session_state["poisson_nucleus"]),
        nucleus_onset=float(st.session_state["nucleus_onset"]),
        expected_ranges=expected_for(),
        active_windows=active_windows,
        segment_break_1=float(st.session_state["segment_break_1"]),
        segment_break_2=float(st.session_state["segment_break_2"]),
    )


# ============================================================ cell presets ==

TERM_SYMBOLS = {
    "tension": "T₀", "membrane": "Eₘ", "cortex": "E_cx", "interior": "Ec",
    "nucleus_shell": "E_ne", "nucleus": "Eₙ",
}
# The three classic elements. The in-plane spring and the nuclear envelope
# are extras, offered only where a cell type calls for them, and deliberately
# not in this tuple: every loop that walks the classic model must keep
# walking exactly three.
TERM_ORDER = ("membrane", "interior", "nucleus")
ALL_TERMS = (
    "tension", "membrane", "cortex", "interior", "nucleus_shell", "nucleus",
)


# Fixation is a state of the cell, not a kind of cell. A fixed C2C12 and a
# fixed cardiomyocyte keep the geometry they had when they were alive, which
# is what the prefactors are built from; what changed is the chemistry. So
# this is a tick beside the materials rather than an entry in the cell-type
# list, and the parameters below it are the living cell's.
FIXED_TYPE = "Fixed cell"


def fixed_cell_on():
    """Whether this curve is being fitted as a chemically fixed cell."""
    return bool(st.session_state.get("fixed_cell", False))


def fixed_cell_control():
    """The tick that turns the whole cell into one cross-linked solid."""

    def _changed():
        # Ticking it leaves one material on and nothing else; unticking it
        # puts back what this cell type normally starts with, rather than
        # leaving every box cleared for the person to find.
        if st.session_state.get("fixed_cell"):
            for term in ALL_TERMS:
                st.session_state[f"use_{term}"] = term == "interior"
            # One solid, loaded from first contact. Left at "at ε₁" the one
            # term would start partway along and the early curve would have
            # nothing fitting it at all.
            st.session_state["cyto_starts_at"] = "from the very start"
        else:
            wanted = DEFAULT_TERMS_BY_TYPE.get(
                st.session_state.get("cell_type"), {}
            )
            for term in ALL_TERMS:
                st.session_state[f"use_{term}"] = bool(wanted.get(term, False))

    st.checkbox(
        "🧊 Chemically fixed cell", key="fixed_cell", on_change=_changed,
        help="Fixation cross-links protein to protein throughout, so the "
        "membrane stops being a shell that stretches and the cell becomes "
        "solid. That is Lulevich eq 6, a Hertzian contact, and the "
        "materials below become cross-linked solids rather than layers of "
        "a living cell.",
    )
    if fixed_cell_on():
        st.caption(
            "Hertzian terms only, all ε³ᐟ². One is the usual case. A fixed "
            "cardiomyocyte may need two or three, cross-linked bundles "
            "being stiffer than the cytoplasm around them, and the only "
            "thing that tells them apart is where each one starts: at "
            "contact, at ε₁, at ε₂. Published whole fixed cells sit near "
            "150 to 230 kPa."
        )


def terms_for(cell_type):
    """The elements this cell type can be fitted with, in display order."""
    if fixed_cell_on():
        return OPTIONAL_TERMS[FIXED_TYPE]
    return OPTIONAL_TERMS.get(cell_type, TERM_ORDER)


def term_name(term, cell_type=None):
    """This cell type's name for one element.

    Every label the app shows goes through here. A cardiomyocyte has no
    nucleus term, and the way to make sure the word never appears is to have
    no place where it is written down outside the component table.
    """
    if cell_type is None:
        cell_type = st.session_state.get("cell_type")
    names = COMPONENT_SETS.get(cell_type, DEFAULT_COMPONENTS)
    return names.get(term, DEFAULT_COMPONENTS.get(term, (term, "")))[0]


def plain_name(term, cell_type=None):
    """The element's name without its emoji.

    The emoji earns its place on a checkbox, where it makes four similar
    labels tellable apart at a glance. It does not belong in a table header,
    a spreadsheet column or a sentence, so those go through here.
    """
    name = term_name(term, cell_type)
    head, _, rest = name.partition(" ")
    return rest.strip() if rest and not head[:1].isalnum() else name


def term_label(term, cell_type=None):
    """Name plus symbol, e.g. "Sarcomeric myofibrils (Eₙ)"."""
    return f"{term_name(term, cell_type)} ({TERM_SYMBOLS.get(term, term)})"


def retell(text):
    """Put this cell type's names into a sentence the model wrote.

    The model layer has no idea what the cell is called, so it writes
    "nucleus" for the deep element, which is the right word for a myoblast
    and the wrong one for a cardiomyocyte. Translating at the boundary keeps
    the model honest and generic while nothing on screen says a word that
    does not belong to the cell in front of you.
    """
    if not isinstance(text, str):
        return text
    deep = term_name("nucleus")
    if deep.lower() == "nucleus":
        return text
    return text.replace("nucleus", deep.lower()).replace("Nucleus", deep)


# How the elements share the load. These are physics, not fitting procedure:
# parallel and series here describe the spring network, while "All at once"
# and "Stage by stage" describe how the fit is carried out.
MODELS = {
    "Segmented (each part takes over in turn)":
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
    "Cardiomyocyte (Morales Maldonado)":
        "A strong shell around fluid that does not compress. The shell carries "
        "the load as ε³; there is no separate Hertzian interior, because an "
        "incompressible fluid resists through pressure rather than as an "
        "elastic solid. A second term is available for the myofibrils once "
        "they are met. PROVISIONAL: built from your description, not from the "
        "paper's equations.",
}
MODEL_KEYS = {
    "Segmented (each part takes over in turn)": "segmented",
    "Side by side (every element acts everywhere)": "parallel",
    "Stacked (elements in line)": "series",
    "Side by side, then stacked": "hybrid_ps",
    "Stacked, then side by side": "hybrid_sp",
    "Compare these and rank them": "auto",
    "Cardiomyocyte (Morales Maldonado)": "segmented",
}

# Older saved presets and stored database records name the model with the
# wording this app used before, including "parallel below, series above",
# which nobody could read. They are kept here only so that an old record still
# refits; nothing in the interface offers these names any more.
LEGACY_MODEL_NAMES = {
    # Named after a myoblast's three parts, which was wrong for every cell
    # that does not have those three. Records written under it still load.
    "Segmented (membrane → cytoskeleton → nucleus)": "segmented",
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
    "starts stretching at ε₁": "late",
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
    ("late", "zero"): "Cytoskeleton alone, then the membrane starts stretching at ε₁",
    ("late", "break"): "Neither carries load until ε₁, which is not a model of anything",
}


def composition_label(membrane, cyto_start):
    """Plain words for a composition, and never a crash for a new one.

    A dict lookup here was a KeyError waiting for the next composition to be
    added, which is exactly what happened.
    """
    return COMPOSITION_LABELS.get(
        (membrane, cyto_start),
        f"membrane {membrane}, cytoskeleton from "
        f"{'zero' if cyto_start == 'zero' else 'ε₁'}",
    )

STAGE_COLORS = ("#2ca02c", "#9467bd", "#e377c2", "#ff7f0e")

# Starting points, not literature constants. They set the geometry, the
# plausibility bands used for warnings, and the initial fit windows for a
# cell type. Edit them for your own line and save the result as a preset.
# What the three terms are called, per cell type. The maths is the same; the
# names are not. A cardiomyocyte is not a myoblast with a different modulus:
# it is packed with myofibrils and, on the Morales Maldonado picture, behaves
# as a strong shell around fluid that does not compress, so calling the second
# term "cytoskeleton" and the third "nucleus" would misdescribe it.
COMPONENT_SETS = {
    # A myoblast's nucleus is a balloon inside a balloon. It has an envelope,
    # two membranes and the lamina under them, and that envelope is a shell
    # that resists being stretched exactly as the cell's own membrane does;
    # what it contains resists being squeezed. Modelling the whole nucleus as
    # one Hertzian lump said it was jelly with no skin, which is the one
    # thing a nucleus is known not to be.
    # The names, the squares and the symbols are the same four whichever
    # way the load is shared: one components table for the whole app, so a
    # person switching from the regime-by-regime fit to side by side finds
    # the components they already had rather than four new ones.
    "Myoblast (C2C12)": {
        "membrane": ("🟥 Membrane", "the skin around the cell"),
        "interior": ("🟧 Cytoskeleton", "the scaffolding filling the cell"),
        "nucleus_shell": (
            "🟦 Nuclear envelope",
            "the skin around the nucleus, stretched as it is squashed",
        ),
        "nucleus": (
            "🟪 Inside the nucleus",
            "what the envelope contains, squeezed like everything else",
        ),
    },
    # A cardiomyocyte is modelled as a shell around one incompressible
    # interior, and that is three springs, not four.
    #
    # There is no nucleus term. Not renamed, not switched off by default:
    # absent. A nucleus cannot be told apart from the rest of the interior of
    # a cardiomyocyte, by the plates or by anyone looking down a microscope,
    # so a term claiming to measure one measures nothing. The myofibrils are
    # not a separate deep spring either: they fill the cell, and together
    # with the cytoskeleton around them they are treated as a single
    # incompressible material.
    #
    # What that costs is nothing. Fitting the four measured WT curves with
    # the deep spring removed changes chi-squared per point by less than a
    # tenth (cell 11: 1.34 against 1.32; cell 5: identical), because what the
    # deep spring was really absorbing was the cell running out of room, and
    # running out of room is what an incompressible interior does. It is
    # described by the confinement exponent q instead, where it belongs.
    # Three materials, in the order the plates meet them: the sarcolemma,
    # the packed interior, and the myofibrils met deeper in. No cortical
    # layer of its own: it is part of what the sarcolemma carries, and a
    # separate Hertzian term for it can only be told from the interior by
    # its onset, which is a claim this cell does not need.
    #
    # No nucleus either. Nothing in a cardiomyocyte tells one apart from the
    # rest of the interior, so a term claiming to measure one measures
    # nothing.
    "Cardiomyocyte": {
        "membrane": (
            "🎈 Sarcolemma",
            "the membrane around the cell, resisting being stretched",
        ),
        "interior": (
            "🧬 Non-sarcomeric cytoskeleton",
            "the general scaffolding, loaded from first contact",
        ),
        "nucleus": (
            "🧵 Sarcomeric cytoskeleton (myofibrils)",
            "the contractile bundles, reached deeper in",
        ),
    },
    # A chemically fixed cell is no longer a shell around a filling.
    # Fixation cross-links protein to protein throughout, so membrane,
    # cytoskeleton and nucleus become one solid, and one solid squeezed
    # between two plates is a Hertzian contact and nothing else. It is the
    # elastic-sphere limit of the same model: the ε³ shell term has nothing
    # separate left to describe, and a fit offered one will split the curve
    # between two terms that are no longer two materials.
    # More than one Hertzian term is allowed here, and for a fixed
    # cardiomyocyte it is the interesting case: fixation cross-links the
    # myofibril bundles into something stiffer than the cytoplasm around
    # them, and the plates meet that only after they have squashed past the
    # softer material. Every term obeys the same ε^3/2 law, so what tells
    # them apart is where each one starts and nothing else. Two terms with
    # the same law and the same onset are one term with two names, which is
    # why each of these begins somewhere different.
    "Fixed cell": {
        "cortex": (
            "🧊 Outer solid, from contact",
            "the outermost cross-linked layer, loaded from first touch",
        ),
        "interior": (
            "🧊 The fixed cell",
            "one cross-linked solid; from first contact on its own, or "
            "from ε₁ when an outer solid is ticked as well",
        ),
        "nucleus": (
            "🧱 A deeper, stiffer solid, from ε₂",
            "a second Hertzian body met further in, such as fixed "
            "myofibril bundles",
        ),
    },
}
DEFAULT_COMPONENTS = COMPONENT_SETS["Myoblast (C2C12)"]

# The pictures of the cell the app will actually test, in the words they
# would be argued in. Each is a claim a person can agree or disagree with,
# not a setting they have to translate, and the search reports which of them
# the curve supports rather than reporting four abstract compositions.
#
# For a ventricular cardiomyocyte every picture is a shell around one
# incompressible interior. What varies between them is the order the two
# load in and whether the membrane carries an in-plane spring as well, which
# are the two things an experiment can actually change.
# What the sample is, which decides which pictures are worth testing.
# What the sample is. The in-plane spring belongs to the wild-type membrane
# and the experiment takes it away, so this is not a switch for adding a
# term: it is which cell is on the stage.
MEMBRANE_PROTEIN_STATES = {
    "Present (wild type)":
        "The spring is part of this membrane, so every picture carries it "
        "and the fit measures its tension T₀.",
    "Removed (knockout)":
        "The spring has been taken out of the cell, so it is taken out of "
        "the model: no picture is offered one. If the curve still needs a "
        "term linear in ε, that is a result worth knowing rather than a "
        "term to fit away.",
    "Not known — test for it":
        "Both are tried and the curve decides, which is the comparison that "
        "says whether the spring is still contributing mechanically.",
}


def cardiomyocyte_hypotheses(state=None):
    """
    The pictures worth testing for a ventricular cardiomyocyte.

    Three materials: the sarcolemma, the non-sarcomeric cytoskeleton, and
    the sarcomeric cytoskeleton — the myofibrils — reached deeper in. No
    nucleus, because nothing here tells one apart from the rest of the
    interior, and no cortical layer of its own, because it obeys the same
    Hertzian law as the interior and could only be separated from it by an
    onset this cell does not need.

    The interior is anchored to the sarcolemma and so is never left unloaded
    while the membrane deforms alone: it carries the load from first
    contact in every picture. What varies is the two things the curve can
    answer: whether the sarcolemma answers from contact or only begins to
    stretch at ε₁, and whether the deeper myofibrils are reached at all
    inside the analysed range.

    ``state`` is accepted and ignored.
    """
    shell_late = {"membrane": "late", "cyto_start": "zero"}
    shell_early = {"membrane": "continue", "cyto_start": "zero"}
    return [
        {
            "key": "interior_first",
            "label": "Cytoskeleton first, then the sarcolemma stretches, "
                     "then the myofibrils",
            "detail": "the membrane only starts stretching at ε₁; "
                      "myofibrils at ε₂",
            "terms": ("membrane", "interior", "nucleus"), **shell_late,
        },
        {
            "key": "coupled",
            "label": "Sarcolemma and cytoskeleton together, then the "
                     "myofibrils",
            "detail": "the membrane loads from first contact; myofibrils "
                      "at ε₂",
            "terms": ("membrane", "interior", "nucleus"), **shell_early,
        },
        {
            "key": "interior_first_no_myofibrils",
            "label": "Cytoskeleton first, then the sarcolemma, no myofibrils "
                     "reached",
            "detail": "the same, but nothing deeper is met in this range",
            "terms": ("membrane", "interior"), **shell_late,
        },
        {
            "key": "coupled_no_myofibrils",
            "label": "Sarcolemma and cytoskeleton together, no myofibrils "
                     "reached",
            "detail": "the membrane from contact, nothing deeper met",
            "terms": ("membrane", "interior"), **shell_early,
        },
    ]


HYPOTHESES = {
    # One picture, because there is only one material. The comparison the
    # other cell types run has nothing to compare, and saying so is better
    # than offering a choice between a thing and itself.
    "Fixed cell": [
        {
            "key": "one_solid",
            "label": "One cross-linked solid",
            "detail": "a Hertzian contact over the whole curve",
            "terms": ("interior",),
            "membrane": "continue", "cyto_start": "zero",
        },
        {
            "key": "two_solids",
            "label": "One solid, then a stiffer one deeper in",
            "detail": "a second Hertzian body met at ε₂",
            "terms": ("interior", "nucleus"),
            "membrane": "continue", "cyto_start": "zero",
        },
        {
            "key": "three_solids",
            "label": "Three solids, each met further in",
            "detail": "Hertzian from contact, from ε₁ and from ε₂",
            "terms": ("cortex", "interior", "nucleus"),
            "membrane": "continue", "cyto_start": "break",
        },
    ],
    "Myoblast (C2C12)": [
        {
            "key": "handover_with_envelope",
            "label": "Membrane, then cytoskeleton, then the nucleus with its "
                     "envelope",
            "detail": "the nucleus is a balloon of its own, a skin around a "
                      "filling",
            "terms": ("membrane", "interior", "nucleus_shell", "nucleus"),
            "membrane": "freeze", "cyto_start": "break",
        },
        {
            "key": "coupled_with_envelope",
            "label": "Membrane and cytoskeleton together, then the nucleus "
                     "with its envelope",
            "detail": "both load from first contact; the nucleus has a skin",
            "terms": ("membrane", "interior", "nucleus_shell", "nucleus"),
            "membrane": "continue", "cyto_start": "zero",
        },
        {
            "key": "envelope_only",
            "label": "Membrane, then cytoskeleton, then the nuclear envelope "
                     "alone",
            "detail": "the plates feel the skin of the nucleus but not what "
                      "is inside it",
            "terms": ("membrane", "interior", "nucleus_shell"),
            "membrane": "freeze", "cyto_start": "break",
        },
        {
            "key": "handover",
            "label": "Membrane first, then cytoskeleton, then nucleus",
            "detail": "each takes over from the last",
            "terms": ("membrane", "interior", "nucleus"),
            "membrane": "freeze", "cyto_start": "break",
        },
        {
            "key": "coupled",
            "label": "Membrane and cytoskeleton together, then the nucleus",
            "detail": "both load from first contact",
            "terms": ("membrane", "interior", "nucleus"),
            "membrane": "continue", "cyto_start": "zero",
        },
        {
            "key": "no_nucleus",
            "label": "Membrane and cytoskeleton only",
            "detail": "the nucleus is never reached",
            "terms": ("membrane", "interior"),
            "membrane": "freeze", "cyto_start": "break",
        },
    ],
}


def agrees_with_prior(spec, cell_type=None):
    """
    Whether one arrangement is consistent with what this cell type does.

    An element the cell type carries throughout cannot be arranged so that
    it stops. "Membrane holds what it reached at ε₁" is a legitimate
    arrangement for a cell whose shell does that; for a C2C12, whose
    sarcolemma goes on taking load to the end, it is a picture of a
    different cell, and offering it to the comparison is how the fit came
    back saying the membrane stopped carrying at the first boundary.
    """
    throughout = component_prior(cell_type).get("throughout") or ()
    if not throughout:
        return True
    if ("membrane" in throughout
            and spec.get("membrane", "continue") != "continue"):
        return False
    if ("interior" in throughout
            and spec.get("cyto_start", "zero") != "zero"):
        return False
    return True


def hypotheses_for(cell_type, terms=None, exact=False):
    """
    The named pictures to test for this cell type.

    ``terms`` restricts them to the materials that are actually ticked.
    ``exact`` goes further: every picture carries exactly that mixture, so
    the comparison is over arrangements alone and fitting can never change
    which boxes are ticked. Which components are in the model is a separate
    question with its own criterion, asked by the search button, and a fit
    that quietly answered it as a side effect was answering it twice.
    Without that, unticking a material had no effect at all: every picture
    carried its own list of terms and the winner wrote them straight back,
    so a box you had just cleared reappeared with a modulus beside it. A
    fit without the membrane is a legitimate thing to ask for, and asking
    for it should produce exactly that.

    A picture stripped down to nothing is dropped, and two that collapse
    onto the same set of materials and the same order are one picture.
    """
    if fixed_cell_on():
        # One solid, two, or three, told apart by where each starts. Which
        # of them is on the list is decided by the boxes, the same as for a
        # living cell.
        found = HYPOTHESES[FIXED_TYPE]
    elif cell_type in INCOMPRESSIBLE_INTERIOR:
        found = cardiomyocyte_hypotheses()
    else:
        found = HYPOTHESES.get(cell_type, [])
    # Marked, not filtered. What is known about a cell type breaks a tie in
    # its favour; it does not overrule a curve that says something else
    # loudly, because "usually" is what the prior claims and no more.
    found = [dict(spec, expected=agrees_with_prior(spec, cell_type))
             for spec in found]
    if terms is None:
        return found

    keep = set(terms)
    trimmed, seen = [], set()
    if exact:
        whole = tuple(term for term in ALL_TERMS if term in keep)
        if not whole:
            return []
        for spec in found:
            membrane = spec.get("membrane", "continue")
            if "membrane" not in whole:
                membrane = "continue"
            signature = (membrane, spec.get("cyto_start", "zero"))
            if signature in seen:
                continue
            seen.add(signature)
            label, detail = spec["label"], spec.get("detail", "")
            if whole != tuple(spec["terms"]):
                # A name that lists a material the picture no longer has is
                # a name that lies, and it is the name the page prints as
                # the answer.
                label = " + ".join(plain_name(term) for term in whole)
                if membrane == "late":
                    label += ", membrane from ε₁"
                detail = "the components you ticked" + (
                    ", with the deeper layer met at ε₂"
                    if "nucleus" in whole else ""
                )
            trimmed.append(dict(spec, terms=whole, membrane=membrane,
                                label=label, detail=detail))
        return trimmed
    for spec in found:
        here = tuple(term for term in spec["terms"] if term in keep)
        if not here:
            continue
        # With the membrane gone there is nothing for "the membrane starts
        # stretching at ε₁" to mean, so those pictures collapse onto the
        # one that is left rather than being compared against themselves.
        membrane = spec.get("membrane", "continue")
        if "membrane" not in here:
            membrane = "continue"
        signature = (here, membrane, spec.get("cyto_start", "zero"))
        if signature in seen:
            continue
        seen.add(signature)
        # A trimmed picture needs a name that describes it. Keeping the
        # original said "then the membrane stretches" about a fit with no
        # membrane in it, which is a label that lies.
        label, detail = spec["label"], spec.get("detail", "")
        if here != tuple(spec["terms"]):
            label = " + ".join(plain_name(term) for term in here)
            if membrane == "late":
                label += ", membrane from ε₁"
            detail = "the materials you ticked" + (
                ", with the deeper layer met at ε₂" if "nucleus" in here else ""
            )
        trimmed.append(dict(spec, terms=here, membrane=membrane,
                            label=label, detail=detail))
    return trimmed


# Cell types whose interior is one incompressible material rather than a
# soft filling with something denser in the middle. For these there is no
# deep spring at all: the interior does not compress, so past about half the
# cell's height the resistance is the cell running out of room, which is the
# confinement exponent q and not a third modulus. q is measured per cell.
INCOMPRESSIBLE_INTERIOR = ("Cardiomyocyte",)

# Cell types with sarcomeres. Only for the geometry read-out and for the
# in-plane membrane protein, both of which are about muscle; neither of them
# puts a spring in the fit.
HAS_SARCOMERES = ("Cardiomyocyte",)

# Where a deep term does exist, whose radius its prefactor uses: a nucleus is
# a body at the centre with its own radius, anything running the length of
# the cell uses the cell's own.
DEEP_USES_CELL_RADIUS = ("Cardiomyocyte",)

# Elements a cell type can be fitted with, in the order they are shown,
# always outside inwards.
OPTIONAL_TERMS = {
    # Outside inwards: the shell, the cortex under it, the general
    # scaffolding, the myofibrils. No in-plane spring: the plain
    # cardiomyocyte has to be settled before a membrane protein is added
    # to it, and a term nobody has asked for quietly takes force from the
    # ones that were asked for.
    "Cardiomyocyte": ("membrane", "interior", "nucleus"),
    # The nucleus is two elements, an envelope and what it contains.
    "Myoblast (C2C12)": ("membrane", "interior", "nucleus_shell", "nucleus"),
    "Custom": ("membrane", "interior", "nucleus_shell", "nucleus"),
    # Hertzian terms only, told apart by where each starts: from contact,
    # from ε₁, from ε₂. One is the usual case; a fixed cardiomyocyte may
    # want two or three.
    "Fixed cell": ("cortex", "interior", "nucleus"),
}

# Which of those are ticked when the cell type is chosen.
DEFAULT_TERMS_BY_TYPE = {
    # A wild-type cardiomyocyte has the in-plane spring, so it is on. The
    # knockout is the experiment that takes it away, and choosing that
    # genotype in the sidebar is what removes it from the model.
    "Cardiomyocyte": {
        "membrane": True, "interior": True, "nucleus": True,
        "cortex": False, "nucleus_shell": False, "tension": False,
    },
    "Myoblast (C2C12)": {
        "membrane": True, "interior": True, "nucleus": True,
        "nucleus_shell": True, "tension": False, "cortex": False,
    },
    "Fixed cell": {
        "membrane": False, "interior": True, "nucleus": False,
        "nucleus_shell": False, "tension": False, "cortex": False,
    },
}


def wants_confinement(cell_type=None):
    """Whether q should be measured from the curve for this cell.

    Two reasons to measure it, and either is enough. A cell that is not a
    free sphere has less room to spread as it is flattened. And a cell whose
    interior does not compress has to put that volume somewhere. Both show up
    as the same factor, so both are answered by the same measurement.
    """
    if cell_type is None:
        cell_type = st.session_state.get("cell_type")
    if fixed_cell_on():
        # A cross-linked solid has no fluid to displace and no shell to run
        # out of room inside, and eq 6 is written without a confinement
        # factor. Measuring one here would fit the model to itself.
        return False
    return (
        cell_type in INCOMPRESSIBLE_INTERIOR
        or str(st.session_state.get("cell_shape", "")).startswith("Belt")
    )


def has_deep_term(cell_type=None):
    """Whether this cell type is modelled with a deep element at all."""
    if cell_type is None:
        cell_type = st.session_state.get("cell_type")
    return "nucleus" in terms_for(cell_type)

# How each cell type is expected to behave at the first boundary.
#
# A myoblast shows the membrane alone at small deformation, an almost pure
# cube law, so the cytoskeleton is taken to start at the boundary.
#
# A cardiomyocyte does not: the measured local exponent near contact is about
# 1.7, not 3. That is the signature of the membrane's cube law and the
# cytoskeleton's 3/2 law acting together from the very start, with the 3/2
# term dominating. Physically the cortex and the myofibrils are anchored to
# the membrane through the costameres, so there is no stretch where the
# membrane is deforming on its own. Starting the cytoskeleton at zero is what
# reproduces a 1.7 rather than a 3.
DEFAULT_COMPOSITION_BY_TYPE = {
    # A C2C12 is met in this order: the membrane from first contact, the
    # cytoskeleton joining it at ε₁ and both carrying load together from
    # there, the nuclear envelope at ε₂, and what the envelope contains
    # after that. The membrane goes on stiffening rather than holding at
    # ε₁, so past ε₁ it is a mixture of the two.
    #
    # The cytoskeleton starts AT ε₁ and not at zero, and that is not a
    # detail of taste. Loaded from zero its 3/2 law and the membrane's
    # cube law run over the same stretch of curve, their basis functions
    # correlate at about 0.98, and a fit that cannot tell two shapes apart
    # gives all of the force to one of them and returns exactly zero for
    # the other. A component that comes back as 0 has not been removed by
    # the page: it has been made unmeasurable by where it was told to
    # start. A separate onset is what makes it measurable.
    "Myoblast (C2C12)": {
        "membrane_after_break": "holds what it reached",
        "cyto_starts_at": "at ε₁",
    },
    "Cardiomyocyte": {
        "membrane_after_break": "starts stretching at ε₁",
        # The interior is anchored to the sarcolemma, so it is never left
        # unloaded while the membrane deforms alone: it carries the load
        # from first contact, which is what a slope near 3/2 there means.
        "cyto_starts_at": "from the very start",
    },
}

# What every other cell type starts from, and what a cell type without an
# entry above is reset TO. This is not decoration: switching to a
# cardiomyocyte and back used to leave its composition behind, so a myoblast
# was then fitted as though its membrane kept stiffening and its cytoskeleton
# loaded from zero. The fit still succeeded and still looked good, and every
# modulus was wrong. A default that only ever gets applied one way is not a
# default, it is a one-way door.
BASE_COMPOSITION = {
    "membrane_after_break": "holds what it reached",
    "cyto_starts_at": "at ε₁",
}


# A neutral name for every slot, used only where a cell type does not name
# one because it does not have one. Deliberately not the myoblast's names:
# falling back to those would put the word "nucleus" into a cardiomyocyte's
# component list, which is exactly what the cell-type names exist to avoid.
GENERIC_COMPONENTS = {
    "tension": ("In-plane spring", "a spring taut in the membrane"),
    "membrane": ("Membrane", "the skin around the cell"),
    "cortex": ("Cortex", "the layer just under the membrane"),
    "interior": ("Interior", "the material filling the cell"),
    "nucleus_shell": ("Deep shell", "a shell met deeper in"),
    "nucleus": ("Deep element", "whatever the plates reach deeper in"),
}


def components_for(cell_type):
    """
    Names for every slot, whether or not this cell type uses it.

    Every slot is filled, so that code which reaches for a name it does not
    expect to be missing gets a name rather than a KeyError. Which slots are
    actually fitted is ``terms_for``, and that is the only thing that
    decides what appears on the page.
    """
    names = dict(
        GENERIC_COMPONENTS,
        **COMPONENT_SETS.get(cell_type, DEFAULT_COMPONENTS),
    )
    if fixed_cell_on():
        # The materials left are cross-linked solids, so they are named that
        # way rather than keeping the living cell's words for its layers.
        names.update(COMPONENT_SETS[FIXED_TYPE])
    return names


CELL_TYPES = {
    "Myoblast (C2C12)": {
        "cell_height_um": 8.0,
        "radius_aspect": 0.55,
        "nucleus_fraction": 0.35,
        "membrane_thickness_nm": 4.0,
        "nucleus_onset": 0.20,
        "cell_shape": "Sphere (a rounded cell)",
        "confinement": 0.0,
        "weighting": "uniform",
        "schematic_style": "Mechanics schematic",
        # expected bands in pascals
        "expected": {"Em": (2e5, 2e7), "Ei": (2e2, 1e4), "En": (1e3, 5e4)},
        # The membrane carries load from the very start of the compression, so
        # its window opens at zero rather than partway along.
        "windows": {"membrane": (0.0, 0.40), "interior": (0.0, 0.40),
                    "nucleus": (0.40, 1.0)},
    },
    # Set from a measured WT curve rather than from round numbers: an adult
    # cardiomyocyte lying on the dish is about 19 um tall, and it lies there
    # as a rod, so its radius is half its height (aspect 0.5) rather than the
    # 0.55 of a rounded-up cell. The membrane is taken as 8 nm, not the 4 nm
    # of a bare bilayer, because what carries load here is the bilayer plus
    # the protein coat welded to it.
    # Everything the same as a myoblast except the two things that really
    # differ: the cell is taller, and the sarcolemma is about twice the
    # thickness of a bare bilayer. Same aspect factor, same Poisson ratios,
    # same starting windows. Keeping the shared parameters shared is what
    # makes a modulus from one cell type comparable with the other; a
    # difference invented for one of them is a difference that turns up in
    # the answer and cannot be traced.
    "Cardiomyocyte": {
        "cell_height_um": 19.0,
        "radius_aspect": 0.55,
        "nucleus_fraction": 0.35,
        "membrane_thickness_nm": 8.0,
        "nucleus_onset": 0.20,
        "cell_shape": "Sphere (a rounded cell)",
        "schematic_style": "Balloon with a spring inside",
        # These curves span four decades of force. Weighted uniformly the
        # fit is decided by the last tenth of the curve and misses the first
        # half by tens of per cent; weighted by 1/|F| it holds to a few per
        # cent everywhere, which is what "it fits the curve" has to mean
        # when the curve is plotted on a log axis.
        "weighting": "relative",
        # Only a starting value: q is measured from every curve, because an
        # interior that does not compress is what it describes.
        "confinement": 1.10,
        "expected": {"Em": (5e4, 5e7), "Ei": (1e2, 1e5), "En": (5e2, 2e5)},
        "windows": {"membrane": (0.0, 0.35), "interior": (0.0, 0.35),
                    "nucleus": (0.35, 1.0)},
    },
    # Same geometry as the living cell it was made from; what changed is the
    # chemistry, not the shape. Fixation makes a cell several times stiffer,
    # so the plausibility band is wider at the top.
    "Fixed cell": {
        "cell_height_um": 8.0,
        "radius_aspect": 0.55,
        "nucleus_fraction": 0.35,
        "membrane_thickness_nm": 4.0,
        "nucleus_onset": 0.20,
        "cell_shape": "Sphere (a rounded cell)",
        "confinement": 0.0,
        "weighting": "relative",
        "expected": {"Em": (2e5, 2e7), "Ei": (5e2, 5e5), "En": (1e3, 5e4)},
        "windows": {"membrane": (0.0, 0.40), "interior": (0.0, 1.0),
                    "nucleus": (0.40, 1.0)},
    },
    "Custom": {
        "cell_height_um": 8.09,
        "radius_aspect": 0.55,
        "nucleus_fraction": 0.35,
        "membrane_thickness_nm": 4.0,
        "nucleus_onset": 0.15,
        "cell_shape": "Sphere (a rounded cell)",
        "confinement": 0.0,
        "expected": {"Em": (1e3, 1e9), "Ei": (1e0, 1e7), "En": (1e1, 1e7)},
        "windows": {"membrane": (0.0, 0.40), "interior": (0.0, 0.40),
                    "nucleus": (0.40, 1.0)},
    },
}

# What the dropdown offers. "Fixed cell" is not a kind of cell you grew, it
# is what you did to one, so it is a tick beside the materials and its entry
# above is only the plausibility band that tick switches to.
SELECTABLE_CELL_TYPES = [name for name in CELL_TYPES if name != FIXED_TYPE]


def expected_for(cell_type=None):
    """The plausibility band to judge this fit against.

    A fixed cell is several times stiffer than the cell it was made from, so
    warning about it against the living band would flag every good fit.
    """
    if cell_type is None:
        cell_type = st.session_state.get("cell_type")
    if fixed_cell_on():
        return CELL_TYPES[FIXED_TYPE]["expected"]
    return CELL_TYPES.get(cell_type, {}).get("expected")


# Where each cell type's boundaries start, before anything is measured.
# These are the placements the model is usually written with, not fits: a
# C2C12's membrane holds at about a sixth of the way down and its nucleus is
# met a little under half way. A curve that disagrees says so when the
# boundary search is pressed.
DEFAULT_BOUNDARIES_BY_TYPE = {
    # A C2C12 is met by its sarcolemma and its cytoskeleton from first
    # contact and by its nucleus somewhere between 44 % and 70 %, so ε₂
    # starts in the middle of that band rather than at a number with
    # nothing behind it. ε₁ is kept for the arrangements that use it; with
    # both outer elements carrying load throughout it does nothing.
    "Myoblast (C2C12)": {"segment_break_1": 0.15, "segment_break_2": 0.50},
    "Cardiomyocyte": {"segment_break_1": 0.15, "segment_break_2": 0.40},
}

# What is already known about where each element of a cell type carries
# load. This is prior knowledge, not a measurement, and it is used in two
# places: the range a component starts on, and the interval the search is
# allowed to move a boundary within. A search with no prior wanders --
# press it twice on the same curve after nudging one bar and the boundaries
# can end up on the other side of the cell -- and a boundary that moves
# further than the biology allows is not a measurement of anything.
COMPONENT_PRIORS = {
    "Myoblast (C2C12)": {
        # Load-bearing from first contact to the end of the squash.
        "throughout": ("membrane", "interior"),
        # Where the deep elements are first met. The measured signature is a
        # bump in the force at about half the squash: that is the probe
        # reaching the nucleus, and it runs for roughly another quarter of
        # the deformation after it starts. The onset is therefore near 0.50,
        # and the band is that give or take, not a third of the curve.
        "onset": {"nucleus": (0.44, 0.62), "nucleus_shell": (0.44, 0.62)},
        # The feature itself, for drawing on the curve and for saying what
        # the boundary is supposed to coincide with.
        "bump": {"from": 0.50, "span": 0.25},
        # Once met, a deep element keeps carrying to the end: nothing in a
        # C2C12 stops taking load except the shell, and only if it is
        # arranged to hold at ε₁.
        "runs_to_the_end": ("interior", "nucleus", "nucleus_shell"),
        "why": (
            "in a C2C12 the sarcolemma and the cytoskeleton carry load from "
            "first contact, and the nucleus shows as a bump at about 50 % "
            "relative deformation that runs on for another 25 %, so ε₂ sits "
            "between 44 % and 62 %"
        ),
    },
}


def bump_window(cell_type=None):
    """Where this cell type's deep element shows itself, if it is known."""
    bump = component_prior(cell_type).get("bump")
    if not bump:
        return None
    start = float(bump["from"])
    return (start, start + float(bump.get("span", 0.0)))


def component_prior(cell_type=None):
    """What is known in advance about this cell type's elements."""
    if cell_type is None:
        cell_type = st.session_state.get("cell_type")
    return COMPONENT_PRIORS.get(cell_type, {})


def carries_throughout(term, cell_type=None):
    """Whether this element is known to bear load over the whole squash."""
    return term in (component_prior(cell_type).get("throughout") or ())


def onset_band(term, cell_type=None, lo=0.0, hi=1.0):
    """
    The interval this element's onset is allowed to sit in.

    Returns the whole range where nothing is known, so a cell type without
    a prior is searched exactly as before.
    """
    band = (component_prior(cell_type).get("onset") or {}).get(term)
    if not band:
        return (float(lo), float(hi))
    a, b = float(band[0]), float(band[1])
    return (float(np.clip(a, lo, hi)), float(np.clip(b, lo, hi)))


def deep_onset_band(cell_type=None, lo=0.0, hi=1.0):
    """The ε₂ band: where the deep elements of this cell type are met."""
    onsets = component_prior(cell_type).get("onset") or {}
    bands = [onset_band(term, cell_type, lo, hi)
             for term in ("nucleus_shell", "nucleus") if term in onsets]
    if not bands:
        return (float(lo), float(hi))
    return (min(b[0] for b in bands), max(b[1] for b in bands))


def default_boundaries(cell_type=None):
    """
    This cell type's starting ε₁ and ε₂, as settings to apply.

    Boundaries learned from a batch of real curves win over the written-down
    ones: they are the same claim, measured on this lab's cells rather than
    taken from the literature.
    """
    if cell_type is None:
        cell_type = st.session_state.get("cell_type")
    learned = (st.session_state.get("learned_boundaries") or {}).get(cell_type)
    if learned:
        return {
            "segment_break_1": round(float(learned["break_1"]), 3),
            "segment_break_2": round(float(learned["break_2"]), 3),
        }
    return dict(DEFAULT_BOUNDARIES_BY_TYPE.get(cell_type, {
        "segment_break_1": DEFAULTS["segment_break_1"],
        "segment_break_2": DEFAULTS["segment_break_2"],
    }))


# Session keys that are not settings and must never be carried through a
# rerun by hand: the data, the fit, the connections and the caches.
NOT_A_SETTING = (
    "data", "results", "gs_manager", "onedrive_store", "_last_fit",
    "_last_fit_signature", "_plot_png", "video_path", "video_info",
    "video_track", "video_saved_frame", "exploration", "composition_search",
    "arrangement_search", "component_search", "confinement_scan",
    "hypothesis_search", "boundary_search", "element_window_search",
    "pw_boundary_search", "pw_placements", "pw_selected", "pw_collection",
)


def rerun_keeping_settings(extra=None, forget=()):
    """See below; ``forget`` names settings this rerun should NOT carry."""
    """
    Rerun, and carry every setting on the page through it.

    Streamlit forgets a widget's value when a run ends without drawing that
    widget. Half of this page's settings live in a sidebar panel built after
    the analysis, so a button that reruns from inside the analysis ended the
    run before they were drawn and they reverted -- which is how pressing a
    button in step 1 silently changed the arrangement in step 3. Writing
    them into the pending block re-asserts them at the top of the next run,
    before any widget exists, which is the only place Streamlit allows it.
    """
    pending = {
        key: st.session_state[key]
        for key in DEFAULTS
        if key in st.session_state and key not in NOT_A_SETTING
        and key not in set(forget)
    }
    pending.update(extra or {})
    st.session_state["_pending_settings"] = pending
    st.rerun()


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
        "cell_shape",
        "confinement",
        "weighting",
        "schematic_style",
    ):
        if key in preset:
            st.session_state[key] = preset[key]
    for term, window in preset["windows"].items():
        st.session_state[f"celltype_window_{term}"] = tuple(window)
    # Which terms this cell type is modelled with. A cardiomyocyte is packed
    # with myofibrils and is treated as one stiff interior, so the nucleus
    # term is switched off: left on it has nothing distinct to describe and
    # simply trades off against the interior modulus, making both unstable.
    # Every term, not only this cell type's, so one left switched on by the
    # previous cell type cannot survive the change.
    wanted = dict.fromkeys(ALL_TERMS, False)
    wanted.update({"membrane": True, "interior": True, "nucleus": True,
                   "nucleus_shell": True, "cortex": True})
    wanted.update(DEFAULT_TERMS_BY_TYPE.get(name, {}))
    for term in ALL_TERMS:
        st.session_state[f"use_{term}"] = bool(
            wanted.get(term, False) and term in terms_for(name)
        )
    # Composition too, and always to a known state rather than only where a
    # cell type happens to override it.
    composition = dict(BASE_COMPOSITION)
    composition.update(DEFAULT_COMPOSITION_BY_TYPE.get(name, {}))
    for key, value in composition.items():
        st.session_state[key] = value
    # Which way this cell type is fitted. A C2C12 is fitted carried
    # forward -- one component to a stretch, each modulus found in turn and
    # then held -- and that is where it starts, whatever the cell before it
    # was fitted with. A cell type the four regimes are not written for has
    # only the other way, so it gets that.
    st.session_state["c2c12_fit_mode"] = (
        PIECEWISE_MODE if (HAS_PIECEWISE and name in PIECEWISE_CELL_TYPES)
        else ADVANCED_MODE
    )
    st.session_state["pw_reach_note"] = None
    # The boundaries and everything measured for the previous cell type.
    st.session_state["segment_break_1"] = DEFAULTS["segment_break_1"]
    st.session_state["segment_break_2"] = DEFAULTS["segment_break_2"]
    for key in ("arrangement_search", "composition_search", "component_search",
                "confinement_scan", "hypothesis_search", "exploration"):
        st.session_state[key] = None
    st.session_state.pop("_auto_picked", None)
    for key in list(st.session_state.keys()):
        if key.startswith("window_"):
            del st.session_state[key]


def active_terms():
    """Terms currently switched on, always outermost first.

    Walks every term the current cell type has, so a cell type with an
    in-plane spring reports it and one without never can, whatever is left
    behind in session state from an earlier cell.

    A knockout is enforced here rather than by unticking a box. The
    experiment removed that protein from the cell, so the model must not
    have it either, and a genotype is not something a stray tick should be
    able to overrule.
    """
    available = terms_for(st.session_state.get("cell_type"))
    if fixed_cell_on():
        # Only the Hertzian terms, whichever of them are ticked, and never
        # the shell terms: a fixed cell has no separate membrane to stretch.
        # At least one has to be on, or there is no model at all.
        on = tuple(t for t in available
                   if st.session_state.get(f"use_{t}", False))
        return on or ("interior",)
    on = tuple(t for t in available if st.session_state.get(f"use_{t}", False))
    if str(st.session_state.get("membrane_protein", "")).startswith("Removed"):
        on = tuple(t for t in on if t != "tension")
    return on


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
    for term in ALL_TERMS:
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
            # The pair the widgets read, and the mirror kept for anything
            # that still stores a preset as one combined window.
            pending["window_combined"] = value
            pending["window_start"] = round(value[0], 4)
            pending["window_end"] = round(value[1], 4)

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
    # The fitted range is a pair everywhere now, so a drag sets both ends.
    # It used to set only the far one, because the near one could not be
    # moved; dragging a box and having half of it ignored is worse than
    # either behaviour.
    if key == "window_range":
        lo_now = round(float(window[0]), 4)
        hi_now = round(float(window[1]), 4)
        if (st.session_state.get("window_start"),
                st.session_state.get("window_end")) != (lo_now, hi_now):
            st.session_state["_pending_settings"] = {
                "window_start": lo_now, "window_end": hi_now,
            }
            st.rerun()
        return
    value = window[1] if key == "window_end" else window
    if st.session_state.get(key) != value:
        st.session_state["_pending_settings"] = {key: value}
        st.rerun()


# The bar and the two boxes are three ways of saying the same thing, so one
# pair of numbers is the truth and all three widgets are drawn from it. The
# alternative, letting each widget own its own value, is what produces a box
# reading 0.42 beside a bar sitting at 0.60.
def range_bounds(lo_key, hi_key, floor, ceiling, step):
    """The stored range, clamped so it is inside the data and lo < hi."""
    step = float(step)
    floor, ceiling = float(floor), float(ceiling)
    if ceiling - floor < step:
        ceiling = floor + step
    # Rounded first and clamped second. The other order lets a value a
    # hair under the top of the data round up past it, and a slider whose
    # value is above its own maximum is a hard error, not a warning.
    lo = float(np.clip(round(_as_float(st.session_state.get(lo_key), floor), 4),
                       floor, ceiling - step))
    hi = float(np.clip(round(_as_float(st.session_state.get(hi_key), ceiling), 4),
                       lo + step, ceiling))
    return lo, hi


def _as_float(value, fallback):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return float(fallback) if not np.isfinite(out) else out


def suggested_range_note(lo, hi):
    """
    Say that the range on screen is the one the app suggested, and why.

    The range is chosen for you the moment a curve loads, and a number
    chosen for somebody without being announced is one they have to guess
    the provenance of. Once it has been moved, the same line says so and
    offers the suggestion back, because "what did it start as" is otherwise
    unanswerable without reloading the file.
    """
    suggested = st.session_state.get("_suggested_window")
    if not suggested:
        return
    start = float(suggested["start"])
    end = float(suggested["end"])
    why = (
        ("starting " + str(suggested.get("why_start") or "").strip()
         if suggested.get("bad_contact")
         else "starting at first contact, ε = 0")
        + ", and ending at " + (str(suggested.get("why_end")
                                    or "the end of the curve").strip())
    )
    if abs(lo - start) < 5e-4 and abs(hi - end) < 5e-4:
        st.caption(
            f"✅ **This is the suggested range**, ε {start:.3f} to "
            f"{end:.3f}, chosen from this curve when it loaded: {why}. "
            "Move either end and it becomes yours."
        )
        return
    left, right = st.columns([2.4, 1])
    with left:
        st.caption(
            f"This range is yours, not the suggested one. The suggestion "
            f"was ε {start:.3f} to {end:.3f}: {why}."
        )
    with right:
        if st.button("↺ Back to the suggested range",
                     key="restore_suggested_range", **STRETCH):
            rerun_keeping_settings({
                "window_start": start,
                "window_end": end,
                "window_combined": (start, end),
            })


def epsilon_range_control(lo_key, hi_key, floor, ceiling, step,
                          label="Fitted range", help_text=None,
                          boxes=True, prefix=""):
    """
    One fitted range, set by a bar or by typing either end.

    ``lo_key`` and ``hi_key`` are plain session keys, not widget keys, so
    anything else in the app (a preset, a drag on the plot, the winner of a
    search) can write them at any time and the widgets follow on the next
    pass. The widgets keep their own keys, which are written here before
    they are built and read back in the callbacks below.

    Returns the pair actually in force, already clamped.
    """
    step = float(step)
    floor, ceiling = float(floor), float(ceiling)
    if ceiling - floor < step:
        ceiling = floor + step
    bar_key = f"{prefix}{lo_key}__bar"
    lo_box, hi_box = f"{prefix}{lo_key}__box", f"{prefix}{hi_key}__box"
    lo, hi = range_bounds(lo_key, hi_key, floor, ceiling, step)

    def _store(a, b):
        a = float(np.clip(round(float(a), 4), floor, ceiling - step))
        b = float(np.clip(round(float(b), 4), a + step, ceiling))
        st.session_state[lo_key] = a
        st.session_state[hi_key] = b

    def _from_bar():
        pair = st.session_state.get(bar_key) or (lo, hi)
        _store(pair[0], pair[1])

    def _from_boxes():
        a = _as_float(st.session_state.get(lo_box), lo)
        b = _as_float(st.session_state.get(hi_box), hi)
        # If the two ends cross, the one that was just typed wins and the
        # other gives way. Snapping back the number someone has this second
        # finished typing is the most annoying way to handle it.
        if b <= a:
            if abs(a - lo) > 1e-9:
                b = min(a + step, ceiling)
            else:
                a = max(b - step, floor)
        _store(a, b)

    # Written back, so the stored pair is the one on the page rather than a
    # wider one left over from another curve that the widgets are quietly
    # clamping. Anything reading the range later reads what is shown.
    st.session_state[lo_key], st.session_state[hi_key] = lo, hi
    st.session_state[bar_key] = (lo, hi)
    st.session_state[lo_box], st.session_state[hi_box] = lo, hi
    st.slider(
        label, min_value=floor, max_value=ceiling, step=step,
        key=bar_key, on_change=_from_bar, help=help_text,
    )
    if boxes:
        near, far = st.columns(2)
        with near:
            st.number_input(
                "from ε", min_value=floor, max_value=ceiling, step=step,
                format="%.3f", key=lo_box, on_change=_from_boxes,
                help="Type the near end. Normally 0: the model describes a "
                     "cell from first contact.",
            )
        with far:
            st.number_input(
                "to ε", min_value=floor, max_value=ceiling, step=step,
                format="%.3f", key=hi_box, on_change=_from_boxes,
                help="Type the far end. 1.0 is the whole curve.",
            )
    return range_bounds(lo_key, hi_key, floor, ceiling, step)


def element_window_key(term):
    """Session key holding one element's own window."""
    return f"element_window_{term}"


def element_windows(terms=None, lo=0.0, hi=1.0):
    """
    Each element's own stretch of the squash, as the page has it set.

    Returns None when the per-element windows are switched off, which is
    what tells the model to place the elements from the composition's own
    boundaries instead. Anything else would silently turn every fit into a
    different model.
    """
    terms = tuple(terms if terms is not None else active_terms())
    out = {}
    for term in terms:
        window = st.session_state.get(element_window_key(term))
        if not window:
            continue
        try:
            a, b = float(window[0]), float(window[1])
        except (TypeError, ValueError, IndexError):
            continue
        # Clamped from zero, not from the fitted range's start: an onset is
        # where a law begins, and that is contact.
        a = float(np.clip(a, 0.0, hi))
        b = float(np.clip(b, a, hi))
        if b > a:
            out[term] = (round(a, 4), round(b, 4))
    return out or None


def default_element_window(term, lo, hi, e1, e2, membrane="freeze",
                           cyto_start="break"):
    """
    Where the composition would put this component, before anyone moves it.

    A boundary outside the fitted range does not narrow anything: a shell
    told to hold at ε₁ = 0.15 on a curve fitted from 0.27 never reaches that
    boundary inside the range, so it acts across the whole of it. Read
    literally, that window collapsed to nothing and took the fit with it,
    which is what a curve with a bad contact looked like.
    """
    # Contact is ε = 0, not wherever the fitted range happens to start. A
    # curve fitted from 0.27 because the approach misbehaved there is still
    # a cell that was squashed from zero, and a shell whose law is measured
    # from 0.27 instead is a different model with a different modulus. That
    # was worth 0.08 of R² on the one reference curve with a bad contact.
    lo, hi = 0.0, float(hi)
    span = max(hi - lo, 1e-6)
    # There is one way of fitting on this page and it is "each on its own
    # stretch": the membrane on 0 to ε₁, the cytoskeleton on ε₁ to ε₂, the
    # nuclear envelope on ε₂ to ε₃ and the inside of the nucleus from ε₃
    # on. That is what a window is, so it is what a window starts as.
    stretches = own_stretch_windows()
    if term in stretches:
        a, b = stretches[term]
        return (float(np.clip(a, lo, hi)),
                float(np.clip(max(b, a + 0.01 * span), lo, hi)))

    def onset(value):
        # An onset past the far end is a component never reached inside the
        # range. It keeps a sliver at the top rather than an empty window,
        # so the fit hands it zero rather than failing.
        return float(np.clip(value, lo, hi - 0.01 * span))

    # An element the cell type is known to load throughout gets the whole
    # squash by way of its arrangement, not by an override here: the
    # composition defaults put a C2C12's shell on "keeps stiffening" and its
    # cytoskeleton on "from the very start", which lands on (0, hi) two
    # lines below. Overriding it here instead would make the arrangement
    # radios inert, which is a worse kind of prior: one that cannot be
    # argued with.
    if term in ("nucleus", "nucleus_shell"):
        return (onset(e2), hi)
    if term == "interior" and cyto_start == "break":
        return (onset(e1), hi)
    if term in ("membrane", "tension"):
        if membrane == "late":
            return (onset(e1), hi)
        if membrane == "freeze":
            end = float(e1)
            # Held at a boundary the range never reaches: it is still
            # stretching everywhere in view.
            return (lo, hi if end <= lo + 0.01 * span
                    else float(np.clip(end, lo + 0.01 * span, hi)))
    return (lo, hi)


# ------------------------------------------------ the components table --
#
# One table, one row shape, whichever way the load is shared. A row is a
# small bordered card so that four of them side by side read as four
# things rather than as one block of text:
#
#   ☑ 🟥 Membrane · Eₘ
#   K_shell [min(x,u) − s]³
#   [=========== the range it acts over ===========]
#   [s, u] = [2.0, 91.2] %                 Eₘ = 0.42 MPa
#
# Every way of sharing the load builds its rows and hands them here, so a
# person switching from one to another finds the same four components laid
# out the same way, not a new interface.

COMPONENT_TABLE_HEAD = ("tick · component · symbol · law · the range it acts "
                        "over · what the fit made of it")


def component_row_table(rows, columns=2):
    """Draw the components table and return {key: slot} for the values."""
    # A tick box with no value in session state comes up empty, whatever
    # the app meant it to be, and an empty box beside a component the fit
    # has just measured is the page lying about its own model. So every
    # row's state is written before its box is drawn: a component this
    # page fits by default arrives ticked, on the first run and on every
    # run after a reset that cleared it.
    for row in rows:
        if row["tick_key"] not in st.session_state:
            st.session_state[row["tick_key"]] = bool(
                row.get("default_on", True))
    st.caption(COMPONENT_TABLE_HEAD)
    grid = st.columns(columns, gap="medium") if columns > 1 else None
    slots = {}
    for index, row in enumerate(rows):
        cell = grid[index % columns] if grid else st.container()
        with cell:
            card = st.container(border=True)
            with card:
                on = bool(st.session_state.get(
                    row["tick_key"], row.get("default_on", True)))
                st.checkbox(f"{row['label']}  ·  {row['symbol']}",
                            key=row["tick_key"], help=row.get("help"))
                st.caption(row["law"])
                slider = row.get("slider")
                if slider:
                    st.slider(**dict(slider, disabled=slider.get(
                        "disabled", not on)))
                foot_l, foot_r = st.columns([1.25, 1])
                with foot_l:
                    st.caption(row.get("range_text", "") if on else "off")
                with foot_r:
                    slots[row["key"]] = st.empty()
                for extra in row.get("extras", ()):
                    extra(on)
    return slots


def component_controls(terms, names, lo, hi, step, e1, e2,
                       membrane="freeze", cyto_start="break"):
    """
    Every component of the cell: whether it is in the model, and where it acts.

    One row each, because they are one decision. A component is in the fit
    or it is not, and if it is, it carries load over some stretch of the
    squash and not the rest; splitting those across two parts of the page
    made a person hold one in their head while setting the other.

    The rows are drawn by the same table the regime-by-regime fit uses, so
    the components look and behave the same whichever way the load is
    shared. The ranges follow the fit until somebody takes them over. That
    is the honest default: epsilon 1 and epsilon 2 are found from the
    curve, and a range that did not move with them would be describing a
    model that is no longer the one being fitted.
    """
    if not terms:
        return None
    rows = []
    for term in terms:
        key = element_window_key(term)
        automatic = default_element_window(term, lo, hi, e1, e2, membrane,
                                           cyto_start)
        # Until a range is moved by hand it follows the boundaries. Moving
        # one marks it, and from then on it is that person's number: a fit
        # that shifts ε₁ must not quietly undo what they set.
        if not st.session_state.get(f"_window_touched_{term}") \
                or not st.session_state.get(key):
            st.session_state[key] = automatic

        # The bar runs from zero, not from where the fitted range starts:
        # a law is measured from contact, and clamping it up to the range
        # start is what turned a curve with a bad contact into a fit with
        # no membrane in it at all.
        bar = f"{key}__bar"
        pair = st.session_state[key]
        a = float(np.clip(pair[0], 0.0, hi - step))
        b = float(np.clip(pair[1], a + step, hi))
        st.session_state[bar] = (a, b)

        def _store(term=term, key=key, bar=bar):
            got = st.session_state.get(bar)
            if got:
                st.session_state[key] = (
                    round(float(got[0]), 4), round(float(got[1]), 4)
                )
                st.session_state[f"_window_touched_{term}"] = True

        law = (MATERIAL_LAWS.get(term) or {}).get("law") or shape_mark(term)
        rows.append({
            "key": term, "tick_key": f"use_{term}", "label": names[term][0],
            "default_on": bool(DEFAULT_TERMS_BY_TYPE.get(
                st.session_state.get("cell_type"), {}).get(term, True)),
            "symbol": TERM_SYMBOLS.get(term, term),
            "law": f"{shape_mark(term)}  ·  {law}",
            "help": names[term][1],
            "range_text": f"acts over ε = {a:.3f} to {b:.3f}",
            "slider": {
                "label": f"acts over ε · {plain_name(term)}",
                "min_value": 0.0, "max_value": float(hi), "step": float(step),
                "key": bar, "on_change": _store,
                "label_visibility": "collapsed",
                "help": "Where this component starts carrying load and where "
                        "it stops taking more. Past the far end it holds what "
                        "it reached rather than vanishing, so the curve has "
                        "no step in it.",
            },
        })
    component_row_table(rows, columns=2)

    automatic = {
        term: default_element_window(term, lo, hi, e1, e2, membrane,
                                     cyto_start)
        for term in terms
    }
    drifted = [
        term for term in terms
        if st.session_state.get(f"use_{term}")
        and st.session_state.get(f"_window_touched_{term}")
        and st.session_state.get(element_window_key(term)) != automatic[term]
    ]
    if drifted:
        st.caption(
            "Moved by hand, so the boundaries no longer place "
            + ", ".join(plain_name(t) for t in drifted)
            + (" is" if len(drifted) == 1 else " are")
            + " away from where the fit would put "
            + ("it." if len(drifted) == 1 else "them.")
        )
        if st.button("↺ Reset windows to ε₁, ε₂",
                     key="reset_element_windows", **STRETCH):
            for term in terms:
                st.session_state.pop(f"_window_touched_{term}", None)
            rerun_keeping_settings({
                element_window_key(term): window
                for term, window in automatic.items()
            })
    else:
        st.caption(
            "A range is where a component takes on load: it starts at the "
            "near edge and stops taking **more** at the far edge, holding "
            "what it reached to the end of the curve, so nothing ever drops "
            "out of the force. Each one follows the boundaries until you "
            "move it: ε₁ and ε₂ place them, and the ranges move when those "
            "do."
            + ("  A C2C12 usually goes membrane first, then the "
               "cytoskeleton — or both together from contact — then the "
               "nuclear envelope and what it contains, met together."
               if str(st.session_state.get("cell_type", "")).startswith("Myoblast")
               else "")
        )

    # After the components, because it is a statement about the sample
    # rather than one more component, and because ticking it takes all of
    # them away.
    fixed_cell_control()
    return element_windows(active_terms(), lo, hi)


def fit_at_the_current_settings(model, lo, hi, terms):
    """
    Fit the curve as the page has it set, and move nothing.

    The page refits on every pass, so what this adds is the deliberate act
    and the confinement: q is a fitted quantity, not a boundary, and it is
    measured here at the boundaries already on screen rather than at
    boundaries this button chose for itself.
    """
    if not terms:
        return
    if wants_confinement() and hasattr(model, "scan_confinement"):
        with st.spinner("Measuring the confinement at these boundaries…"):
            try:
                scan = model.scan_confinement(
                    lo, hi,
                    e1=float(st.session_state["segment_break_1"]),
                    e2=float(st.session_state["segment_break_2"]),
                    membrane=MEMBRANE_CHOICES.get(
                        st.session_state["membrane_after_break"], "freeze"),
                    cyto_start=CYTO_CHOICES.get(
                        st.session_state["cyto_starts_at"], "break"),
                    use_nucleus="nucleus" in terms,
                    use_tension="tension" in terms,
                    use_nucleus_shell="nucleus_shell" in terms,
                    use_cortex="cortex" in terms,
                    weighting=st.session_state["weighting"],
                )
            except Exception as exc:  # pragma: no cover - defensive
                scan = {"success": False, "error": str(exc)}
        if scan.get("success"):
            st.session_state["confinement_scan"] = scan
            st.session_state["_fitted_on_purpose"] = True
            rerun_keeping_settings({"confinement": round(float(scan["q"]), 2),
                                    "_fit_from_curve": True})
    st.session_state["_fitted_on_purpose"] = True
    rerun_keeping_settings({"_fit_from_curve": True})


def fit_verdict(fit=None):
    """What the fit on screen came to, at the settings it was given."""
    if fit is None:
        fit = st.session_state.get("_last_fit")
    if not (fit and fit.get("success")):
        st.caption(
            "Press **▶ Fit & plot** on the control board and the answer "
            "appears here, with the fitted curve."
        )
        return
    chi = fit.get("chi_squared_reduced", float("nan"))
    # The boundaries this fit has, which are the lines on the curve below.
    where = [f"{name} = {value:.3f}" for value, name in fit_edges(fit)]
    said = (
        (f"`fit {fit['fit_id']}` · " if fit.get("fit_id") else "")
        + f"**R² = {fit.get('r_squared', float('nan')):.5f}**"
        + (f", χ²/dof = {chi:.3g}" if np.isfinite(chi) else "")
        + " at " + (", ".join(where) if where else "the range shown")
        + f", over ε {float(fit['epsilon_range'][0]):.3f} to "
        f"{float(fit['epsilon_range'][1]):.3f}."
    )
    if np.isfinite(chi) and chi > 10:
        st.info(
            said + " χ²/dof well above 1 means the model is missing "
            "something real here, not that the numbers are noisy. Try "
            "estimating ε₁, ε₂ on the board, or another arrangement under "
            "Advanced, then ▶ Fit & plot."
        )
    else:
        st.success(said)


# ====================================================== four-regime C2C12 ==
#
# The C2C12 fit a person reaches for first: four regimes met in turn, each
# anchored to the force the one before it ended on, so the curve is
# continuous everywhere and the fit has nothing to converge but four small
# convex problems. No components to tick, no windows, no searches. The
# curve loads and the answer is on the page.

PIECEWISE_MODE = "4-regime piecewise (C2C12)"
ADVANCED_MODE = "Spring-network models (advanced)"
FIT_MODES = (PIECEWISE_MODE, ADVANCED_MODE)
PIECEWISE_CELL_TYPES = ("Myoblast (C2C12)",)
PW_BOUNDARY_KEYS = ("pw_b1", "pw_b2", "pw_b3", "pw_end")
PW_BAND_KEYS = (("pw_band1_lo", "pw_band1_hi"), ("pw_band2_lo", "pw_band2_hi"))
PW_SPAN_KEYS = ("pw_span_lo", "pw_span_hi")
# The names the boundaries go by everywhere: on the inputs, on the curve,
# in the results and in the sheet (ε₁ and ε₂ are its two boundary columns).
EPS_NAMES = ("ε₁", "ε₂", "ε₃")
EPS_ROLES = ("R2 starts: cell stretch + cytoskeleton",
             "R3 starts: nuclear envelope reached",
             "R4 starts: dense intranuclear packing")
PW_COLORS = {"R1": "#7f8c8d", "R2": "#2ca02c", "R3": "#1f77b4", "R4": "#9467bd"}
PW_BANDS = {"R1": "rgba(127,140,141,0.10)", "R2": "rgba(44,160,44,0.08)",
            "R3": "rgba(31,119,180,0.08)", "R4": "rgba(148,103,189,0.08)"}
# Where each coefficient's modulus goes in the record and the sheet, which
# were laid out for the spring-network models. The two that have no column
# there get their own.
PW_TERM_OF = {
    "K_shell": "membrane", "K_cyto": "interior",
    "K_nucleus": "nucleus_shell", "K_core": "nucleus",
}


# ---- the control board and the fit it last applied ------------------------
#
# The control board holds what the next fit will use; the plot and the
# results show the fit last applied with ▶ Fit & plot. Everything the fit
# reads (ε, the components on or off, their ranges, p₀ and bounds, the
# membrane's reach) goes through _pw_get, which reads the applied values
# while the applied side of the page is being drawn, and the board's
# otherwise.
PW_APPLIED_KEYS = ("pw_b1", "pw_b2", "pw_b3", "pw_end", "pw_settings",
                   "pw_until", "pw_membrane_throughout", "pw_target_r2",
                   "pw_style", "pw_best_carry",
                   "pw_use_K_shell", "pw_use_K_cyto",
                   "pw_use_K_nucleus", "pw_use_K_core")
_PW_SOURCE = [None]


def _pw_get(key, default=None):
    """A fit parameter: the applied one while drawing the fit, else the board's."""
    source = _PW_SOURCE[0]
    if source is not None and key in source:
        return source[key]
    return st.session_state.get(key, default)


def pw_board_values():
    """The fit parameters as the control board has them now."""
    import copy
    return {key: copy.deepcopy(st.session_state.get(key, DEFAULTS.get(key)))
            for key in PW_APPLIED_KEYS}


class pw_applied_values:
    """Within this block every fit parameter is read from ``values``."""

    def __init__(self, values):
        self.values = values

    def __enter__(self):
        _PW_SOURCE[0] = self.values
        return self.values

    def __exit__(self, *exc):
        _PW_SOURCE[0] = None
        return False


def piecewise_offered(cell_type=None):
    """Whether this cell type has the four-regime fit at all."""
    if cell_type is None:
        cell_type = st.session_state.get("cell_type")
    # A fixed cell is one cross-linked solid, not four regimes of a living
    # one, so ticking it hands the curve to the models that describe that.
    return (bool(HAS_PIECEWISE) and cell_type in PIECEWISE_CELL_TYPES
            and not fixed_cell_on())


def piecewise_on():
    """Whether the curve on the page is being fitted with the four regimes."""
    return (
        piecewise_offered()
        and st.session_state.get("c2c12_fit_mode",
                                 DEFAULTS["c2c12_fit_mode"]) == PIECEWISE_MODE
    )


def fit_mode_control():
    """
    Which way the fit goes this run, and why the four-regime one is missing.

    The choice itself is not made here any more: it is one of the ways the
    components can share the load, chosen at the top of the parameter
    block (load_sharing_control). This only decides which of the two
    layouts draws it.
    """
    if (PIECEWISE_PROBLEM and st.session_state.get("cell_type")
            in PIECEWISE_CELL_TYPES):
        piecewise_problem_note()
    if not piecewise_offered():
        return False
    if st.session_state.get("c2c12_fit_mode") not in FIT_MODES:
        st.session_state["c2c12_fit_mode"] = DEFAULTS["c2c12_fit_mode"]
    return piecewise_on()


# One spring network, several ways for its components to share the load.
# Piecewise is one of them: the same membrane, cytoskeleton, nuclear
# envelope and nucleus, taking over regime by regime, each regime starting
# from the force the one before it ended on.
PW_SHARE = "Piecewise (regime by regime, each anchored to the last)"
# The second of the two ways to fit: every element solved on its own
# stretch of the squash and acting nowhere else.
OWN_STRETCH = "Each on its own stretch (fitted only inside its own range)"
SHARING_MATHS = {
    OWN_STRETCH:
        r"F(\varepsilon) = \sum_k a_k E_k \,\langle \varepsilon - s_k"
        r"\rangle^{p_k}\,\mathbb{1}_{[s_k,\,u_k]}(\varepsilon), \qquad "
        r"[s_k, u_k] = [\varepsilon_{k-1},\, \varepsilon_k]",
    PW_SHARE: r"F(x) = \hat F(\varepsilon_{k-1}) + \sum_{j \in k} \theta_j\,"
              r"\phi_j(x) + C_k(x),\;\; x \in [\varepsilon_{k-1}, \varepsilon_k)",
    "Segmented (each part takes over in turn)":
        r"F(\varepsilon) = \sum_k a_k E_k \langle \varepsilon - s_k \rangle^{p_k}",
    "Side by side (every element acts everywhere)":
        r"F(\varepsilon) = \sum_k a_k E_k\,\varepsilon^{p_k}",
    "Stacked (elements in line)":
        r"\varepsilon(F) = \sum_k (F / a_k E_k)^{1/p_k}",
}
SHARING_HELP = {
    PW_SHARE:
        "**Found in turn, then held.** The membrane's modulus is fitted "
        "on the first stretch, 0 to ε₁. It is then **held at that value** "
        "and goes on carrying load, and only the cytoskeleton's modulus is "
        "fitted on ε₁ to ε₂. Both are then held and only the nuclear "
        "envelope is fitted on ε₂ to ε₃, and then all three are held and "
        "only the inside of the nucleus is fitted from ε₃ on. Each stretch "
        "starts from the force the one before it ended on, so the curve "
        "has no step in it. Four small fits, one unknown each.",
    OWN_STRETCH:
        "**Each on its own stretch.** The membrane is fitted on 0 to ε₁, "
        "the cytoskeleton on ε₁ to ε₂, the nuclear envelope on ε₂ to ε₃, "
        "the inside of the nucleus from ε₃ on — and each one acts **only "
        "inside its own stretch**. A component contributes nothing before "
        "its range and nothing after it, so at 20 % a cytoskeleton whose "
        "range is 30 to 50 % is exactly zero. Four independent fits, one "
        "unknown each, with no force carried across a boundary.",
}
# The list is read while deciding, so every entry is one short line: the
# shape of the sum, not a sentence about it. The sentence is written under
# the chosen one, where it is actually read.
SHARING_SHORT = {
    PW_SHARE: "① Carried forward  ·  each modulus found in turn, then held",
    OWN_STRETCH: "② Each on its own stretch  ·  nothing carried across",
}
# The four the app measures, under the name each way of sharing gives them,
# so a tick survives a change of model: nothing about which components the
# cell has depends on how their forces are added up.
SHARED_TICKS = (("pw_use_K_shell", "use_membrane"),
                ("pw_use_K_cyto", "use_interior"),
                ("pw_use_K_nucleus", "use_nucleus_shell"),
                ("pw_use_K_core", "use_nucleus"))


def _carry_the_ticks(to_piecewise):
    """Carry which components are ticked across a change of sharing."""
    for pw_key, legacy_key in SHARED_TICKS:
        source, target = ((legacy_key, pw_key) if to_piecewise
                          else (pw_key, legacy_key))
        st.session_state[target] = bool(st.session_state.get(source, True))


# Which component is met first, second, third and fourth. It is a choice,
# not a property of the model: the same four elements can be met in any
# order, and both ways of fitting take the order from here. The default is
# the C2C12's own: the membrane at contact, the cytoskeleton next, then the
# nuclear envelope, then what it contains.
COMPONENT_ORDER_DEFAULT = ("membrane", "interior", "nucleus_shell", "nucleus")
# The name each component goes by on either side: the spring network's term
# and the four-regime engine's coefficient.
ORDER_COEFFICIENT = {"membrane": "K_shell", "interior": "K_cyto",
                     "nucleus_shell": "K_nucleus", "nucleus": "K_core"}
ONSET_NAMES = ("at contact, ε = 0", "at ε₁", "at ε₂", "at ε₃")


def component_order():
    """The four components in the order they are met, validated."""
    got = st.session_state.get("component_order") or []
    order = [t for t in got if t in COMPONENT_ORDER_DEFAULT]
    if len(set(order)) != len(COMPONENT_ORDER_DEFAULT):
        return COMPONENT_ORDER_DEFAULT
    return tuple(dict.fromkeys(order))


def component_order_control():
    """
    Pick which component is met first, second, third and fourth.

    One row per position, because that is the question: what starts at
    contact, what joins at ε₁, what joins at ε₂, what joins at ε₃. Both
    ways of fitting read it, so the order is the model and not a property
    of one of them.
    """
    names = components_for(st.session_state.get("cell_type"))
    order = list(component_order())
    picked = []
    cols = st.columns(len(order))
    for index, (col, term) in enumerate(zip(cols, order)):
        with col:
            choice = st.selectbox(
                f"{index + 1}ᵉ · starts {ONSET_NAMES[index]}",
                list(COMPONENT_ORDER_DEFAULT),
                index=list(COMPONENT_ORDER_DEFAULT).index(term),
                format_func=lambda t: names[t][0],
                key=f"component_order_{index}",
            )
            picked.append(choice)
    if len(set(picked)) == len(picked):
        if tuple(picked) != tuple(order):
            st.session_state["component_order"] = list(picked)
            rerun_keeping_settings()
    else:
        st.caption("⚠️ Two positions name the same component, so the order "
                   "below is still **"
                   + " → ".join(names[t][0] for t in order)
                   + "**. Give each position a component of its own.")
    st.caption("Met in this order: "
               + " → ".join(f"{names[t][0]} {ONSET_NAMES[i]}"
                            for i, t in enumerate(component_order())))


def pw_regimes():
    """
    The four regimes, built from the order the components are met in.

    The engine is handed one regime per component: the first from contact
    with a free intercept, then one at each boundary. Nothing is met in
    pairs unless the order says so, because every position holds exactly
    one component.
    """
    if not PW_REGIMES:
        return ()
    terms = {t.name: t for regime in PW_REGIMES for t in regime.terms}
    out = []
    for index, (regime, term) in enumerate(zip(PW_REGIMES, component_order())):
        name = ORDER_COEFFICIENT[term]
        if name not in terms:
            return PW_REGIMES
        out.append(_dataclasses.replace(
            regime, terms=(terms[name],), free_offset=(index == 0),
            title=PW_COMPONENT_TITLES.get(name, regime.title),
            equation="",
        ))
    return tuple(out)


def pw_start_index():
    """Which boundary each coefficient starts at, in the chosen order."""
    return {ORDER_COEFFICIENT[term]: index
            for index, term in enumerate(component_order())}


def sharing_options():
    """
    The three ways of fitting the four components, and there are three.

    All three fit the same four components over the same four stretches,
    one unknown at a time, with the same boundaries. They differ in one
    thing: how far each component goes on carrying load after the stretch
    it was fitted on. A cell type the four regimes are not written for has
    the spring network instead.
    """
    if piecewise_offered():
        return [PW_STYLES[key] for key in PW_STYLES]
    return [OWN_STRETCH]


def style_of_label(label):
    """The style key behind one of the three labels."""
    for key, said in PW_STYLES.items():
        if said == label:
            return key
    return None


def current_sharing():
    """The way the page is fitting now."""
    if piecewise_on():
        return PW_STYLES[piecewise_style()]
    return OWN_STRETCH


def own_stretch_boundaries():
    """(0, ε₁, ε₂, ε₃, end) as fractions, for the stretch-by-stretch fit."""
    end = float(st.session_state.get("window_end", 1.0))
    e1 = float(st.session_state.get("segment_break_1", 0.05))
    e2 = float(st.session_state.get("segment_break_2", 0.40))
    e3 = float(st.session_state.get("pw_b3", 75.0)) / 100.0
    e1 = float(np.clip(e1, 0.0, end))
    e2 = float(np.clip(e2, e1, end))
    e3 = float(np.clip(e3, e2, end))
    return 0.0, e1, e2, e3, end


def own_stretch_windows():
    """One stretch each, in the order the components are met."""
    edges = own_stretch_boundaries()
    return {term: (edges[index], edges[index + 1])
            for index, term in enumerate(component_order())}


def _load_sharing_changed():
    """Switch the fit, carrying the boundaries over: they are the cell's."""
    chosen = st.session_state.get("load_sharing")
    was_piecewise = piecewise_on()
    style = style_of_label(chosen)
    if style:
        st.session_state["pw_style"] = style
        st.session_state["pw_style_note"] = None
        # The mixture is searched again for the curve on the page, and the
        # fit that follows is made of what it finds.
        st.session_state["pw_best_carry"] = None
        st.session_state["_pw_restyle"] = True
    if style:
        if not was_piecewise:
            # ε₂, where the nucleus is reached, means the same in both. ε₁
            # comes across only where it is a contact boundary here too.
            (l1, h1), (l2, h2), _b3 = piecewise_bands()
            e1 = 100.0 * float(st.session_state.get("segment_break_1", 0.05))
            e2 = 100.0 * float(st.session_state.get("segment_break_2", 0.5))
            if l2 <= e2 <= h2 and e2 < float(st.session_state.get("pw_b3", 75.0)):
                st.session_state["pw_b2"] = round(e2, 2)
            if l1 <= e1 <= h1:
                st.session_state["pw_b1"] = round(e1, 2)
            _carry_the_ticks(to_piecewise=True)
        st.session_state["c2c12_fit_mode"] = PIECEWISE_MODE
        return
    if was_piecewise:
        st.session_state["segment_break_1"] = round(
            float(st.session_state.get("pw_b1", 5.0)) / 100.0, 4)
        st.session_state["segment_break_2"] = round(
            float(st.session_state.get("pw_b2", 50.0)) / 100.0, 4)
        _carry_the_ticks(to_piecewise=False)
    st.session_state["c2c12_fit_mode"] = ADVANCED_MODE
    # The other way is the segmented spring network with every element cut
    # down to its own stretch: the membrane on 0 to ε₁, the cytoskeleton on
    # ε₁ to ε₂, the nuclear envelope on ε₂ to ε₃, the inside of the nucleus
    # from ε₃ on, and nothing acting outside its own range.
    st.session_state["model_kind"] = "Segmented (each part takes over in turn)"
    st.session_state["membrane_after_break"] = "holds what it reached"
    st.session_state["cyto_starts_at"] = "at ε₁"
    for term, window in own_stretch_windows().items():
        st.session_state[element_window_key(term)] = (round(window[0], 4),
                                                      round(window[1], 4))
        st.session_state[f"_window_touched_{term}"] = True


def load_sharing_control():
    """
    How the components share the load: the first parameter of the fit.

    One model, one list of components, one set of boundaries; what changes
    is how the components add up. Piecewise sits in the same list as side
    by side, stacked and segmented, because that is all it is.
    """
    options = sharing_options()
    st.session_state["load_sharing"] = current_sharing()
    if st.session_state["load_sharing"] not in options:
        st.session_state["load_sharing"] = options[0]
    # A list one line per entry rather than a column of paragraphs: seven
    # ways of adding the same forces up are scanned, not read, and the one
    # sentence that matters is the one under whichever is chosen.
    st.selectbox(
        "How the cell is fitted", options, key="load_sharing",
        format_func=lambda name: (
            PW_STYLE_SHORT.get(style_of_label(name))
            or SHARING_SHORT.get(name, name)),
        on_change=_load_sharing_changed,
        help="The same four components, the same ticks, the same "
        "boundaries and the same one-unknown-at-a-time solve in all three. "
        "The only difference is how far each component goes on carrying "
        "load after the stretch it was fitted on.",
    )
    chosen = st.session_state["load_sharing"]
    style = style_of_label(chosen)
    if style:
        st.latex(r"\phi_j(x) = \big[\min(x,\,u_j) - s_j\big]_+^{\,p_j}")
        st.latex(PW_STYLE_MATHS[style])
        st.caption(PW_STYLE_HELP[style])
    else:
        if chosen in SHARING_MATHS:
            st.latex(SHARING_MATHS[chosen])
        st.caption(SHARING_HELP.get(chosen) or MODELS.get(chosen, ""))
    st.caption("Your ticks in the table below stay as they are when you "
               "change this.")


def piecewise_boundaries():
    """(0, b1, b2, b3, end) in percent, as the page has them."""
    return (0.0,) + tuple(
        float(_pw_get(key, DEFAULTS[key])) for key in PW_BOUNDARY_KEYS
    )


# The components of the four-regime model, in the order the plates meet
# them: (coefficient, name, modulus, colour, law). The same list builds the
# range rows, the curves on the plot and the range track under it, so a
# component cannot appear in one and not the others.
PW_COMPONENTS = (
    # The square in each name is the colour of that component's line and bar
    # on the plot. The names and symbols are the spring network's own, for
    # the four it shares with every other way of sharing the load. Four
    # components, four ticks, four bars, four rows in the results: what is
    # ticked is what is fitted and what is drawn, with nothing else in the
    # model.
    ("K_shell", "🟥 Membrane", "Eₘ", "#d62728",
     r"$K_{shell}\,[\min(x,u)-s]_+^{3}$"),
    ("K_cyto", "🟧 Cytoskeleton", "Ec", "#ff7f0e",
     r"$K_{cyto}\,[\min(x,u)-s]_+^{3/2}$"),
    ("K_nucleus", "🟦 Nuclear envelope", "E_ne", "#1f77b4",
     r"$K_{ne}\,[\min(x,u)-s]_+^{3}$"),
    ("K_core", "🟪 Inside the nucleus", "Eₙ", "#9467bd",
     r"$K_{core}\,[\min(x,u)-s]_+^{3/2}$"),
)
# The engine names its moduli E_shell, E_cyto, E_core; the page writes them
# with the spring network's symbols and in its units, so a modulus reads the
# same in the results, on the bars, in the routes table and in the sheet.
DISPLAY_SYMBOL = {"E_shell": "Eₘ", "E_cyto": "Ec", "E_ne": "E_ne",
                  "E_nc": "E_nc", "E_core": "Eₙ", "E_align": "E_align",
                  "A_L": "A_L"}
DISPLAY_UNIT = {"E_shell": ("MPa", 1e6), "E_ne": ("MPa", 1e6),
                "E_cyto": ("kPa", 1e3), "E_nc": ("kPa", 1e3),
                "E_core": ("kPa", 1e3), "E_align": ("kPa", 1e3)}


def modulus_display(symbol, value_pa, se_pa=None):
    """A modulus as the results table writes it: 4 figures, its own unit."""
    if value_pa is None or not np.isfinite(float(value_pa)):
        return "—"
    unit, scale = DISPLAY_UNIT.get(symbol, ("kPa", 1e3))
    value_pa = float(value_pa) if abs(float(value_pa)) > 1e-6 else 0.0
    text = f"{value_pa / scale:.4g}"
    if se_pa is not None and np.isfinite(float(se_pa)):
        text += f" ± {float(se_pa) / scale:.3g}"
    return f"{text} {unit}"


PW_COMPONENT_COLORS = {c[0]: c[3] for c in PW_COMPONENTS}
# What a regime is called when it is that component's own stretch.
PW_COMPONENT_TITLES = {c[0]: c[1].split(" ", 1)[1] for c in PW_COMPONENTS}
# Which boundary each component starts at: its regime's start. Moving a
# component's start moves that boundary, and every component sharing it.
# The fallback order, used before the page has one of its own. Which
# boundary a component actually starts at is pw_start_index(), read from
# the order the components are met in.
PW_START_INDEX = {"K_shell": 0, "K_cyto": 1, "K_nucleus": 2, "K_core": 3}
# Any of the four can be switched off; all four are on by default.
PW_SWITCHABLE = tuple(PW_START_INDEX)
# Not a component and not on the board: the constant the first stretch is
# fitted with, [0, ε₁), where the probe is settling onto the cell. It is
# drawn as the bottom layer of the stack so the layers still add up to the
# fitted curve, and it is named as what it is.
PW_BASELINE = ("k_align", "⬛ Baseline C₀ over [0, ε₁)", "C₀", "#9a9a9a",
               r"$C_0$")


def effective_piecewise_settings(settings=None, untils=None, off=()):
    """
    Guesses and bounds, plus each component's own end and on/off state.

    Pure, so a stored cell can be refitted from its record exactly as the
    page fitted it.
    """
    out = {k: dict(v) for k, v in (settings or {}).items() if isinstance(v, dict)}
    for name, until in (untils or {}).items():
        if until is not None:
            out.setdefault(name, {})["until"] = float(until)
    for name in off or ():
        out.setdefault(name, {}).update({"lower": 0.0, "upper": 0.0})
    return out


def piecewise_off():
    """
    The components held at zero: any the board unticks, and the contact
    slope, which is not a component of this model at all.

    k_align is a term of the engine's first regime. The page does not
    offer it, so it is held at zero on every fit, every search and every
    plot, and the first stretch is the constant C₀ and nothing else.
    """
    return ("k_align",) + tuple(
        name for name in PW_SWITCHABLE if not _pw_get(f"pw_use_{name}", True))


def piecewise_until():
    """The component ends set by hand, cleaned."""
    stored = _pw_get("pw_until") or {}
    return {k: float(v) for k, v in stored.items()
            if k in PW_SWITCHABLE and v is not None}


def piecewise_model_settings():
    """What the fit, the search and the plot all use."""
    return effective_piecewise_settings(
        piecewise_settings(), piecewise_until(), piecewise_off())


def piecewise_ranges(bounds=None):
    """{coefficient: (start, until)} in percent, as the fit will use them."""
    return component_ranges(
        bounds or piecewise_boundaries(), regimes=pw_regimes(),
        settings=piecewise_model_settings(), carry=piecewise_carry(),
    )


# Which of the app's C2C12 elements each four-regime coefficient belongs
# to, so the prior written for those elements speaks for these.
PW_ELEMENT_OF = {
    "membrane": ("K_shell",),
    "interior": ("K_cyto",),
    "nucleus_shell": ("K_nucleus",),
    "nucleus": ("K_core",),
}


# ================================================ what the literature says ==
#
# Two things are known about a C2C12 before this curve is fitted, and both
# are about the answer rather than about the procedure:
#
#   * its cytoskeleton is reported at 10 to 15 kPa;
#   * at about 50 % relative deformation the curve bumps, and that bump is
#     the nuclear envelope and the lamina under it being met.
#
# They are used where a prior belongs -- in choosing between fits that the
# data cannot tell apart -- and nowhere else. No coefficient is clamped, no
# modulus is nudged: every number reported is still the plain least-squares
# estimate with its own standard error, and the page says which of them
# landed where the literature says they should.
LITERATURE = {
    "Myoblast (C2C12)": {
        "moduli": {
            # symbol: (low, high, in Pa, what it is)
            "E_cyto": (10.0e3, 15.0e3,
                       "the cytoskeleton of a C2C12, 10–15 kPa"),
        },
        "bump_pct": 50.0,
        "bump_is": "the nuclear envelope and the lamina under it",
        "source": "what is reported for C2C12 myoblasts",
    },
}


def literature_for(cell_type=None):
    """What is known about this cell type before the curve is fitted."""
    if cell_type is None:
        cell_type = st.session_state.get("cell_type")
    return LITERATURE.get(cell_type) or {}


def modulus_in_range(symbol, value_pa, cell_type=None):
    """
    True / False / None: inside the reported range, outside, or not known.

    None is not a failure, it is the honest answer where the literature has
    nothing to say about that component.
    """
    window = (literature_for(cell_type).get("moduli") or {}).get(symbol)
    if not window or value_pa is None or not np.isfinite(float(value_pa)):
        return None
    return bool(window[0] <= float(value_pa) <= window[1])


def literature_fit_score(moduli, eps2_pct, cell_type=None):
    """
    How well one candidate fit agrees with what is known, for ranking.

    Returns (how many moduli land where they should, how far the bump is
    from where it is expected). Bigger first, then smaller: a candidate
    that puts the cytoskeleton at 12 kPa and the bump at 49 % is preferred
    to one that puts them at 5 900 kPa and 62 %, when the curve itself
    cannot tell them apart.
    """
    known = literature_for(cell_type)
    inside = 0
    for symbol, window in (known.get("moduli") or {}).items():
        value = (moduli or {}).get(symbol)
        if value is None or not np.isfinite(float(value)):
            continue
        if window[0] <= float(value) <= window[1]:
            inside += 1
    bump = known.get("bump_pct")
    away = (abs(float(eps2_pct) - float(bump))
            if bump is not None and eps2_pct is not None
            and np.isfinite(float(eps2_pct)) else 0.0)
    return inside, away


def literature_note(fit, cell_type=None):
    """One line: which of the reported numbers this fit agrees with."""
    known = literature_for(cell_type)
    if not known or not (fit and fit.get("success")):
        return ""
    said = []
    for symbol, window in (known.get("moduli") or {}).items():
        field = {"E_cyto": "Ei_kPa", "E_shell": "Em_MPa",
                 "E_ne": "Ene_MPa", "E_core": "En_kPa"}.get(symbol)
        scale = 1e3 if (field or "").endswith("kPa") else 1e6
        try:
            value = float(fit.get(field)) * scale
        except (TypeError, ValueError):
            continue
        unit, divisor = (("kPa", 1e3) if window[1] < 1e6 else ("MPa", 1e6))
        inside = window[0] <= value <= window[1]
        said.append(
            ("✅ " if inside else "⚠️ ")
            + f"{DISPLAY_SYMBOL.get(symbol, symbol)} = {value / divisor:.4g} "
            f"{unit}, " + ("inside" if inside else "outside")
            + f" the {window[0] / divisor:g}–{window[1] / divisor:g} {unit} "
            f"{window[2].split(',')[0]}")
    bump = known.get("bump_pct")
    edges = {name: value for value, name in fit_edges(fit)}
    if bump is not None and "ε₂" in edges:
        at = 100.0 * float(edges["ε₂"])
        close = abs(at - bump) <= 10.0
        said.append(("✅ " if close else "⚠️ ")
                    + f"the bump at ε₂ = {at:.1f} %, "
                    + ("near" if close else "away from")
                    + f" the {bump:g} % where {known.get('bump_is', 'it')} "
                    "is met")
    if not said:
        return ""
    return ("**Against what is reported for this cell type:** "
            + " · ".join(said))


def piecewise_prior(cell_type=None):
    """
    The C2C12 constraints the boundaries are found inside, in percent.

    Read from the app's own C2C12 prior (COMPONENT_PRIORS), the one place
    what is known about the cell is written down:

    * the contact artefact is under 5 % (ε₁ between 1 and 5 %);
    * the sarcolemma and the cytoskeleton carry load from first contact to
      the end;
    * the nucleus is met as a bump at about 50 % that runs on for about
      another 25 %, so ε₂ sits in the prior's onset band (44 to 62 %) and
      ε₃, where dense packing takes over, about 25 % after it (give or take
      10);
    * once met, the nucleus, envelope and contents alike, keeps carrying
      load to the end.
    """
    prior = component_prior(cell_type or "Myoblast (C2C12)")
    lo, hi = deep_onset_band(cell_type or "Myoblast (C2C12)", 0.0, 1.0)
    bump = prior.get("bump") or {"from": 0.50, "span": 0.25}
    span = float(bump.get("span", 0.25)) * 100.0
    start = float(bump.get("from", 0.50)) * 100.0
    to_end = set(prior.get("throughout") or ()) | set(
        prior.get("runs_to_the_end") or ())
    carry = tuple(c for element in ("membrane", "interior", "nucleus_shell",
                                    "nucleus")
                  if element in to_end for c in PW_ELEMENT_OF[element])
    return {
        "eps1_band": (1.0, 5.0),
        "eps2_band": (round(lo * 100.0, 2), round(hi * 100.0, 2)),
        "span_band": (max(5.0, span - 10.0), span + 10.0),
        "defaults": (5.0, start, start + span),
        "carry": carry,
        "why": prior.get("why", ""),
    }


def piecewise_defaults():
    """(ε₁, ε₂, ε₃, end) a curve starts from before its own are found."""
    return tuple(piecewise_prior()["defaults"]) + (DEFAULTS["pw_end"],)


# While the target search is running it tries the first component both
# carried on and holding what it reached. It cannot write the tick's own
# session key -- the widget already exists this run -- so it says so here
# instead, and puts it back when it is done.
_PW_CARRY_OVERRIDE = [None]


# =========================================================== the three ways ==
#
# Three ways of fitting the same four components over the same four
# stretches, one unknown at a time. They differ in one thing only: how far
# each component goes on carrying load after the stretch it was fitted on.
# Write the fit as
#
#     F(x) = sum_j theta_j * phi_j(x; s_j, u_j),
#     phi_j(x) = [ min(x, u_j) - s_j ]_+ ^ p_j
#
# with s_j the boundary the component joins at and u_j where it stops
# taking on more load (past u_j it holds the force it had reached, so the
# curve never steps). s_j is the same in all three. Only u_j changes:
#
#   ① all carried on   u_j = x_end for every j
#   ② one at a time    u_j = s_{j+1}, its own stretch and no further
#   ③ best mixture     u_j in {s_{j+1}, x_end}, chosen for the best R²
#
# so ① and ② are the two ends of ③, and ③ is a search over the 2^4 = 16
# ways of choosing between them.

PW_STYLES = {
    "carried": "Piecewise · every component carried on to the end",
    "handoff": "Piecewise · one at a time, each on its own stretch",
    "best": "Piecewise · the mixture of the two that fits best",
}
PW_STYLE_SHORT = {
    "carried": "① Carried on to the end  ·  each joins and stays",
    "handoff": "② One at a time  ·  each on its own stretch, then held",
    "best": "③ Best mixture  ·  each way tried, the best R² kept",
}
PW_STYLE_MATHS = {
    "carried": r"u_j = x_{end}\;\;\forall j: \quad \hat F(x) = \sum_j "
               r"\hat\theta_j\,\big[x - s_j\big]_+^{\,p_j}",
    "handoff": r"u_j = s_{j+1}: \quad \hat F(x) = \sum_j \hat\theta_j\,"
               r"\big[\min(x,\,s_{j+1}) - s_j\big]_+^{\,p_j}",
    "best": r"\hat u = \arg\max_{u_j \,\in\, \{s_{j+1},\, x_{end}\}} "
            r"R^2\big(\hat F(\,\cdot\,; u)\big) \quad (2^4 = 16 \text{ ways})",
}
PW_STYLE_HELP = {
    "carried":
        "**Each joins and then stays.** The membrane carries load from "
        "first contact to the end of the squash. The cytoskeleton joins at "
        "ε₁ and also carries to the end; the nuclear envelope joins at ε₂ "
        "and carries to the end; the inside of the nucleus joins at ε₃ and "
        "carries to the end. Each one is fitted on the stretch where it "
        "joins, with everything found before it **held at the value "
        "already found** and still adding its force, so every stretch has "
        "one unknown in it.",
    "handoff":
        "**One at a time, handed on.** The membrane is fitted on 0 to ε₁ "
        "and stops taking on more load there: its modulus becomes a fixed "
        "constant, and the force it had reached is carried forward as a "
        "constant while the cytoskeleton is fitted on ε₁ to ε₂. Then that "
        "one is fixed in turn for the nuclear envelope on ε₂ to ε₃, and "
        "that one for the inside of the nucleus from ε₃ on. One element "
        "fitted at a time, each on its own stretch, the boundaries "
        "optimised as always.",
    "best":
        "**The mixture that fits best.** Every component is tried both "
        "ways — carrying on to the end, or stopping at the next boundary "
        "and holding — and the combination with the highest R² is kept. "
        "Sixteen ways in all; the one chosen is written out under the "
        "results, component by component, so it is never a mystery which "
        "mixture the numbers came from.",
}


def piecewise_style():
    """Which of the three ways the page is fitting with."""
    style = _pw_get("pw_style", DEFAULTS.get("pw_style", "carried"))
    return style if style in PW_STYLES else "carried"


def carry_subsets():
    """Every way of choosing, per component, carried on or handed over."""
    names = [ORDER_COEFFICIENT[term] for term in component_order()]
    for mask in range(1 << len(names)):
        yield tuple(name for index, name in enumerate(names)
                    if mask >> index & 1)


def best_mixture(epsilon, force_N, model=None):
    """
    Try every component both ways and keep the combination that fits best.

    Sixteen fits of four coefficients: the same boundaries, the same
    components, the same one-unknown-at-a-time solve, and the only thing
    that changes is which components go on carrying load past their own
    stretch. Returns {"carry", "r2", "rows"}.
    """
    bounds = piecewise_boundaries()
    settings = effective_piecewise_settings(
        piecewise_settings(), piecewise_until(), piecewise_off())
    geometry = piecewise_geometry(model) if model is not None else None
    eps2 = float(bounds[2]) if len(bounds) > 2 else None
    rows = []
    best = None
    for carry in carry_subsets():
        result = fit_piecewise(epsilon, force_N, boundaries_pct=bounds,
                               regimes=pw_regimes(), settings=settings,
                               carry=carry)
        r2 = (float(result.get("r_squared", float("nan")))
              if result.get("success") else float("nan"))
        zeros = sum(1 for name, value in (result.get("coefficients") or {}).items()
                    if name not in ("C0",) and value is not None
                    and np.isfinite(value) and abs(float(value)) < 1e-30)
        inside = 0
        if geometry is not None and result.get("success"):
            try:
                moduli = piecewise_moduli(result, geometry,
                                          regimes=pw_regimes())
                inside, _away = literature_fit_score(
                    {row["symbol"]: row["E_Pa"] for row in moduli.values()},
                    eps2)
            except Exception:  # pragma: no cover - defensive
                inside = 0
        rows.append({"carry": carry, "r2": r2, "unmeasured": zeros,
                     "inside": inside})
        if not np.isfinite(r2):
            continue
        # Ties and near-ties go to the combination that measures every
        # component and agrees with what is reported for this cell type: a
        # mixture that fits a hair closer by silencing one of them, or by
        # putting the cytoskeleton two orders of magnitude out, has not
        # learnt anything about the cell.
        key = (-zeros, inside, round(r2, 9), len(carry))
        if best is None or key > best["key"]:
            best = {"key": key, "carry": carry, "r2": r2, "unmeasured": zeros,
                    "inside": inside}
    if best is None:
        return None
    return {"carry": best["carry"], "r2": best["r2"],
            "unmeasured": best["unmeasured"],
            "inside": best.get("inside", 0), "rows": rows}


def mixture_note(carry):
    """Which components are carried on and which hand over, in words."""
    names = components_for(st.session_state.get("cell_type"))
    carried, handed = [], []
    for term in component_order():
        (carried if ORDER_COEFFICIENT[term] in carry else handed).append(
            names[term][0])
    parts = []
    if carried:
        parts.append("carried on to the end: " + ", ".join(carried))
    if handed:
        parts.append("stops at the next boundary and holds: "
                     + ", ".join(handed))
    return " · ".join(parts)


def piecewise_carry():
    """
    The components that keep acting to the end of the fit.

    The prior's: membrane and cytoskeleton from first contact, the nucleus
    once met. The membrane's tick takes it out of that list.
    """
    # Which components carry on past their own stretch IS the choice
    # between the three ways of fitting, so it is read from that choice
    # and nowhere else.
    order = component_order()
    style = piecewise_style()
    if style == "handoff":
        carry = ()
    elif style == "best":
        kept = tuple(_pw_get("pw_best_carry", None) or ())
        carry = tuple(ORDER_COEFFICIENT[t] for t in order
                      if ORDER_COEFFICIENT[t] in kept)
    else:
        carry = tuple(ORDER_COEFFICIENT[t] for t in order)
    throughout = (_PW_CARRY_OVERRIDE[0] if _PW_CARRY_OVERRIDE[0] is not None
                  else _pw_get("pw_membrane_throughout", True))
    if style == "carried" and not throughout and order:
        # The first component holding what it reached at its own boundary,
        # which is what the target search tries when nothing else reaches
        # it, and what the tick beside that component says.
        first = ORDER_COEFFICIENT[order[0]]
        carry = tuple(c for c in carry if c != first)
    return carry


def piecewise_span():
    """ε₃ − ε₂, low and high, in percent: how long the nuclear bump lasts."""
    return tuple(sorted(float(st.session_state.get(k, DEFAULTS[k]))
                        for k in PW_SPAN_KEYS))


def piecewise_bands():
    """
    The three search bands, (low, high) in percent.

    ε₃'s is where ε₂'s band and the bump's length allow it; the search also
    holds ε₃ − ε₂ inside the bump's length for every placement it tries.
    """
    b1, b2 = (
        tuple(sorted((float(st.session_state.get(lo, DEFAULTS[lo])),
                      float(st.session_state.get(hi, DEFAULTS[hi])))))
        for lo, hi in PW_BAND_KEYS
    )
    span = piecewise_span()
    end = float(st.session_state.get("pw_end", DEFAULTS["pw_end"]))
    b3 = (b2[0] + span[0], max(b2[0] + span[0], min(b2[1] + span[1], end - 2.0)))
    return (b1, b2, b3)


def constraint_notes(bounds):
    """Where the boundaries in use break the C2C12 constraints, if anywhere."""
    (l1, h1), (l2, h2), _b3 = piecewise_bands()
    s_lo, s_hi = piecewise_span()
    e1, e2, e3 = (float(v) for v in bounds[1:4])
    notes = []
    if not l1 - 1e-6 <= e1 <= h1 + 1e-6:
        notes.append(f"ε₁ = {e1:.1f} % is outside the contact zone "
                     f"({l1:g}–{h1:g} %)")
    if not l2 - 1e-6 <= e2 <= h2 + 1e-6:
        notes.append(f"ε₂ = {e2:.1f} % is outside where a C2C12 nucleus is "
                     f"met ({l2:g}–{h2:g} %)")
    if not s_lo - 1e-6 <= e3 - e2 <= s_hi + 1e-6:
        notes.append(f"ε₃ − ε₂ = {e3 - e2:.1f} % is outside the nuclear "
                     f"bump's length ({s_lo:g}–{s_hi:g} %)")
    return notes


def piecewise_signature(epsilon, force_N):
    """What a boundary search depends on: the curve, the model and the end."""
    data = st.session_state.get("data") or {}
    return repr((
        data.get("source"), int(np.size(epsilon)),
        round(float(force_N[-1]), 15) if np.size(force_N) else 0.0,
        sorted((k, sorted(v.items())) for k, v in piecewise_model_settings().items()),
        piecewise_carry(),
        round(float(st.session_state.get("pw_end", DEFAULTS["pw_end"])), 4),
        piecewise_bands(), piecewise_span(),
    ))


def boundary_source(bounds, placements=None):
    """Where the boundaries in use came from, in a few words."""
    inner = tuple(round(float(b), 2) for b in bounds[1:4])
    rows = (placements or {}).get("rows") or {}
    selected = (st.session_state.get("pw_selected") or {}).get("key")
    # The row in use first, so two routes landing on the same numbers are
    # reported as the one that was chosen.
    for key in ([selected] if selected else []) + list(PW_FALLBACK):
        row = rows.get(key)
        if row and inner == tuple(round(float(v), 2) for v in row["best_pct"]):
            return PW_ROW_LABELS[key]
    defaults = tuple(round(float(b), 2) for b in piecewise_defaults()[:3])
    if inner == defaults:
        return "the C2C12 defaults"
    return "set by hand"


def piecewise_settings():
    """The user's changes to initial guesses and bounds, cleaned."""
    stored = _pw_get("pw_settings") or {}
    return {k: dict(v) for k, v in stored.items() if isinstance(v, dict)}


def _pw_number(text):
    """'1e-9', '0', 'inf', '-inf' or blank -> float, or None if unreadable."""
    text = str(text).strip().lower().replace("∞", "inf").replace("+", "")
    if text in ("", "none", "nan"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _pw_text(value):
    """A float as the editor shows it: short, and inf spelled out."""
    value = float(value)
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0:
        return "0"
    return f"{value:.4g}"


def piecewise_geometry(model, probe_um=None, coat_nm=None):
    """What the modulus conversion needs, from the model the page built."""
    if probe_um is None:
        probe_um = st.session_state.get("probe_diameter_um", 0.0)
    if coat_nm is None:
        coat_nm = st.session_state.get("protein_coat_nm", 200.0)
    probe_um = float(probe_um or 0.0)
    return Geometry(
        cell_height=float(model.cell_height),
        cell_radius=float(model.R0),
        nucleus_radius=float(model.R_nucleus),
        probe_radius=probe_um * 0.5e-6 if probe_um > 0 else None,
        membrane_thickness=float(model.h_membrane),
        envelope_thickness=float(getattr(model, "h_envelope", 40e-9)),
        coat_thickness=float(coat_nm) * 1e-9,
        nu_membrane=float(model.nu_m),
        nu_interior=float(model.nu_i),
        nu_nucleus=float(model.nu_n),
    )


def run_piecewise_fit(model, epsilon, force_N):
    """Fit and convert, at the page's boundaries and settings."""
    result = fit_piecewise(
        epsilon, force_N,
        boundaries_pct=piecewise_boundaries(),
        regimes=pw_regimes(),
        settings=piecewise_model_settings(),
        carry=piecewise_carry(),
    )
    if result.get("success"):
        result["moduli"] = piecewise_moduli(result, piecewise_geometry(model),
                                            regimes=pw_regimes())
    return result


def pressure_text(value, se=None):
    """A modulus in the unit that reads best: Pa, kPa or MPa."""
    if value is None or not np.isfinite(value):
        return "—"
    if abs(value) < 1e-6:
        # A coefficient held at its lower bound, not a tiny measurement.
        return "0 Pa"
    for factor, unit in ((1e6, "MPa"), (1e3, "kPa"), (1.0, "Pa")):
        if abs(value) >= factor or unit == "Pa":
            text = f"{value / factor:.3g} {unit}"
            if se is not None and np.isfinite(se):
                text = f"{value / factor:.3g} ± {se / factor:.2g} {unit}"
            return text
    return f"{value:.3g} Pa"


def piecewise_fit_id(result):
    """The fingerprint of a four-regime fit: its ε and its coefficients."""
    if not (result and result.get("success")):
        return ""
    return fit_id({
        "success": True, "coupling": "piecewise",
        "boundaries_pct": [round(float(v), 6) for v in result["boundaries_pct"]],
        "coefficients": {k: float(v) for k, v in
                         sorted((result.get("coefficients") or {}).items())},
        "r_squared": float(result.get("r_squared", float("nan"))),
    })


def piecewise_as_fit(result, model, probe_um=None, settings=None, found=None,
                     off=None, placements=None):
    lamina = lamina_summary(result, piecewise_geometry(model, probe_um=probe_um))
    """
    The four-regime result in the shape the rest of the page reads.

    The database, the download and the results tab were written against the
    spring-network fits. Rather than teach each of them a second shape, the
    four moduli that have a column there are put in it, and everything else
    rides along under ``piecewise``.
    """
    moduli = result.get("moduli") or {}
    b = result["boundaries_pct"]

    def e(name):
        row = moduli.get(name) or {}
        value = float(row.get("E_Pa", float("nan")))
        # A modulus held at its lower bound comes back as 1e-27 Pa, not 0;
        # it is written as the 0 it is, the same everywhere it is shown.
        if np.isfinite(value) and abs(value) < 1e-6:
            value = 0.0
        return value, float(row.get("E_se_Pa", float("nan")))

    shell, shell_se = e("K_shell")
    cyto, cyto_se = e("K_cyto")
    envelope, envelope_se = e("K_nucleus")
    peri, peri_se = e("K_nuc_cyto")
    core, core_se = e("K_core")
    align, align_se = e("k_align")
    tension = float((moduli.get("k_align") or {}).get("tension_N_per_m", float("nan")))

    def finite(v):
        return float(v) if np.isfinite(v) else 0.0

    # Where each modulus was measured: the component's own range, the one
    # on its bar and on the plot.
    ranges = result.get("ranges") or {}

    def span(name, fallback):
        a, u = ranges.get(name, fallback)
        return (float(a) / 100, min(float(u), result["epsilon_range"][1] * 100) / 100)

    spans = {
        "alignment": span("k_align", (b[0], b[1])),
        "membrane": span("K_shell", (b[1], b[2])),
        "interior": span("K_cyto", (b[1], b[2])),
        "nucleus_shell": span("K_nucleus", (b[2], b[3])),
        "perinuclear": span("K_nuc_cyto", (b[2], b[3])),
        "nucleus": span("K_core", (b[3], b[4])),
        "lamina": span("A_lamina", (b[2], b[3])),
    }
    flat = {}
    for name, row in moduli.items():
        flat[f"{name}"] = row["K"]
        flat[f"{name}_se"] = row["K_se"]
        flat[f"{row['symbol']}_Pa"] = row["E_Pa"]
        flat[f"{row['symbol']}_se_Pa"] = row["E_se_Pa"]
    flat["C0_N"] = result["coefficients"].get("C0", float("nan"))
    if lamina:
        flat["A_lamina_N"] = lamina["A_N"]
        flat["A_lamina_se_N"] = lamina["A_se_N"]
        flat["lamina_work_J"] = lamina.get("work_J", float("nan"))
        flat["lamina_peak_pct"] = lamina["peak_pct"]
    flat.update({f"{k}_N": v for k, v in result["anchors"].items()})

    fit = {
        "success": True,
        "fit_id": piecewise_fit_id(result),
        "coupling": "piecewise",
        "model_label": PIECEWISE_MODE,
        # The components this fit is made of: the ticked ones and no
        # others, so a component held at zero is absent from the results,
        # the strip and the copied row rather than reported as 0 ± 0.
        "terms": tuple(
            term for name, term in (("K_shell", "membrane"),
                                    ("K_cyto", "interior"),
                                    ("K_nucleus", "nucleus_shell"),
                                    ("K_core", "nucleus"))
            if name not in (piecewise_off() if off is None else off)),
        "Em": finite(shell), "Em_MPa": finite(shell) / 1e6,
        "Em_MPa_std": finite(shell_se) / 1e6,
        "Ei": finite(cyto), "Ei_kPa": finite(cyto) / 1e3,
        "Ei_kPa_std": finite(cyto_se) / 1e3,
        "Ene": finite(envelope), "Ene_MPa": finite(envelope) / 1e6,
        "Ene_MPa_std": finite(envelope_se) / 1e6,
        "En": finite(core), "En_kPa": finite(core) / 1e3,
        "En_kPa_std": finite(core_se) / 1e3,
        "E_nc_kPa": finite(peri) / 1e3, "E_nc_kPa_std": peri_se / 1e3,
        "E_align_kPa": finite(align) / 1e3,
        "E_align_kPa_std": align_se / 1e3,
        "A_lamina_nN": finite((lamina or {}).get("A_N", float("nan"))) * 1e9,
        "A_lamina_nN_std": float((lamina or {}).get("A_se_N", float("nan"))) * 1e9,
        "load_sharing": PW_SHARE,
        "lamina_work_fJ": finite((lamina or {}).get("work_J", float("nan"))) * 1e15,
        "lamina_peak_pct": finite((lamina or {}).get("peak_pct", float("nan"))),
        "T_align_mN_m": finite(tension) * 1e3,
        "membrane_areal_modulus": finite(shell) * float(model.h_membrane),
        "r_squared": result["r_squared"],
        "adj_r_squared": result.get("adj_r_squared", float("nan")),
        "chi_squared": result.get("chi_squared", float("nan")),
        "chi_squared_reduced": result.get("chi_squared_reduced", float("nan")),
        "noise_sigma": result.get("noise_sigma", float("nan")),
        "rmse": result["rmse"],
        "n_points": result["n_points"],
        "n_params": int(sum(1 for v in (result.get("coefficients") or {}).values()
                            if v is not None)),
        "epsilon_range": tuple(result["epsilon_range"]),
        "weighting": "uniform",
        "warnings": list(result.get("warnings", [])),
        "R0": float(model.R0),
        "R_nucleus": float(model.R_nucleus),
        # ε₁ and ε₂ are the sheet's two boundary columns; ε₃ rides along
        # with the rest under "piecewise".
        "break_1": b[1] / 100.0,
        "break_2": b[2] / 100.0,
        "break_3": b[3] / 100.0,
        "Km_kT": float("nan"),
        "piecewise": {
            "boundaries_pct": list(b),
            "coefficients": dict(result["coefficients"]),
            "anchors_N": dict(result["anchors"]),
            "spans": {k: list(v) for k, v in spans.items()},
            "flat": flat,
            "probe_diameter_um": float(
                st.session_state.get("probe_diameter_um", 0.0)
                if probe_um is None else probe_um
            ),
            "membrane_throughout": "K_shell" in (result.get("carry") or ()),
            "component_ranges_pct": {k: list(v) for k, v in ranges.items()},
            "components_off": list(piecewise_off() if off is None else off),
            "boundary_source": boundary_source(b, placements),
            "boundary_search": (
                {
                    "best_pct": list(found["best_pct"]),
                    "delta_bic": found["delta_bic"],
                    "strength": found["strength"],
                    "intervals": found["intervals"],
                }
                if found and tuple(round(float(v), 2) for v in b[1:4])
                == tuple(round(float(v), 2) for v in found["best_pct"])
                else None
            ),
            "placement": {
                "method": PW_ROW_LABELS.get(
                    st.session_state.get("pw_method", "refined"), ""),
                "reason": (st.session_state.get("pw_selected") or {}).get("reason", ""),
                "target_r2": float(st.session_state.get("pw_target_r2", 0.999)),
                "routes": {
                    PW_ROW_LABELS[k]: {"eps_pct": list(v["best_pct"]),
                                       "r_squared": v["r2"]}
                    for k, v in ((placements or {}).get("rows") or {}).items()
                },
            },
            "settings": piecewise_settings() if settings is None else settings,
        },
    }
    return fit


def piecewise_stage_plan(result):
    """The regimes as the record's stage plan: which elements, over what."""
    names = {
        "R1": ["baseline"], "R2": ["membrane", "interior"],
        "R3": ["nucleus_shell", "perinuclear_cytoskeleton"], "R4": ["nucleus"],
    }
    return [
        {"terms": names.get(r["key"], [r["key"]]),
         "range": (r["domain_pct"][0] / 100.0, r["domain_pct"][1] / 100.0)}
        for r in result.get("regimes") or []
    ]


# Label tags are placed in pixels, so how wide a tag is has to be guessed
# before plotly draws it. These are generous for a 13 px font.
TAG_CHAR_PX = 8.0
TAG_PAD_PX = 14.0
TAG_ROW_PX = 24
ASSUMED_PLOT_PX = 560.0


def tag_levels(positions, texts, x_span, plot_px=ASSUMED_PLOT_PX, x_max=None):
    """
    Where each tag goes so that no two overlap and none runs off the plot.

    A tag normally starts just right of its line. One that would run past
    the right edge of the plot (``x_max``) is put on the line's left
    instead. Tags are then laid out left to right, each on the lowest row
    whose last tag ends before this one starts, widths estimated from the
    text. Returns ([(row, "left" | "right"), ...], rows needed).
    """
    px_per_unit = plot_px / max(float(x_span), 1e-9)
    x_max = float(x_max) if x_max is not None else float("inf")
    spans = []
    for x, text in zip(positions, texts):
        width = (len(text) * TAG_CHAR_PX + TAG_PAD_PX) / px_per_unit
        side = "right" if x + width > x_max else "left"
        spans.append((x - width, x, side) if side == "right" else (x, x + width, side))
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    ends, placed = [], [None] * len(spans)
    for i in order:
        left, right, side = spans[i]
        for row, last in enumerate(ends):
            if left >= last:
                ends[row] = right
                placed[i] = (row, side)
                break
        else:
            ends.append(right)
            placed[i] = (len(ends) - 1, side)
    return placed, max(len(ends), 1)


# How each kind of tagged line looks: (line colour, width, dash, tag
# border, tag font size, tag text colour). A "region" tag names a shaded
# stretch and has no line of its own.
TAG_STYLES = {
    "boundary": ("#222222", 1.5, "dash", "#222222", 13, "#111111"),
    "end": ("#888888", 1.0, "dot", "#aaaaaa", 11, "#555555"),
    "artefact": ("#8c6d31", 1.5, "solid", "#8c6d31", 12, "#6b4f1d"),
    "rupture": ("#ff7f0e", 2.0, "dashdot", "#ff7f0e", 12, "#b35900"),
    "region": (None, 0, None, "#8c6d31", 12, "#6b4f1d"),
}
ARTEFACT_FILL = "rgba(140,109,49,0.13)"
ARTEFACT_PCT = 5.0


def add_tag_lines(fig, items, x_span, x_max, plot_px=None):
    """
    Vertical lines with their tags above the plot, none overlapping.

    ``items`` are dicts with ``x`` (figure units), ``text``, ``kind`` (a
    TAG_STYLES key) and optionally ``bold`` (the part of the text to set in
    bold). Each line runs the height of the plot and on up into the top
    margin to its own tag, so no tag sits on the data, a curve, the legend
    or another line; tags that would touch go on separate rows, and one
    that would run off the right edge turns to the left of its line.
    Returns the number of tag rows, for the top margin.
    """
    items = [dict(item) for item in items
             if item.get("x") is not None and np.isfinite(float(item["x"]))]
    if not items:
        return 0
    placed, n_rows = tag_levels([float(i["x"]) for i in items],
                                [i["text"] for i in items], x_span,
                                plot_px=plot_px or ASSUMED_PLOT_PX,
                                x_max=x_max)
    for item, (row, side) in zip(items, placed):
        colour, width, dash, border, size, ink = TAG_STYLES.get(
            item.get("kind", "boundary"), TAG_STYLES["boundary"])
        x = float(item["x"])
        if colour is not None:
            fig.add_shape(
                type="line", xref="x", yref="paper", x0=x, x1=x, y0=0, y1=1,
                line={"color": colour, "width": width, "dash": dash},
                layer="below",
            )
            # The line carries on above the plot, in pixels, up to its own
            # tag, so a tag on a higher row is still plainly its line's.
            fig.add_shape(
                type="line", xref="x", yref="paper", ysizemode="pixel",
                yanchor=1.0, x0=x, x1=x, y0=0, y1=4 + TAG_ROW_PX * row + 10,
                line={"color": colour, "width": 1},
            )
        text = item["text"]
        bold = item.get("bold")
        if bold and bold in text:
            text = text.replace(bold, f"<b>{bold}</b>", 1)
        fig.add_annotation(
            x=x, y=1.0, xref="x", yref="paper", showarrow=False, text=text,
            xanchor=side, yanchor="bottom", xshift=3 if side == "left" else -3,
            yshift=4 + TAG_ROW_PX * row,
            font={"size": size, "color": ink},
            bgcolor="rgba(255,255,255,0.95)", bordercolor=border,
            borderwidth=1, borderpad=2,
        )
    return n_rows


def boundary_tag_items(bounds, scale=1.0, end_label=True):
    """ε₁, ε₂, ε₃ (and the end of the fit) as tag items, in figure units."""
    items = [{"x": float(v) * scale, "text": f"{name} = {float(v):.1f} %",
              "bold": name, "kind": "boundary"}
             for name, v in zip(EPS_NAMES, bounds[1:4])]
    if end_label:
        items.append({"x": float(bounds[-1]) * scale,
                      "text": f"end = {float(bounds[-1]):.1f} %", "kind": "end"})
    return items


def add_boundary_lines(fig, bounds, scale=1.0, end_label=True, x_span=None,
                       x_max=None, extra=()):
    """
    ε₁, ε₂, ε₃ (and the end of the fit) as lines with tags above the plot,
    plus any ``extra`` tag items (the contact artefact, a rupture). Returns
    the number of tag rows, so the caller can make the top margin fit.

    ``scale`` turns percent into the figure's x unit: 1 for a percent axis,
    0.01 for one in ε as a fraction. ``x_span`` and ``x_max`` are the x
    axis's width and right edge, in percent.
    """
    span = (float(x_span) if x_span
            else max(float(bounds[-1]) - float(bounds[0]), 1.0) * 1.05)
    return add_tag_lines(
        fig, boundary_tag_items(bounds, scale, end_label) + list(extra),
        span * scale, (x_max if x_max is not None else span) * scale,
    )


def legend_rows(labels, plot_px=None):
    """How many rows a horizontal legend of these labels needs."""
    width = float(plot_px or ASSUMED_PLOT_PX)
    rows, used = 1, 0.0
    for label in labels:
        need = 7.0 * len(str(label)) + 44.0
        if used and used + need > width:
            rows, used = rows + 1, 0.0
        used += need
    return rows


PW_VIEWS = {
    "stacked": "Σⱼ Fⱼ(x) = F̂(x)   · stacked layers",
    "own": "Fⱼ(x) on [sⱼ, uⱼ]   · each from zero",
}


def _rgba(hex_colour, alpha):
    """#rrggbb as an rgba() string."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def piecewise_figure(epsilon, force_N, result, style, log_y=False, off=(),
                     view="stacked", note=None):
    """
    The data, the fitted curve, and every component over its own range.

    The top panel is the curve: the data, the total fit in black and the
    components. Stacked, each component is a layer that starts at its own
    boundary, thickens while it takes load, keeps its thickness once it
    holds, and the layers add up to the black curve exactly; each from
    zero, it is a dashed line from where it starts. The bottom panel is the
    same ranges as bars. Everything is drawn from the fit's own boundaries
    and ranges, the ones the controls show and the results report. The
    first stretch, to ε₁, is shaded as the contact artefact: the probe
    settling onto the cell, fitted by the contact line and nothing else.
    """
    x = np.asarray(epsilon, dtype=float) * 100.0
    y, unit = from_newtons(np.asarray(force_N, dtype=float), style.force_unit)
    scale = float(from_newtons(np.array([1.0]), style.force_unit)[0][0])
    fig = go.Figure()
    b = result["boundaries_pct"]
    end = float(b[-1])
    ranges = result.get("ranges") or {}
    stacked = view == "stacked" and not log_y and component_force is not None
    for regime in result["regimes"]:
        a, z = regime["domain_pct"]
        # The regime's name inside its band, at the top left and nudged off
        # the boundary line; the curve rises to the right, so that corner is
        # the one the data leave empty.
        fig.add_vrect(
            x0=a, x1=z,
            fillcolor=(ARTEFACT_FILL if regime["key"] == "R1"
                       else PW_BANDS.get(regime["key"], "rgba(0,0,0,0.05)")),
            line_width=0, layer="below",
            annotation_text=regime["key"], annotation_position="top left",
            annotation_font_size=12, annotation_font_color="#666666",
            annotation_xshift=4, annotation_yshift=-4,
        )
    legend_names = []
    if stacked:
        # One grid for every layer, with every boundary and every [s, u] on
        # it, so each layer's corner is drawn exactly where it is.
        grid = np.linspace(float(b[0]), end, 700)
        marks = [v for pair in ranges.values() for v in pair] + list(b)
        grid = np.unique(np.concatenate(
            [grid, [v for v in marks if b[0] <= v <= end]]))
        for name, label, symbol, colour, _law in (PW_BASELINE,) + PW_COMPONENTS:
            # The baseline layer carries C₀, the force the curve starts
            # from, so it stays or the layers would not add up to F̂.
            if name in off and name != PW_BASELINE[0]:
                continue
            layer = component_force(result, name, grid)
            if layer is None:
                continue
            a, u = ranges.get(name, (grid[0], grid[-1]))
            legend_names.append(
                PW_BASELINE[1] if name == PW_BASELINE[0]
                else f"{label} · [{a:.1f}, {u:.1f}] %")
            fig.add_trace(go.Scatter(
                x=grid, y=layer * scale, mode="lines", stackgroup="components",
                name=legend_names[-1],
                line={"color": colour, "width": 0.8},
                fillcolor=_rgba(colour, 0.30),
                hovertemplate=f"{symbol}<br>x = %{{x:.1f}} %<br>F_j = %{{y:.4g}} "
                              f"{unit}<extra></extra>",
            ))
    keep = (y > 0) if log_y else np.ones_like(y, dtype=bool)
    fig.add_trace(go.Scatter(
        x=x[keep], y=y[keep], mode="markers", name="Experimental data",
        marker={"color": style.data_color, "size": max(3, int(style.marker_size * 0.55))},
    ))
    # The whole fitted curve, every regime, one line.
    pieces = regime_curves(result)
    if pieces:
        xs = np.concatenate([p[1] for p in pieces])
        fs = np.concatenate([p[2] for p in pieces])
        fig.add_trace(go.Scatter(
            x=xs, y=fs * scale, mode="lines",
            name="Fitted curve F̂" + (" = Σ layers" if stacked else ""),
            line={"color": style.fit_color or "#000000",
                  "width": max(2, int(style.line_width * 0.8))},
        ))
    if not stacked:
        # Each component's own force, over its own range.
        for name, label, symbol, colour, _law in PW_COMPONENTS:
            if name in off:
                continue
            curve = component_curve(result, name)
            if curve is None:
                continue
            cx, cf = curve
            a, u = ranges.get(name, (cx[0], cx[-1]))
            legend_names.append(f"{label} · [{a:.1f}, {u:.1f}] %")
            fig.add_trace(go.Scatter(
                x=cx, y=cf * scale, mode="lines", name=legend_names[-1],
                line={"color": colour, "width": 2, "dash": "dash"},
                hovertemplate=f"{symbol}<br>x = %{{x:.1f}} %<br>F = %{{y:.4g}} {unit}"
                              "<extra></extra>",
            ))
    anchor_x, anchor_y, anchor_text = [], [], []
    for name, value in result["anchors"].items():
        where = float(name.split("_")[1].replace("pct", ""))
        anchor_x.append(where)
        anchor_y.append(value * scale)
        anchor_text.append(f"{name.replace('pct', '%')} = {value * scale:.4g} {unit}")
    if anchor_x:
        fig.add_trace(go.Scatter(
            x=anchor_x, y=anchor_y, mode="markers", name="C⁰ anchors",
            marker={"symbol": "diamond", "size": 11, "color": "#ffffff",
                    "line": {"color": "#000000", "width": 2}},
            text=anchor_text, hovertemplate="%{text}<extra></extra>",
        ))

    # The range track: the component rows, as bars on a panel of their own.
    # One row per component that is on: the track and the ticked rows on
    # the board are the same list, in the same order.
    ticks, labels = [], []
    drawn_rows = [c for c in PW_COMPONENTS if c[0] not in off]
    for row, (name, label, symbol, colour, _law) in enumerate(drawn_rows):
        a, u = ranges.get(name, (np.nan, np.nan))
        ticks.append(row)
        labels.append(symbol)
        if not np.isfinite(a):
            continue
        fig.add_trace(go.Scatter(
            x=[a, u], y=[row, row], mode="lines", yaxis="y2",
            line={"color": colour, "width": 10}, showlegend=False,
            hovertemplate=f"{label}: {a:.1f}–{u:.1f} %<extra></extra>",
        ))
        if u < end - 1e-6 and name not in ("k_align", "A_lamina"):
            fig.add_trace(go.Scatter(
                x=[u, end], y=[row, row], mode="lines", yaxis="y2",
                line={"color": colour, "width": 2, "dash": "dot"},
                showlegend=False,
                hovertemplate=f"{symbol} holds what it reached<extra></extra>",
            ))
    x_lo = min(0.0, float(np.nanmin(x)) if x.size else 0.0) - 1.0
    x_hi = max(end, float(np.nanmax(x)) if x.size else end) + 1.5
    # The contact artefact is named once, above the plot, over its band.
    artefact = [{"x": float(b[0]), "text": "contact artefact [0, ε₁)",
                 "kind": "region"}]
    tag_rows = add_boundary_lines(fig, b, x_span=x_hi - x_lo, x_max=x_hi,
                                  extra=artefact)
    if note:
        # Which fit this is, top left, under the regime names: the corner
        # the rising curve and its layers leave empty.
        fig.add_annotation(
            text=str(note), xref="paper", yref="paper", x=0.01, y=1.0,
            xanchor="left", yanchor="top", yshift=-24, showarrow=False,
            font={"size": 11, "color": "#444444"},
            bgcolor="rgba(255,255,255,0.88)", bordercolor="#999999",
            borderwidth=1, borderpad=3,
        )
    n_rows = legend_rows(legend_names + ["Experimental data", "Fitted curve F̂",
                                         "C⁰ anchors"])
    fig.update_layout(
        height=int(style.height * 1.3) + TAG_ROW_PX * tag_rows + 22 * n_rows,
        template="simple_white",
        # Top: room for the tags, one row each. Bottom: the x axis, then
        # the legend under it, so nothing is drawn over the data.
        margin={"l": 80, "r": 20, "t": 16 + TAG_ROW_PX * tag_rows,
                "b": 78 + 24 * n_rows},
        legend={"orientation": "h", "yref": "container", "yanchor": "bottom",
                "y": 0.005, "xanchor": "left", "x": 0.0, "font": {"size": 12},
                "traceorder": "normal"},
        xaxis={"title": "Relative deformation x (%)", "anchor": "y2",
               "range": [x_lo, x_hi]},
        yaxis={"title": f"Force ({unit})", "domain": [0.30, 1.0]},
        yaxis2={"domain": [0.0, 0.24], "anchor": "x", "tickvals": ticks,
                "ticktext": labels, "range": [len(ticks) - 0.5, -0.5],
                "showgrid": False, "zeroline": False,
                "title": {"text": "[s, u]", "font": {"size": 12}}},
    )
    if log_y:
        # Only the force axis; the range track stays linear.
        fig.update_layout(yaxis={"type": "log"})
    return fig


def piecewise_graph_note(result, off=(), view="stacked", log_y=False,
                         fit_id_text="", n_points=None):
    """
    What is on the graph above, in one line, written from the same fit.

    The plot is read from a distance and the legend only names things; this
    says what is actually drawn right now, which components with which
    ranges, which are left out, where the boundaries are and what the
    shaded first stretch means. It is written from ``result``, the applied
    fit, so it cannot describe a plot that is not there.
    """
    b = result.get("boundaries_pct") or ()
    ranges = result.get("ranges") or {}
    drawn, left_out = [], []
    for name, label, _symbol, _colour, _law in PW_COMPONENTS:
        if name in off:
            left_out.append(label)
            continue
        a, u = ranges.get(name, (float("nan"), float("nan")))
        drawn.append(f"{label} [{a:.1f}, {u:.1f}] %"
                     if np.isfinite(a) else label)
    how = ("stacked, so the layers add up to F̂" if view == "stacked" and not log_y
           else "each drawn from zero over its own range")
    parts = [
        "**On the graph now:** the data"
        + (f" ({int(n_points)} points)" if n_points else "")
        + " and the fitted curve F̂"
        + (f" of `fit {fit_id_text}`" if fit_id_text else "")
        + (f", R² = {result['r_squared']:.5f}"
           if np.isfinite(result.get("r_squared", float("nan"))) else ""),
        ("components " + how + ": " + "; ".join(drawn)) if drawn
        else "no components (all of them are switched off)",
    ]
    if view == "stacked" and not log_y:
        parts.append("under them the grey baseline layer C₀, the force the "
                     "curve starts from, which is not a component")
    if left_out:
        parts.append("switched off, so neither fitted nor drawn: "
                     + ", ".join(left_out))
    if len(b) >= 5:
        parts.append(
            f"boundaries ε₁ = {b[1]:.2f} %, ε₂ = {b[2]:.2f} %, "
            f"ε₃ = {b[3]:.2f} %, fitted to x_end = {b[4]:.1f} %"
        )
        parts.append(
            f"the shaded stretch [0, {b[1]:.2f}) % is the contact artefact, "
            "the probe settling onto the cell, carried by the baseline C₀ "
            "and no component"
        )
    parts.append("force axis logarithmic" if log_y else "force axis linear")
    parts.append("the bars under the curve are the same ranges, one row per "
                 "component")
    st.caption(" · ".join(parts) + ".")


def profile_figure(found, index):
    """One boundary's profile likelihood: Δ(−2 ln L) across its band."""
    key = ("eps1", "eps2", "eps3")[index]
    grid, delta = found["profiles"][key]
    grid = np.asarray(grid, dtype=float)
    delta = np.array([np.nan if d is None else d for d in delta], dtype=float)
    shown = np.clip(delta, 0.0, 30.0)
    best = found["intervals"][key]["best"]
    fig = go.Figure(go.Scatter(x=grid, y=shown, mode="lines",
                               line={"color": "#1f77b4", "width": 2},
                               name="Δ(−2 ln L)"))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#2ca02c",
                  annotation_text="68 %", annotation_position="top left")
    fig.add_hline(y=3.84, line_dash="dot", line_color="#ff7f0e",
                  annotation_text="95 %", annotation_position="top left")
    fig.add_vline(x=best, line_color="#000000", line_width=1.5)
    spec = found["spec_pct"][index]
    fig.add_vline(x=spec, line_color="#888888", line_dash="dash", line_width=1,
                  annotation_text="spec", annotation_position="bottom right")
    fig.update_layout(
        height=230, template="simple_white", showlegend=False,
        margin={"l": 50, "r": 10, "t": 30, "b": 45},
        title={"text": f"{EPS_NAMES[index]} = {best:.2f} %", "font": {"size": 14}},
        xaxis_title=f"{EPS_NAMES[index]} (%)", yaxis_title="Δ(−2 ln L)",
    )
    fig.update_yaxes(range=[0, 30])
    return fig


def piecewise_residual_figure(epsilon, force_N, fitted, style):
    """Data minus fit, in the display unit."""
    residual, unit = from_newtons(
        np.asarray(force_N, dtype=float) - np.asarray(fitted, dtype=float),
        style.force_unit,
    )
    fig = go.Figure(go.Scatter(
        x=np.asarray(epsilon, dtype=float) * 100.0, y=residual, mode="markers",
        marker={"size": 4, "color": "#555555"}, name="residual",
    ))
    fig.add_hline(y=0, line_color="#000000", line_width=1)
    fig.update_layout(height=260, template="simple_white",
                      margin={"l": 70, "r": 20, "t": 10, "b": 50},
                      xaxis_title="Relative deformation (%)",
                      yaxis_title=f"Residual ({unit})", showlegend=False)
    return fig


def piecewise_parameter_editor():
    """Initial guesses and bounds, one row per coefficient, editable."""
    stored = piecewise_settings()
    rows = []
    for regime in PW_REGIMES:
        for term in regime.terms:
            if term.name == "k_align":
                continue
            mine = stored.get(term.name, {})
            rows.append({
                "Regime": regime.key,
                "Coefficient": term.name,
                "Law": ("sin²(π·dx/w)" if term.shape == "lump"
                        else f"dx^{term.power:g}"),
                "Initial guess": _pw_text(mine.get("p0", term.p0)),
                "Lower bound": _pw_text(mine.get("lower", term.lower)),
                "Upper bound": _pw_text(mine.get("upper", term.upper)),
            })
    frame = pd.DataFrame(rows)
    edited = st.data_editor(
        frame, key="pw_param_editor", hide_index=True, num_rows="fixed",
        disabled=["Regime", "Coefficient", "Law"],
        **STRETCH,
    )
    wanted, problems = {}, []
    defaults = {t.name: t for r in PW_REGIMES for t in r.terms}
    for _, row in edited.iterrows():
        name = row["Coefficient"]
        term = defaults[name]
        values = {}
        for column, key in (("Initial guess", "p0"), ("Lower bound", "lower"),
                            ("Upper bound", "upper")):
            number = _pw_number(row[column])
            if number is None:
                problems.append(f"{name}: '{row[column]}' is not a number")
                number = getattr(term, key)
            values[key] = number
        if values["lower"] > values["upper"]:
            problems.append(f"{name}: the lower bound must not be above the upper")
            values["lower"], values["upper"] = term.lower, term.upper
        changed = {k: v for k, v in values.items() if v != getattr(term, k)}
        if changed:
            wanted[name] = changed
    if wanted != stored:
        # The fit above was made with the old values; run the page again
        # with these so what is drawn is what was typed.
        rerun_keeping_settings({"pw_settings": wanted})
    for problem in problems:
        st.caption(f"⚠️ {problem}; using the default instead.")
    st.caption(
        "Every regime is linear in its coefficients once its anchor is "
        "fixed, so the fit has one best answer inside the bounds. The "
        "initial guess is where Trust-Region Reflective starts; it can slow "
        "the solver down but cannot change the answer. Bounds do change it: "
        "a coefficient sitting on its bound is flagged in the results."
    )


def keep_unrendered_settings():
    """
    Hold on to the settings of whichever sharing of the load is not shown.

    Streamlit drops a widget's value at the end of any run that does not
    draw it, and with the four-regime fit on screen none of the other
    models' controls are drawn. Writing each value back to itself keeps it,
    so switching to the advanced models finds them as they were left.
    Called at the very end of the run, when every widget that is going to
    be drawn has been: those refuse the write and are skipped (they are
    kept by being drawn), and nothing drawn afterwards can complain that
    its value was set through session state.
    """
    for key in DEFAULTS:
        if key in NOT_A_SETTING or key.startswith("_"):
            continue
        if key not in st.session_state:
            continue
        try:
            st.session_state[key] = st.session_state[key]
        except Exception:
            pass


PW_TEX = {"K_shell": r"K_{shell}", "K_cyto": r"K_{cyto}",
          "K_nucleus": r"K_{ne}", "K_nuc_cyto": r"K_{nc}",
          "K_core": r"K_{core}", "A_lamina": r"A_L"}


def piecewise_equations_latex(bounds, ranges, off=()):
    """
    The fitted model with every component's range written into it.

    Written as the sum it is: past the contact zone each component adds
    K (x - start)^p from its own start to its own end and holds after, so
    the equation carries exactly the numbers on the range bars.
    """
    e1, end = float(bounds[1]), float(bounds[4])
    terms = []
    for name, _label, _symbol, _colour, _law in PW_COMPONENTS:
        if name == "k_align" or name in off or name not in ranges:
            continue
        a, u = ranges[name]
        if name == "A_lamina":
            terms.append(
                rf"A_L\,\sin^2\!\Big(\pi\frac{{x-{a:.4g}}}{{{u - a:.4g}}}\Big)"
                rf"\,\mathbb{{1}}_{{[{a:.4g},\,{u:.4g}]}}(x)"
            )
            continue
        power = {"K_shell": "3", "K_nucleus": "3"}.get(name, "3/2")
        terms.append(
            rf"{PW_TEX[name]}\,\big[\min(x,{u:.4g})-{a:.4g}\big]_+^{{{power}}}"
        )
    body = r" \\ &+ ".join(terms) if terms else "0"
    return [
        rf"F(x) = k_{{align}}\,x + C_0, \qquad 0 \le x < {e1:.4g}",
        r"\begin{aligned} F(x) = F_{" + f"{e1:.4g}" + r"\%} &+ " + body
        + r" \end{aligned}" + rf"\qquad {e1:.4g} \le x \le {end:.4g}",
    ]


def _tex_number(value, digits=4):
    """A number in LaTeX, in scientific form when it needs to be."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return r"\text{—}"
    if not np.isfinite(value):
        return r"\text{—}"
    if value == 0:
        return "0"
    text = f"{value:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return text


def piecewise_answer(result, fit):
    """
    The answer, as mathematics: the fitted function, stretch by stretch.

    The strip at the top says what each component came to. This says what
    the whole curve came to: F̂ written out with the numbers in it, one
    line per stretch, each starting from the force the one before it ended
    on, and the moduli those coefficients turn into. It is the thing to
    copy into a report.
    """
    if not (result and result.get("success")):
        return
    names = components_for(st.session_state.get("cell_type"))
    moduli = result.get("moduli") or {}
    ranges = result.get("ranges") or {}
    bounds = result["boundaries_pct"]
    off = set(result.get("components_off") or piecewise_off())

    st.markdown("**The answer, written out**")
    st.caption("x is the relative deformation in per cent, F̂ in newtons; "
               "⟨·⟩ is zero before its boundary.")
    for index, regime in enumerate(result.get("regimes") or []):
        a, b = regime["domain_pct"]
        fitted_here = [name for name in regime["params"]
                       if name not in ("C0", "k_align") and name not in off]
        pieces = []
        if index == 0:
            offset = (regime["params"].get("C0") or {}).get("value")
            if offset is not None and np.isfinite(offset) and abs(offset) > 0:
                pieces.append(_tex_number(offset, 3))
        else:
            pieces.append(rf"\hat F({a:.4g}\%)")
        for name in fitted_here:
            row = regime["params"][name]
            power = row.get("power", 1.0)
            power_tex = {3.0: "3", 1.5: "3/2"}.get(float(power),
                                                   f"{float(power):g}")
            pieces.append(
                rf"{_tex_number(row.get('value'), 4)}\,"
                rf"\big\langle x - {a:.4g} \big\rangle^{{{power_tex}}}"
            )
        # What the stretches before it are still adding here.
        carried = [name for name in (regime.get("carried") or {})
                   if name not in off]
        if carried:
            pieces.append(
                r"\underbrace{\textstyle\sum_{j<k} \hat\theta_j\,\phi_j(x)}"
                r"_{\text{" + f"{len(carried)} carried on" + r"}}")
        body = " + ".join(pieces) if pieces else "0"
        last_one = index >= len(result["regimes"]) - 1
        relation = r"\le" if last_one else "<"
        st.latex(rf"\hat F(x) = {body}, \qquad {a:.4g}\,\% \le x "
                 + relation + rf" {b:.4g}\,\%")

    lines = []
    for term, coefficient in ORDER_COEFFICIENT.items():
        row = moduli.get(coefficient)
        if not row or coefficient in off:
            continue
        unit, scale = DISPLAY_UNIT.get(row["symbol"], ("kPa", 1e3))
        value = float(row["E_Pa"]) / scale
        error = float(row.get("E_se_Pa", float("nan"))) / scale
        start, until = ranges.get(coefficient, (float("nan"), float("nan")))
        lines.append(
            rf"{DISPLAY_SYMBOL.get(row['symbol'], row['symbol'])} = "
            rf"{_tex_number(value, 4)}"
            + (rf" \pm {_tex_number(error, 3)}" if np.isfinite(error) else "")
            + rf"\;\text{{{unit}}} \quad \text{{({names[term][0].split(' ', 1)[-1]}, "
            rf"{start:.1f}–{until:.1f}\%)}}")
    if lines:
        st.latex(r"\begin{aligned}" + r" \\ ".join(
            line.replace(" = ", " &= ", 1) for line in lines) + r"\end{aligned}")
    r2 = float(result.get("r_squared", float("nan")))
    chi = float(result.get("chi_squared_reduced", float("nan")))
    st.latex(
        rf"R^2 = {r2:.5f}"
        + (rf", \qquad \chi^2_\nu = {chi:.3g}" if np.isfinite(chi) else "")
        + rf", \qquad n = {int(result.get('n_points', 0))}"
        + rf", \qquad (\varepsilon_1,\varepsilon_2,\varepsilon_3) = "
        rf"({bounds[1]:.2f},\,{bounds[2]:.2f},\,{bounds[3]:.2f})\,\%"
    )


def _band_keys():
    """Every session key that holds a constraint on ε."""
    return [key for pair in PW_BAND_KEYS for key in pair] + list(PW_SPAN_KEYS)


def _widen_the_bands(factor):
    """
    Open the constraints on ε out about their own middle, by ``factor``.

    Each band keeps its centre and grows: the prior is still the place the
    search starts from, it is just allowed to look further out. Returns the
    values written, so they can be reported and put back.
    """
    (l1, h1), (l2, h2), _b3 = piecewise_bands()
    s_lo, s_hi = piecewise_span()
    out = {}
    for (lo_key, hi_key), (lo, hi) in zip(PW_BAND_KEYS, ((l1, h1), (l2, h2))):
        middle, half = (lo + hi) / 2.0, (hi - lo) / 2.0 * factor
        out[lo_key] = round(float(max(0.5, middle - half)), 2)
        out[hi_key] = round(float(min(99.0, middle + half)), 2)
    middle, half = (s_lo + s_hi) / 2.0, (s_hi - s_lo) / 2.0 * factor
    out[PW_SPAN_KEYS[0]] = round(float(max(1.0, middle - half)), 2)
    out[PW_SPAN_KEYS[1]] = round(float(min(90.0, middle + half)), 2)
    return out


# How far the search is allowed to open the constraints out, in order, and
# what each step is called when the page says what it had to do.
# How long a curve may spend placing its own boundaries as it loads. The
# button has no budget: a person who pressed it is waiting on purpose.
ARRIVAL_BUDGET_S = 12.0
ARRIVAL_SEARCH_POINTS = 700
REACH_STEPS = (
    (1.0, "inside the C2C12 prior"),
    (2.0, "with the prior's bands opened out twice as wide"),
    (4.0, "with the prior's bands opened out four times"),
)


def reach_the_target(model, epsilon, force_N, target, budget_seconds=None,
                     interleave=False, search_points=None):
    """
    Open the constraints on ε out, a step at a time, until R² ≥ target.

    The prior is where the search starts, not where it has to end: a cell
    whose nucleus is met at 70 % is a real cell, and a fit that cannot
    reach it inside a band written for the average one is a fit refusing to
    describe the cell in front of it. So each step widens the bands about
    their own middle, searches again, and stops at the first placement that
    reaches the target. What it had to open out is reported, because a fit
    found outside the prior is a result about this cell that the prior did
    not expect.

    ``budget_seconds`` stops it early and keeps the best it has: a curve
    loading should not wait a minute for a search that is not going to
    reach the target anyway, and the button has no such hurry.

    Returns {"reached", "best_pct", "r2", "said", "bands", "step"} or None.
    """
    started = time.monotonic()
    kept = {key: st.session_state.get(key) for key in _band_keys()}
    _SEARCH_POINTS_OVERRIDE[0] = search_points
    was_throughout = bool(st.session_state.get("pw_membrane_throughout", True))
    # Which order to try things in. Left alone, every placement is tried
    # with the first component acting throughout before the model is
    # changed at all. Interleaved, both models are tried at each step: it
    # gets to a usable answer in two searches rather than four, which is
    # what a curve that is still loading needs.
    plan = ([(factor, said, throughout)
             for factor, said in REACH_STEPS
             for throughout in (was_throughout, False)] if interleave else
            [(factor, said, throughout)
             for throughout in (was_throughout, False)
             for factor, said in REACH_STEPS])
    best = None
    try:
        # Two passes, not two tries at each step. The first component acting
        # throughout is the model this page is set up with, so every
        # placement is tried that way first, all the way out; only when none
        # of them reaches the target is the first component asked to hold
        # what it reached instead. A law carried on and extrapolated can
        # cover the whole of the next stretch and leave nothing for the
        # component that joins there, so holding it is often the difference
        # between three components measured and four -- but it is a change
        # to the model, and a change to the model is the last resort.
        for factor, said, throughout in plan:
            _PW_CARRY_OVERRIDE[0] = throughout
            held = ("" if throughout else
                    ", with the first component holding what it reached "
                    "rather than carrying on")
            st.session_state.update(kept)
            if factor != 1.0:
                st.session_state.update(_widen_the_bands(factor))
            st.session_state["pw_placements"] = None
            placements = compute_placements(model, epsilon, force_N)
            rows = placements.get("rows") or {}
            for key, row in rows.items():
                r2 = float(row.get("r2", float("nan")))
                if not np.isfinite(r2):
                    continue
                if best is None or r2 > best["r2"]:
                    best = {"reached": r2 >= target, "r2": r2,
                            "said": said + held,
                            "best_pct": tuple(row["best_pct"]),
                            "route": key, "throughout": throughout,
                            "widened": (factor != 1.0
                                        or throughout != was_throughout),
                            "bands": {k: st.session_state.get(k)
                                      for k in _band_keys()},
                            "placements": placements}
            if best and best["reached"]:
                return best
            if (budget_seconds is not None
                    and time.monotonic() - started > budget_seconds):
                if best:
                    best["stopped_early"] = True
                return best
    finally:
        _PW_CARRY_OVERRIDE[0] = None
        _SEARCH_POINTS_OVERRIDE[0] = None
        # The board goes back to the constraints it had; the boundaries the
        # search found are written to it instead, as typed numbers.
        st.session_state.update(kept)
    return best


def reach_note(found, target):
    """What the search had to do, in one line."""
    if not found:
        return ("The search could not fit this curve at any placement. "
                "Check the fitted range and the components ticked.")
    where = ", ".join(f"ε{sub} = {value:.2f} %" for sub, value
                      in zip("₁₂₃", found["best_pct"]))
    if found["reached"]:
        return (f"✅ **R² ≈ {found['r2']:.6f} ≥ {target:g}**, found "
                f"{found['said']}, at {where}. The boundaries are on the "
                "board as typed numbers, so this fit stays until you "
                "change them.")
    if found.get("stopped_early"):
        return (f"⏱️ **The best found so far is R² ≈ {found['r2']:.6f}**, "
                f"below {target:g}, at {where}. The search stopped there so "
                "the page would not keep you waiting. Press **🎯 Reach R² ≥ "
                f"{target:g}** and it will carry on from the prior, opening "
                "it out as far as it needs to.")
    return (f"⚠️ **The best this curve gives is R² ≈ {found['r2']:.6f}**, "
            f"below {target:g}, even {REACH_STEPS[-1][1]}. It is at {where}. "
            "A curve that will not reach the target usually has something "
            "the model does not describe in it: a rupture, a bad contact, "
            "or a component that is not ticked.")


def carried_forward_summary(bounds=None, target=0.999):
    """
    The whole method on one page: the model, the fit, and where ε comes from.

    Written to be read once and understood, in the order a person meets it:
    what is being fitted, how the moduli come out one at a time, and how the
    three boundaries are found before any of that happens.
    """
    names = components_for(st.session_state.get("cell_type"))
    order = component_order()
    b = list(bounds or piecewise_boundaries())
    edges = (", ".join(f"ε{sub} = {b[i + 1]:.2f} %"
                       for i, sub in enumerate("₁₂₃")) if len(b) >= 4 else "")

    st.markdown("#### 1 · The model")
    st.markdown(
        "The cell is met one component at a time. Each starts at its own "
        "boundary s_j and stops taking on more load at u_j, holding the "
        "force it had reached after that, so the curve never steps:"
    )
    st.latex(r"\phi_j(x) = \big[\min(x,\,u_j) - s_j\big]_+^{\,p_j}, "
             r"\qquad s_j \in \{0, \varepsilon_1, \varepsilon_2, "
             r"\varepsilon_3\}")
    st.markdown(
        "The three ways of fitting differ in u_j and in nothing else: "
        "**① carried on**, u_j = x_end for every component; **② one at a "
        "time**, u_j = the next boundary; **③ best mixture**, each "
        "component tried both ways and the best R² kept. On the stretch "
        "from ε_{k−1} to ε_k the force is everything found so far, plus "
        "the one new component:"
    )
    st.latex(
        r"\hat F(x) \;=\; \underbrace{\hat F(\varepsilon_{k-1})}"
        r"_{\text{where the last stretch ended}} \;+\; "
        r"\underbrace{C_k(x)}_{\text{the ones already fitted, carried on}}"
        r" \;+\; \underbrace{\theta_k\,\phi_k(x)}_{\text{the new one}}"
    )
    st.markdown(
        "with one shape per component — a shell being stretched goes as the "
        "cube of how far it is squashed, something squeezed between two "
        "plates as the three-halves power:"
    )
    st.latex(
        r"\phi_k(x) = \big[\,x - \varepsilon_{k-1}\,\big]_+^{\,p_k}, \qquad "
        r"p = 3 \;\;\text{(a shell)}, \qquad p = 3/2 \;\;"
        r"\text{(a Hertzian contact)}"
    )
    lines = []
    for index, term in enumerate(order):
        power = "3" if ORDER_COEFFICIENT[term] in ("K_shell", "K_nucleus") \
            else "3/2"
        start = "first contact" if index == 0 else f"ε{'₁₂₃'[index - 1]}"
        lines.append(f"{index + 1}. **{names[term][0]}** from {start}, "
                     f"as the power {power}")
    st.markdown("In this cell, in the order you have them met:\n\n"
                + "\n".join(lines))

    st.markdown("#### 2 · The fit: one unknown at a time")
    st.markdown(
        "Because everything found earlier is **held**, each stretch has "
        "exactly one unknown left, and the force depends on it in a straight "
        "line. That makes it an ordinary least squares with one coefficient: "
        "it has one answer, and no starting guess can change it."
    )
    st.latex(
        r"\hat\theta_k \;=\; \arg\min_{\theta \,\ge\, 0} "
        r"\sum_{x_i \in [\varepsilon_{k-1},\,\varepsilon_k)} "
        r"\Big[\, F_i - \hat F(\varepsilon_{k-1}) - C_k(x_i) - "
        r"\theta\,\phi_k(x_i) \,\Big]^{2}"
    )
    st.markdown(
        "The stretch starts from the force the one before it ended on, so "
        "the fitted curve has no step in it. The modulus is that coefficient "
        "divided by a prefactor fixed by the cell's geometry before any "
        "fitting, and its uncertainty comes out of the same least squares:"
    )
    st.latex(
        r"E_k = \frac{100^{\,p_k}\,\hat\theta_k}{A_k}, \qquad "
        r"\mathrm{SE} = \sqrt{\operatorname{diag}\!\big[\hat\sigma^{2}"
        r"(X^{\top}X)^{-1}\big]}, \qquad 95\,\%:\; \hat E \pm 1.96\,"
        r"\mathrm{SE}"
    )

    st.markdown("#### 3 · Where the boundaries come from")
    st.markdown(
        "All of that needs ε₁, ε₂ and ε₃ first, and they are read off this "
        "curve rather than assumed. Two things are tried, and the one that "
        "fits better is kept."
    )
    st.markdown(
        "**Step one — what the curve's own slope says.** A single power law "
        "F = a·xᵖ is a straight line on log–log axes, and its slope *is* the "
        "power:"
    )
    st.latex(r"p(x) \;=\; \frac{d\ln F}{d\ln x}")
    st.markdown(
        "A new component joining in bends that slope, so the places where it "
        "bends fastest are where the boundaries are. This needs no model at "
        "all, and it is usually right to within a few per cent:"
    )
    st.latex(r"\varepsilon^{PL}_j \;=\; \arg\max_{x \,\in\, B_j}\;"
             r"\left|\frac{dp}{d\ln x}\right|")
    st.markdown(
        "**Step two — which placement actually fits best.** Every placement "
        "near that reading is fitted in full, all four stretches with one "
        "unknown each, and the one that leaves the least error behind is "
        "kept:"
    )
    st.latex(
        r"\hat{\boldsymbol\varepsilon} \;=\; \arg\min_{\boldsymbol\varepsilon"
        r"\,\in\, B \,\cap\, [\boldsymbol\varepsilon^{PL} \pm 15\,\%]} "
        r"S(\boldsymbol\varepsilon), \qquad S = \sum_i \big(F_i - \hat F_i"
        r"\big)^{2}"
    )
    prior = piecewise_prior()
    (l1, h1), (l2, h2), _b3 = piecewise_bands()
    s_lo, s_hi = piecewise_span()
    st.markdown("**Step three — inside what this cell type allows.** B is "
                "the C2C12 prior, and no route may leave it:")
    st.latex(rf"{l1:g}\,\% \le \varepsilon_1 \le {h1:g}\,\%, \qquad "
             rf"{l2:g}\,\% \le \varepsilon_2 \le {h2:g}\,\%, \qquad "
             rf"{s_lo:g}\,\% \le \varepsilon_3 - \varepsilon_2 \le "
             rf"{s_hi:g}\,\%")
    if prior.get("why"):
        st.caption("Why those bands: " + prior["why"] + ".")
    known = literature_for()
    if known:
        windows = known.get("moduli") or {}
        st.markdown("**Step three and a half — what is already known about "
                    "this cell type.** Two things are known before this "
                    "curve is fitted, and both are about the answer rather "
                    "than the procedure:")
        lines = []
        for symbol, window in windows.items():
            unit, divisor = (("kPa", 1e3) if window[1] < 1e6 else ("MPa", 1e6))
            lines.append(
                rf"{window[0] / divisor:g}\,	ext{{{unit}}} \le "
                rf"{DISPLAY_SYMBOL.get(symbol, symbol)} \le "
                rf"{window[1] / divisor:g}\,	ext{{{unit}}}")
        if known.get("bump_pct") is not None:
            lines.append(rf"arepsilon_2 pprox {known['bump_pct']:g}\,\%")
        if lines:
            st.latex(r",\qquad ".join(lines))
        st.markdown(
            "The second is a bump in the curve at about "
            f"{known.get('bump_pct', 50):g} %, which is "
            f"{known.get('bump_is', 'the next component')} being met. "
            "Neither is imposed: no coefficient is clamped and no modulus "
            "is nudged, so every number reported is still the plain least "
            "squares estimate with its own standard error. They are used "
            "only where a prior belongs — to choose between placements the "
            "curve itself cannot tell apart, after the fewest-unmeasured "
            "rule and before the closest-fit one — and the results say "
            "which of them the answer agreed with."
        )
    st.markdown(
        "**Step four — and it has to work.** A placement is only kept if the "
        f"whole curve comes back at R² ≥ {target:g}. If it does not, the "
        "other routes in the table above are tried in turn and the first "
        "that reaches it is used — which is what the last line of that table "
        "is telling you."
        + (f" On this cell they came out at **{edges}**." if edges else "")
    )
    st.caption(
        "Every regime is linear in its one coefficient once its anchor is "
        "fixed, so the fit has one best answer inside its bounds. A "
        "coefficient that settles exactly on a bound is flagged in the "
        "results: it means the curve could not tell that component's shape "
        "from another's over its stretch, not that it was removed."
    )


def piecewise_placement_table(placements, bounds, target, selected):
    """
    Every route's boundaries, its fit and its Young's moduli, side by side.

    The point of the table: which placement to believe is decided by what
    each one does to the fit and to the moduli, and a person can see all of
    it at once and pick another row.
    """
    rows = placements.get("rows") or {}
    inner = tuple(round(float(v), 2) for v in bounds[1:4])
    symbols = ("E_shell", "E_cyto", "E_ne", "E_core")
    heads = {s_: f"{DISPLAY_SYMBOL[s_]} ({DISPLAY_UNIT[s_][0]})" for s_ in symbols}

    def same(key):
        row = rows.get(key)
        return bool(row) and inner == tuple(round(float(v), 2)
                                            for v in row["best_pct"])

    # One row marked in use: the chosen one when it is what is on the page,
    # otherwise the first that matches it.
    chosen = (selected or {}).get("key")
    in_use_key = chosen if same(chosen) else next(
        (k for k in ("refined", "power", "everything", "defaults") if same(k)),
        None)
    table = []
    for key in ("refined", "power", "everything", "defaults"):
        row = rows.get(key)
        if not row:
            if key in (placements.get("errors") or {}):
                table.append({"Placement": PW_ROW_LABELS[key], "ε₁ (%)": "—",
                              "ε₂ (%)": "—", "ε₃ (%)": "—", "R²": "—",
                              f"≥ {target:g}": "could not place",
                              **{heads[s_]: "" for s_ in symbols}, "In use": ""})
            continue
        r2 = row["r2"]
        in_use = key == in_use_key
        table.append({
            "Placement": PW_ROW_LABELS[key],
            "ε₁ (%)": f"{row['best_pct'][0]:.2f}",
            "ε₂ (%)": f"{row['best_pct'][1]:.2f}",
            "ε₃ (%)": f"{row['best_pct'][2]:.2f}",
            "R²": f"{r2:.5f}" if np.isfinite(r2) else "—",
            f"≥ {target:g}": ("✅" if np.isfinite(r2) and r2 >= target
                              else "⚠️ " + (f"{row['worst'][0]} {row['worst'][1]:.3f}"
                                            if row.get("worst") else "")),
            **{heads[s_]: modulus_display(s_, row["moduli"].get(s_)).rsplit(" ", 1)[0]
               for s_ in symbols},
            "In use": "◀ in use" if in_use else "",
        })
    st.markdown(r"**Routes** $r$: each row is $\hat F(x;\,\boldsymbol"
                r"\varepsilon_r)$, the full fit at that route's $\boldsymbol"
                r"\varepsilon$, with the model on this page")
    flat_table(
        pd.DataFrame(table),
        align_right=["ε₁ (%)", "ε₂ (%)", "ε₃ (%)", "R²"] + list(heads.values()),
        caption="⚠️ names the regime with the lowest R²_k. Used: the chosen "
        "route if R² ≥ R²★, else the first of (power law → fit everything, "
        "fit everything, power law, defaults) that reaches it.",
    )
    step = int((placements or {}).get("every_nth", 1) or 1)
    if step > 1:
        st.caption(
            f"Scored on every {step}ᵗʰ point: where a boundary lies does not "
            "need every point of a long curve to decide, and the fit the "
            "page shows is always the whole curve. These R² are for choosing "
            "between the rows, not the number reported above the graph."
        )
    reason = (selected or {}).get("reason")
    if reason:
        st.caption("**In use:** " + reason)
    cols = st.columns(len(rows))
    for col, key in zip(cols, [k for k in ("refined", "power", "everything",
                                           "defaults") if k in rows]):
        with col:
            st.button(f"Apply ε from: {PW_ROW_LABELS[key]}", key=f"pw_use_{key}_row",
                      on_click=_pw_use_row, args=(key,), **STRETCH)
    power = placements.get("power") or {}
    if power.get("success"):
        with st.expander("📈 The power law this curve follows (log–log slope)",
                         expanded=False):
            st.caption(
                "The local exponent p = d ln F / d ln x, read straight off "
                "the data. Each element is a power law, so p follows the laws "
                "carrying load, and where it turns upward hardest a new "
                "element has started: those turns are the power-law "
                "boundaries. The dotted lines are the mean exponent in each "
                "stretch."
            )
            st.plotly_chart(power_law_figure(power, bounds),
                            key="pw_power_law", **STRETCH)


def piecewise_search_maths(found):
    """How the boundaries were found: the maths and the three profiles."""
    st.markdown("**🎯 How the boundaries were found by fitting everything**")
    iv = found["intervals"]
    ref = " / ".join(f"{v:g}" for v in found["spec_pct"])
    st.markdown(
        " · ".join(
            f"{EPS_NAMES[i]} = {iv[k]['best']:.2f} % (68 %: {iv[k]['lo68']:.2f}–"
            f"{iv[k]['hi68']:.2f}, 95 %: {iv[k]['lo95']:.2f}–{iv[k]['hi95']:.2f})"
            + (" ⚠️ at the edge of its constraint" if iv[k]["at_band_edge"] else "")
            for i, k in enumerate(("eps1", "eps2", "eps3"))
        )
        + (f" · ΔBIC = {found['delta_bic']:.1f} against {ref} %. "
           if np.isfinite(found["delta_bic"]) else ". ")
        + found["verdict"]
    )
    st.markdown(
        "Once ε₁, ε₂, ε₃ are fixed, the continuous four-regime curve is "
        "linear in all seven coefficients, because every anchor is itself "
        "a sum of the coefficients upstream of it. So each trial placement "
        "is one exact bounded least-squares problem, with your equations, "
        "bounds and the membrane acting throughout if it is ticked:"
    )
    st.latex(r"\hat F(x) = X(x;\varepsilon_1,\varepsilon_2,\varepsilon_3)\,"
             r"\theta,\qquad \theta=(k_{align},C_0,K_{shell},K_{cyto},"
             r"K_{nucleus},K_{nuc\,cyto},K_{core})")
    st.latex(r"S(\varepsilon_1,\varepsilon_2,\varepsilon_3)="
             r"\min_{\theta_{lo}\le\theta\le\theta_{hi}}"
             r"\sum_i\big[F_i-\hat F(x_i)\big]^2,\qquad"
             r"\hat\varepsilon=\arg\min S")
    st.latex(r"\Delta(-2\ln L)(\varepsilon_j)=n\ln\frac{S}{S_{min}}"
             r"\;\le 1\ (68\,\%),\ \le 3.84\ (95\,\%)")
    st.latex(r"\Delta\mathrm{BIC}=n\ln\frac{S_{reference}}{S_{min}}-3\ln n")
    span = found.get("span_pct")
    st.caption(
        f"Searched ε₁ ∈ {found['bands_pct'][0][0]:g}–{found['bands_pct'][0][1]:g} %, "
        f"ε₂ ∈ {found['bands_pct'][1][0]:g}–{found['bands_pct'][1][1]:g} %"
        + (f", ε₃ − ε₂ ∈ {span[0]:g}–{span[1]:g} %" if span else
           f", ε₃ ∈ {found['bands_pct'][2][0]:g}–{found['bands_pct'][2][1]:g} %")
        + " (coarse grid, then coordinate descent at 0.1 %), "
        f"{found['n_evaluations']} placements, n = {found['n_points']} points "
        f"up to {found['end_pct']:.1f} %. ΔBIC above 6 is strong evidence "
        "that the curve itself places the boundaries there rather than "
        "at the defaults, 2 to 6 positive, below 0 none: the 3 ln n term "
        "is the price of letting the data choose three numbers. The intervals assume independent noise; residuals that "
        "run in long stretches make them narrower than they should be. "
        "The coefficients and moduli are then fitted sequentially, "
        "exactly as specified, at the boundaries in use."
    )
    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            st.plotly_chart(profile_figure(found, i),
                            key=f"pw_profile_{i}", **STRETCH)


def _pw_range_moved(name):
    """
    A component's range bar was moved: write it back to the model.

    The start is its regime's boundary, so moving it moves ε₁, ε₂ or ε₃,
    and with it every component that starts there. The end is the
    component's own: at its regime's end it follows the boundaries, and
    anywhere else it is kept as set. Runs as the slider's callback, before
    the page is redrawn, so the boundary inputs, the other bars, the fit
    and the plot all come back from the same numbers.
    """
    got = st.session_state.get(f"pw_range_{name}")
    if not got:
        return
    lo, hi = sorted(float(v) for v in got)
    b = list(piecewise_boundaries())
    end = b[4]
    if name == "k_align":  # not a component of this model
        return
    i = pw_start_index().get(name, PW_START_INDEX.get(name, 1))
    if i == 0:
        # The component met first starts at contact; there is no boundary
        # of its own to move, only where the next one joins.
        start = 0.0
    else:
        start = float(np.clip(lo, b[i - 1] + (0.5 if i == 1 else 1.0),
                              b[i + 1] - 1.0))
        st.session_state[PW_BOUNDARY_KEYS[i - 1]] = round(start, 2)
        b[i] = start
    if name == "A_lamina":
        # The lump spans the nuclear regime, so its far end is ε₃.
        st.session_state["pw_b3"] = round(float(np.clip(hi, start + 1.0, end - 1.0)), 2)
        return
    regime_end = b[i + 1]
    untils = dict(st.session_state.get("pw_until") or {})
    hi = float(np.clip(hi, start + 0.1, end))
    if name == "K_shell":
        # The membrane reaching the end of the fit is what "acts
        # throughout" means, so the tick and the bar are one setting.
        throughout = hi >= end - 0.05
        st.session_state["pw_membrane_throughout"] = throughout
        if throughout or abs(hi - regime_end) < 0.05:
            untils.pop(name, None)
        else:
            untils[name] = round(hi, 2)
    else:
        # Where it ends when nobody has moved it: the end of the fit for a
        # component the C2C12 prior has acting to the end, its regime's
        # end otherwise. Put back there, it follows again.
        default_end = end if name in piecewise_carry() else regime_end
        if abs(hi - default_end) < 0.05:
            untils.pop(name, None)
        else:
            untils[name] = round(hi, 2)
    st.session_state["pw_until"] = untils


def _pw_throughout_changed():
    """The tick decides the membrane's end, so a hand-set one is dropped."""
    untils = dict(st.session_state.get("pw_until") or {})
    untils.pop("K_shell", None)
    st.session_state["pw_until"] = untils


def piecewise_components_panel(bounds, moduli=None, lamina=None, columns=2):
    """
    The components table for the regime-by-regime fit.

    Builds one row per component and hands them to the table every way of
    sharing the load uses: the tick, the symbol, the law, the range [s, u]
    as a two-ended bar, and what the fit made of it.
    """
    ranges = piecewise_ranges(bounds)
    end = float(bounds[4])
    rows = []
    for name, label, symbol, _colour, law in PW_COMPONENTS:
        start, until = ranges[name]
        key = f"pw_range_{name}"
        # Set from the model every run, before the bar is drawn, so the
        # bar always shows the range the fit is about to use.
        shown_start = float(min(max(start, 0.0), end))
        shown_until = float(min(max(until, shown_start), end))
        st.session_state[key] = (round(shown_start, 2), round(shown_until, 2))
        extras = ()
        if name == "K_shell":
            def _throughout(on):
                st.checkbox("acts throughout (to the end of the fit)",
                            key="pw_membrane_throughout",
                            on_change=_pw_throughout_changed, disabled=not on)
            extras = (_throughout,)
        rows.append({
            "key": name, "tick_key": f"pw_use_{name}", "label": label,
            "symbol": symbol, "law": law,
            "default_on": DEFAULTS.get(f"pw_use_{name}", True),
            "help": "One of the four components, on by default. Ticked, it "
                    "is fitted and drawn; unticked, it is held at zero and "
                    "leaves both the fit and the plot. Everything on the "
                    "graph has a tick here.",
            "range_text": rf"$[s,u] = [{start:.1f},\,{until:.1f}]\,\%$",
            "extras": extras,
            "slider": {
                "label": f"{label} acts over (%)", "min_value": 0.0,
                "max_value": float(end), "step": 0.1, "key": key,
                "on_change": _pw_range_moved, "args": (name,),
                "label_visibility": "collapsed",
                "help": "Left end: where it starts carrying load, which is "
                        "its regime's boundary, so it moves that boundary. "
                        "Right end: where it stops adding load; past it the "
                        "component holds the force it reached.",
            },
        })
    slots = component_row_table(rows, columns=columns)
    for name, _label, _symbol, _colour, _law in PW_COMPONENTS:
        row = (moduli or {}).get(name)
        if not st.session_state.get(f"pw_use_{name}", True):
            slots[name].caption(r"$\theta = 0$ (held)")
        elif row:
            # Written exactly as the results table on the right writes it.
            slots[name].markdown(
                f"**{DISPLAY_SYMBOL.get(row['symbol'], row['symbol'])} = "
                f"{modulus_display(row['symbol'], row['E_Pa'])}**"
                + (" ⚠️ on bound" if row.get("at_bound") else ""))
    st.caption(
        r"Row $j$: $F_j(x) = \theta_j\,\phi_j(x;\,s_j,u_j)$, acting on "
        r"$[s_j, u_j]$ and held at $F_j(u_j)$ for $x > u_j$. "
        r"$s_j \in \{\varepsilon_1, \varepsilon_2, \varepsilon_3\}$ is shared "
        r"by the rows that start there; $u_j$ is the row's own. "
        "The layer and the bar of the same colour on the plot are this "
        "row. A ticked row is fitted and drawn; an unticked row is in "
        "neither, so the ticks and the graph always say the same thing."
    )
    return slots


# How the boundaries are placed. Two routes, and the one that chains them:
# read the bends off the curve's log-log slope (no model at all), or fit
# the whole model for every placement and keep the likeliest, or let the
# first tell the second where to look. Each is scored by the fit it gives
# and the Young's moduli that come out, side by side, so which one to
# believe is a table, not a leap of faith.
# On the board a route is written as what it does to ε, not as a name: the
# label is the expression that is minimised or read off, so the choice is
# made on the maths rather than on four words that all sound alike.
PW_METHODS = {
    "refined": "ε̂ = argmin S(ε),  ε ∈ B ∩ [ε_PL ± 15 %]",
    "power": "ε_PL = argmax dp/d ln x,   p = d ln F/d ln x",
    "everything": "ε̂ = argmin S(ε),  ε ∈ B",
    "defaults": "ε = ε_prior  (the C2C12 constraints' middle)",
}
# Step 2 asks one question -- are the boundaries found or kept? -- and the
# route that finds them is a detail of the first answer, under Advanced.
PW_EPS_WAYS = {
    "found": "🎯 Found from this curve",
    "typed": "✍️ Kept exactly as I typed them",
}


def _pw_eps_way_changed():
    """Callback: found or typed. Found re-chooses at once, typed moves nothing."""
    if st.session_state.get("pw_eps_way") == "found":
        _pw_apply_selection()
# The same routes written as names, for the table, the reasons and the
# spreadsheet, where an expression would read as noise.
PW_ROW_LABELS = {
    "refined": "📈→🎯 Power law, then fit everything",
    "power": "📈 Power law only",
    "everything": "🎯 Fit everything only",
    "defaults": "C2C12 defaults",
}
PW_METHOD_HELP = {
    "refined": r"$\hat{\boldsymbol\varepsilon} = \arg\min_{\boldsymbol\varepsilon"
               r"\,\in\, B\,\cap\,[\boldsymbol\varepsilon^{PL} \pm 15]} "
               r"S(\boldsymbol\varepsilon)$: the power law $\boldsymbol"
               r"\varepsilon^{PL}$ says where to look, fitting everything "
               r"decides.",
    "power": r"$\varepsilon_j^{PL} = \arg\max_{x \in B_j} \dfrac{dp}{d\ln x}$, "
             r"$\;p(x) = \dfrac{d\ln F}{d\ln x}$: no model, read off the "
             r"curve (typically within a few %).",
    "everything": r"$\hat{\boldsymbol\varepsilon} = \arg\min_{\boldsymbol"
                  r"\varepsilon \in B} S(\boldsymbol\varepsilon)$, "
                  r"$\;S = \min_{\theta_{lo}\le\theta\le\theta_{hi}} "
                  r"\lVert F - X(\boldsymbol\varepsilon)\theta\rVert^2$.",
    "defaults": r"$\boldsymbol\varepsilon$ = the C2C12 prior's defaults.",
}
# When the chosen route misses the R² target, the others are tried in this
# order and the first that reaches it is used.
PW_FALLBACK = ("refined", "everything", "power", "defaults")


def _pw_score(placement, epsilon, force_N, model):
    """Fit at one placement; its R², worst regime and moduli."""
    bounds = (0.0,) + tuple(float(v) for v in placement) + (
        float(st.session_state.get("pw_end", DEFAULTS["pw_end"])),)
    result = fit_piecewise(epsilon, force_N, boundaries_pct=bounds,
                           regimes=pw_regimes(),
                           settings=piecewise_model_settings(),
                           carry=piecewise_carry())
    if not result.get("success"):
        return {"best_pct": tuple(placement), "r2": float("nan"),
                "moduli": {}, "error": result.get("error", "")}
    moduli = piecewise_moduli(result, piecewise_geometry(model),
                              regimes=pw_regimes())
    fitted = [r for r in result["regimes"] if r["fitted"] and r["key"] != "R1"]
    worst = min(fitted, key=lambda r: r["r_squared"]) if fitted else None
    return {
        "best_pct": tuple(round(float(v), 2) for v in placement),
        "r2": float(result["r_squared"]),
        "worst": (worst["key"], float(worst["r_squared"])) if worst else None,
        "moduli": {row["symbol"]: float(row["E_Pa"]) for row in moduli.values()},
    }


# A boundary search fits the whole model once per placement on a grid, so
# its cost is the number of points times the size of that grid. Where a
# boundary is does not need every point of a 6000-point curve to be
# decided -- a few hundred per stretch settles it to far better than the
# 0.5 % the boundaries are read to -- so the search runs on an evenly
# thinned copy. The fit the page shows is always the whole curve.
SEARCH_POINTS = 1800
# Thinned harder while a curve is loading, where the whole point is not to
# keep anybody waiting; put back to None as soon as that search is done.
_SEARCH_POINTS_OVERRIDE = [None]


def thinned_for_search(epsilon, force_N):
    """An evenly thinned copy of the curve, for placing boundaries on."""
    most = int(_SEARCH_POINTS_OVERRIDE[0] or SEARCH_POINTS)
    n = int(np.size(epsilon))
    if n <= most:
        return epsilon, force_N, 1
    step = int(np.ceil(n / most))
    return np.asarray(epsilon)[::step], np.asarray(force_N)[::step], step


def compute_placements(model, epsilon, force_N):
    """
    Every route's boundaries for this curve, each scored by its own fit.

    Returns {"signature", "rows": {key: score}, "power": ..., "searches":
    {key: find_boundaries result}}.
    """
    signature = piecewise_signature(epsilon, force_N)
    epsilon, force_N, step = thinned_for_search(epsilon, force_N)
    bands, span = piecewise_bands(), piecewise_span()
    end = float(st.session_state.get("pw_end", DEFAULTS["pw_end"]))
    common = dict(end_pct=end, span_pct=span, regimes=pw_regimes(),
                  settings=piecewise_model_settings(), carry=piecewise_carry())
    out = {"signature": signature, "rows": {}, "searches": {}, "errors": {},
           "every_nth": step}
    # 1 · the power law, read straight off the curve
    power = boundaries_from_power_law(
        power_law_profile(epsilon, force_N, end_pct=end), bands, span)
    out["power"] = power
    if power.get("success"):
        out["rows"]["power"] = _pw_score(power["best_pct"], epsilon, force_N, model)
        # 2 · fitting everything, near where the power law says to look
        near = tuple((max(b[0], p - 15.0), min(b[1], p + 15.0))
                     for b, p in zip(bands, power["best_pct"]))
        found = find_boundaries(epsilon, force_N, bands_pct=near,
                                spec_pct=power["best_pct"], **common)
        if found.get("success"):
            out["searches"]["refined"] = found
            out["rows"]["refined"] = _pw_score(found["best_pct"], epsilon, force_N, model)
        else:
            out["errors"]["refined"] = found.get("error", "")
    else:
        out["errors"]["power"] = power.get("error", "")
    # 3 · fitting everything over the whole constraint range
    found = find_boundaries(epsilon, force_N, bands_pct=bands,
                            spec_pct=piecewise_defaults()[:3], **common)
    if found.get("success"):
        out["searches"]["everything"] = found
        out["rows"]["everything"] = _pw_score(found["best_pct"], epsilon, force_N, model)
    else:
        out["errors"]["everything"] = found.get("error", "")
    # and the defaults, for reference
    out["rows"]["defaults"] = _pw_score(piecewise_defaults()[:3], epsilon, force_N, model)
    return out


def select_placement(placements, method, target):
    """
    Which placement to use: the chosen route's, unless it misses the target.

    Returns (key, reason). A route below the R² target hands over to the
    first in PW_FALLBACK that reaches it; when none does, the best fit is
    used and the reason says so.
    """
    rows = placements.get("rows") or {}
    # "typed" used to be one of the routes; it is now the other answer to
    # step 2, so a session that still holds it asks for the recommended
    # route rather than for a KeyError.
    if method not in PW_METHODS:
        method = "refined"

    def r2(key):
        value = (rows.get(key) or {}).get("r2", float("nan"))
        return value if np.isfinite(value) else -np.inf

    def unmeasured(key):
        """How many components this placement leaves at exactly zero."""
        moduli = (rows.get(key) or {}).get("moduli") or {}
        return sum(1 for value in moduli.values()
                   if not np.isfinite(value) or abs(float(value)) < 1e-12)

    def agrees(key):
        """How well this placement agrees with what is reported."""
        row = rows.get(key) or {}
        best_pct = row.get("best_pct") or ()
        eps2 = best_pct[1] if len(best_pct) > 1 else None
        inside, away = literature_fit_score(row.get("moduli"), eps2)
        return inside, -away

    # Every placement that reaches the target, best first: fewest components
    # left unmeasured, then highest R². A boundary that puts one component
    # on top of another leaves that one at zero, and a placement that
    # measures all four is a better answer than one that measures three and
    # fits a hair closer.
    # Every placement that reaches the target, best first: fewest
    # components left unmeasured, then the one that agrees with what is
    # reported for this cell type, then the closest fit. The curve decides
    # what is possible; the literature only chooses between placements the
    # curve cannot tell apart.
    reached = sorted((k for k in rows if r2(k) >= target),
                     key=lambda k: (unmeasured(k), [-v for v in agrees(k)],
                                    -r2(k)))
    if method in rows and r2(method) >= target:
        if not reached or unmeasured(method) <= unmeasured(reached[0]):
            return method, (f"{PW_ROW_LABELS[method]} reaches R² = "
                            f"{r2(method):.5f} ≥ {target:g}.")
        key = reached[0]
        return key, (
            f"{PW_ROW_LABELS[method]} reaches R² = {r2(method):.5f} but "
            f"leaves {unmeasured(method)} component"
            + ("s" if unmeasured(method) != 1 else "")
            + f" at zero, so {PW_ROW_LABELS[key]} is used: it also reaches "
              f"the target (R² = {r2(key):.5f}) and measures "
            + ("every component." if not unmeasured(key)
               else f"{unmeasured(key)} fewer at zero."))
    if reached:
        key = reached[0]
        said = (f"{PW_ROW_LABELS[method]} reached R² = {r2(method):.5f}, "
                f"below the {target:g} target"
                if method in rows else
                f"{PW_ROW_LABELS[method]} could not place the boundaries")
        return key, (f"{said}, so {PW_ROW_LABELS[key]} is used "
                     f"(R² = {r2(key):.5f}).")
    if not rows:
        return None, "No placement could be fitted."
    key = max(rows, key=r2)
    return key, (f"No placement reaches R² ≥ {target:g}; the best, "
                 f"{PW_ROW_LABELS[key]} (R² = {r2(key):.5f}), is used. See "
                 "which regime fits worst in the table.")


def _pw_apply_selection():
    """Callback: the route or the target changed, so re-choose and apply."""
    placements = st.session_state.get("pw_placements")
    if not placements or st.session_state.get("pw_eps_way") == "typed":
        return
    key, reason = select_placement(
        placements, st.session_state.get("pw_method", "refined"),
        float(st.session_state.get("pw_target_r2", 0.999)))
    if key is None:
        return
    for k, v in zip(PW_BOUNDARY_KEYS, placements["rows"][key]["best_pct"]):
        st.session_state[k] = round(float(v), 2)
    st.session_state["pw_selected"] = {"key": key, "reason": reason}


def _pw_use_row(key):
    """Callback: use one row of the placement table as it is."""
    placements = st.session_state.get("pw_placements") or {}
    row = (placements.get("rows") or {}).get(key)
    if not row:
        return
    for k, v in zip(PW_BOUNDARY_KEYS, row["best_pct"]):
        st.session_state[k] = round(float(v), 2)
    st.session_state["pw_selected"] = {
        "key": key, "reason": f"{PW_ROW_LABELS[key]}, chosen from the table."}


def current_placements(epsilon, force_N):
    """The stored placements, if they belong to this curve and model."""
    placements = st.session_state.get("pw_placements")
    if placements and placements.get("signature") == piecewise_signature(
            epsilon, force_N):
        return placements
    return None


def power_law_figure(power, bounds):
    """The curve's local exponent, where it bends, and the boundaries."""
    prof = power.get("profile") or {}
    x = np.asarray(prof.get("x_pct", []), dtype=float)
    p = np.asarray(prof.get("exponent", []), dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=p, mode="lines", name="d ln F / d ln x",
                             line={"color": "#1f77b4", "width": 2.5}))
    for seg in power.get("segments") or []:
        if np.isfinite(seg["mean_exponent"]):
            fig.add_trace(go.Scatter(
                x=[seg["from_pct"], seg["to_pct"]],
                y=[seg["mean_exponent"]] * 2, mode="lines",
                line={"color": "#aaaaaa", "width": 1, "dash": "dot"},
                showlegend=False,
                hovertemplate=f"mean exponent {seg['mean_exponent']:.2f}<extra></extra>",
            ))
    # The exponents the model's laws have, so the reading can be held
    # against them: a stretch sitting on p = 3 is a shell stretching, one
    # on 3/2 a Hertzian contact, one on 1 a network already under tension.
    p_lo = float(np.nanmin(p)) if p.size else 0.0
    p_hi = float(np.nanmax(p)) if p.size else 3.0
    for level, colour, said in ((3.0, "#2ca02c", "p = 3 · shell"),
                                (1.5, "#9467bd", "p = 3/2 · Hertz"),
                                (1.0, "#8c564b", "p = 1 · tension")):
        fig.add_hline(y=level, line_width=1, line_dash="dot",
                      line_color=colour, annotation_text=said,
                      annotation_position="right",
                      annotation_font_size=11, annotation_font_color=colour)
        p_lo, p_hi = min(p_lo, level), max(p_hi, level)
    levels = add_boundary_lines(fig, (0.0,) + tuple(power["best_pct"]) + (bounds[-1],),
                                end_label=False,
                                x_span=float(x.max() - x.min()) if x.size else None,
                                x_max=float(x.max()) if x.size else None)
    pad = 0.12 * max(p_hi - p_lo, 1.0)
    fig.update_layout(
        height=300, template="simple_white", showlegend=False,
        # Room on the right for the p = 3, 3/2 and 1 tags, and a y range
        # that holds every one of those lines as well as the reading.
        margin={"l": 60, "r": 96, "t": 30 + 24 * levels, "b": 50},
        xaxis_title="Relative deformation (%)",
        yaxis={"title": "local exponent p", "range": [p_lo - pad, p_hi + pad]},
    )
    return fig


def piecewise_settings_used(result, geometry, target, source):
    """
    Every setting the fit on the page used, read from the fit itself.

    Built from what was passed to the fit and what came back, never from
    the widgets alone, so it cannot show a setting the fit did not use.
    The spring-network options that have no part in this fit are named as
    such, so a value set for them is not mistaken for part of this one.
    """
    b = result["boundaries_pct"]
    ranges = result.get("ranges") or {}
    off = piecewise_off()
    rows = [
        ("Model", "4-regime piecewise, fitted regime by regime (R1 → R4), "
                  "each anchored at the force the one before it ended on",
         "fixed"),
        ("Boundary placement", f"{PW_ROW_LABELS.get(st.session_state.get('pw_method', 'refined'), '')} "
                               f"· in use: {source}",
         "How to place the boundaries"),
        ("Boundaries", ", ".join(f"{n} = {v:.2f} %" for n, v in zip(EPS_NAMES, b[1:4]))
         + f", end = {b[4]:.1f} %", "Regime boundaries"),
        ("R² target", f"≥ {target:g} · this fit {result['r_squared']:.5f} "
                      + ("✅" if result["r_squared"] >= target else "⚠️"),
         "Fit must reach R² ≥"),
    ]
    (l1, h1), (l2, h2), _b3 = piecewise_bands()
    s_lo, s_hi = piecewise_span()
    rows.append(("C2C12 constraints", f"ε₁ {l1:g}–{h1:g} %, ε₂ {l2:g}–{h2:g} %, "
                                      f"ε₃ − ε₂ {s_lo:g}–{s_hi:g} %",
                 "Edit the C2C12 constraints"))
    for name, label, _symbol, _colour, _law in PW_COMPONENTS:
        a, u = ranges.get(name, (float("nan"), float("nan")))
        rows.append((f"{label}", "off (held at 0)" if name in off
                     else f"acts over {a:.2f}–{u:.2f} %", "Components"))
    for regime in result["regimes"]:
        for name, p in regime["params"].items():
            if name == "C0" or name == "k_align":
                continue
            rows.append((f"{name} guess and bounds",
                         f"p0 {p['p0']:.3g}, [{_pw_text(p['lower'])}, "
                         f"{_pw_text(p['upper'])}]"
                         + (" (switched off)" if name in off else ""),
                         "Initial guesses and bounds"))
    rows += [
        ("C₀ over [0, ε₁)", "the constant the first stretch is fitted with, "
         "by least squares and with no bounds", "fixed"),
        ("Weighting", "uniform: every point counts the same (ordinary least "
                      "squares)", "fixed"),
        ("Solver", " · ".join(f"{r['key']}: {r['engine'] or 'not fitted'}"
                              for r in result["regimes"]), "fixed"),
        ("Cell height h₀", f"{geometry.cell_height * 1e6:.2f} µm", "1 · Cell information"),
        ("Cell radius R₀", f"{geometry.cell_radius * 1e6:.2f} µm "
                           f"({st.session_state.get('radius_mode', '')})",
         "sidebar · Cell geometry"),
        ("Nucleus radius Rₙ", f"{geometry.nucleus_radius * 1e6:.2f} µm "
                              f"({st.session_state.get('nucleus_radius_mode', '')})",
         "sidebar · Cell geometry"),
        ("Probe", (f"{geometry.probe_radius * 2e6:.1f} µm sphere"
                   if geometry.probe_radius else "flat (not recorded)"),
         "1 · Cell information"),
        ("Membrane / envelope / coat thickness",
         f"{geometry.membrane_thickness * 1e9:.1f} / "
         f"{geometry.envelope_thickness * 1e9:.0f} / "
         f"{geometry.coat_thickness * 1e9:.0f} nm", "sidebar"),
        ("Poisson ratios (membrane / interior / nucleus)",
         f"{geometry.nu_membrane:.2f} / {geometry.nu_interior:.2f} / "
         f"{geometry.nu_nucleus:.2f}", "sidebar"),
        ("Not used by this fit",
         "the weighting, confinement q, nucleus onset and arrangement "
         "settings belong to the spring-network models", "—"),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Used in this fit", "Set in"])


def collection_name():
    """What this cell is called in the collection."""
    name = str(st.session_state.get("cell_name") or "").strip()
    if name:
        return name
    source = str((st.session_state.get("data") or {}).get("source") or "cell")
    return source.rsplit(".", 1)[0]


def collection_record(name, source, height_um, epsilon, force_N, result,
                      geometry, route=""):
    """One cell as the collection keeps it: its curve, its ε, its numbers."""
    moduli = piecewise_moduli(result, geometry, regimes=pw_regimes())
    lamina = lamina_summary(result, geometry) or {}
    return {
        "name": name, "source": source, "height_um": float(height_um),
        "epsilon": np.asarray(epsilon, dtype=float),
        "force_N": np.asarray(force_N, dtype=float),
        "fitted_N": predict_piecewise(epsilon, result),
        "bounds_pct": tuple(float(v) for v in result["boundaries_pct"]),
        "route": route,
        "r2": float(result["r_squared"]),
        "chi2_nu": float(result.get("chi_squared_reduced", float("nan"))),
        "n": int(result["n_points"]),
        "moduli": {row["symbol"]: float(row["E_Pa"]) for row in moduli.values()},
        "lamina_A_N": float(lamina.get("A_N", float("nan"))),
        "lamina_work_J": float(lamina.get("work_J", float("nan"))),
        "settings": piecewise_model_settings(),
        "carry": list(result.get("carry") or ()),
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }


def component_results_strip(fit, extra=""):
    """
    What the cell is made of, across the top of the page.

    One card per component: its modulus as best fit ± one standard error,
    and the 95 % interval under it. This is the answer the page exists to
    give, so it is the first thing on it, before the curve; how good the
    fit is and where its boundaries fell are one line under the cards.
    """
    if not (fit and fit.get("success")):
        st.info("Tick the components the cell is made of below, then press "
                "**▶ Fit & plot**. Their moduli appear here, each with its "
                "uncertainty.")
        return
    # Only the components this fit is made of: an unticked one is not a
    # measurement of zero, it is absent, and the strip says what the cell
    # was found to be made of.
    rows = [row for row in result_rows(fit)
            if row[0].startswith("modulus_") and row[2] != "not in this model"]
    if not rows:
        st.warning("No component is ticked, so there is nothing to report. "
                   "Tick at least one below and press ▶ Fit & plot.")
        return
    for card, (_key, label, value, _se, interval, support) in zip(
            st.columns(len(rows)), rows):
        with card:
            with st.container(border=True):
                st.caption(label)
                st.markdown(f"### {value}")
                # A component the fit put at exactly zero was not dropped by
                # the page: it is ticked, it was fitted, and the fit could
                # not tell its shape from another one's over this stretch.
                # Saying so is the difference between a bug and a reading.
                if str(value).split(" ")[0] in ("0", "0.0", "-0"):
                    st.caption(
                        "⚠️ **measured as zero here — it has not been "
                        "removed.** It is ticked, it is in the model, it was "
                        "fitted, and the least squares came back at its "
                        "lower bound: over this stretch its shape could not "
                        "be told from another component's, so that force "
                        "went to the other one. Only your tick takes a "
                        "component out of the model. Move the boundary it "
                        "starts at, change the order the components are met "
                        "in, or untick **acts throughout** on the component "
                        "before it — a law carried on and extrapolated can "
                        "cover the whole of the next stretch and leave "
                        "nothing for the component that joins there.")
                else:
                    st.caption(f"95 % interval: {interval}")
                st.caption(support)
    said_lit = literature_note(fit)
    if said_lit:
        st.caption(said_lit)
    chi = fit.get("chi_squared_reduced", float("nan"))
    where = " · ".join(f"{name} = {value:.3f}" for value, name in fit_edges(fit))
    st.caption(
        "Each value is the best fit ± one standard error, "
        r"SE = √diag[σ̂²(XᵀX)⁻¹]; the interval is ± 1.96 SE. · "
        + (f"`fit {fit['fit_id']}` · " if fit.get("fit_id") else "")
        + f"R² = {float(fit.get('r_squared', float('nan'))):.5f}"
        + (f" · χ²/dof = {float(chi):.3g}" if np.isfinite(chi) else "")
        + f" · {int(fit.get('n_points', 0))} points"
        + (f" · {where}" if where else "")
        + (f" · {extra}" if extra else "")
    )


def piecewise_section(model, epsilon, force_N, rupture):
    """
    The piecewise fit: the graph and the results on top, the control board
    under them, one button.

    The control board holds every parameter of the fit: how the components
    share the load, the components and their ranges, how ε is placed and ε
    itself, R²★, p₀ and bounds, and how the plot draws the components.
    Changes wait on the board until ▶ Fit & plot, which places ε (unless it
    is to be used as typed), fits, and redraws the graph and the results;
    those are all written from that one applied fit, so they cannot
    disagree. A new curve is fitted once on arrival, so it never opens on
    an empty graph.

    Returns (fit, fitted, stage_plan, result) for the sections below.
    """
    if st.session_state.pop("_pw_editor_reset", False):
        st.session_state.pop("pw_param_editor", None)
    # The page draws a fit the moment a curve is loaded, so the ticks have
    # to be on before that fit is made: what is on the graph is what is
    # ticked, and on arrival that is all four components.
    for _name in PW_SWITCHABLE:
        st.session_state.setdefault(f"pw_use_{_name}", True)
    # A session left over from when "as typed" was one of the routes.
    if st.session_state.get("pw_method") not in PW_METHODS:
        st.session_state["pw_method"] = "refined"
        st.session_state["pw_eps_way"] = "typed"

    top = float(np.nanmax(epsilon)) * 100.0 if np.size(epsilon) else 100.0
    data = st.session_state.get("data") or {}
    curve_key = repr((data.get("source"), int(np.size(epsilon)),
                      round(float(force_N[-1]), 15) if np.size(force_N) else 0.0))

    def place_now(arriving=False):
        """
        ε from the chosen route, written to the board before it is drawn.

        On arrival the curve has never been fitted here, so the route is
        not the person's setting left over from the cell before: every
        route is scored and the one that reaches R² ≥ R²★ is the one the
        page starts on. Only when none of them reaches it is the best
        used, and the status line says so.
        """
        method = ("refined" if arriving
                  else st.session_state.get("pw_method", "refined"))
        if not arriving and st.session_state.get("pw_eps_way") == "typed":
            return
        if arriving:
            st.session_state["pw_method"] = "refined"
            st.session_state["pw_eps_way"] = "found"
        placements = current_placements(epsilon, force_N)
        if placements is None:
            with st.spinner("Placing ε₁, ε₂, ε₃ for this cell…"):
                placements = compute_placements(model, epsilon, force_N)
            st.session_state["pw_placements"] = placements
        key, reason = select_placement(
            placements, method, float(st.session_state.get("pw_target_r2", 0.999)))
        if key is None:
            st.warning("No placement satisfies the constraints on this curve; "
                       "ε is used as typed.", icon="⚠️")
            return
        st.session_state["pw_selected"] = {"key": key, "reason": reason}
        for k, v in zip(PW_BOUNDARY_KEYS, placements["rows"][key]["best_pct"]):
            st.session_state[k] = round(float(v), 2)

    def settle_mixture():
        """Which components carry on, when that is a thing to be found."""
        if piecewise_style() != "best":
            st.session_state["pw_best_carry"] = None
            st.session_state["pw_mixture"] = None
            return
        with st.spinner("Trying every component both ways…"):
            found_mix = best_mixture(epsilon, force_N, model)
        st.session_state["pw_mixture"] = found_mix
        st.session_state["pw_best_carry"] = (list(found_mix["carry"])
                                             if found_mix else [])

    def apply_board():
        st.session_state["pw_applied"] = {"curve": curve_key,
                                          "values": pw_board_values()}

    # A new curve: its own ε, fitted at once. A press of ▶ that placed ε
    # comes back here with the new ε already on the board, to be applied.
    # A change of the way the cell is fitted settles its mixture and is
    # applied at once: it is a change of model, not a change of setting.
    if st.session_state.pop("_pw_restyle", False):
        settle_mixture()
        st.session_state["_pw_apply"] = True
    applied_state = st.session_state.get("pw_applied") or {}
    if st.session_state.pop("_pw_apply", False):
        apply_board()
    elif applied_state.get("curve") != curve_key:
        # A C2C12 arrives fitted the way a C2C12 is fitted: carried
        # forward, all four components, and boundaries already taken as far
        # as the target. The search starts inside the prior, so a curve the
        # prior describes costs one placement search, exactly as before;
        # only a curve it does not costs more, and then it says so.
        st.session_state["pw_reach_note"] = None
        st.session_state["pw_method"] = "refined"
        st.session_state["pw_eps_way"] = "found"
        # A curve arrives fitted, with its four components and with its
        # boundaries moved until the whole curve reaches the target. The
        # search starts inside the C2C12 prior, so a curve the prior
        # describes costs the one placement search this page has always
        # done; only a curve it does not costs more, and that is exactly
        # the curve nobody wants to place by hand.
        settle_mixture()
        target_now = float(st.session_state.get("pw_target_r2", 0.999))
        with st.spinner(f"Fitting this cell: placing ε₁, ε₂, ε₃ and moving "
                        f"them until R² ≥ {target_now:g}…"):
            arrived = reach_the_target(
                model, epsilon, force_N, target_now,
                budget_seconds=ARRIVAL_BUDGET_S, interleave=True,
                search_points=ARRIVAL_SEARCH_POINTS)
        if arrived:
            st.session_state["pw_placements"] = arrived.get("placements")
            st.session_state["pw_selected"] = {
                "key": arrived["route"],
                "reason": f"On arrival: {arrived['said']}, "
                          f"R² = {arrived['r2']:.5f}.",
            }
            for key, value in zip(PW_BOUNDARY_KEYS, arrived["best_pct"]):
                st.session_state[key] = round(float(value), 2)
            st.session_state["pw_membrane_throughout"] = bool(
                arrived.get("throughout", True))
            if arrived.get("widened"):
                # Found outside the prior, or with the model changed, so it
                # is kept as typed: placing ε again inside the prior would
                # throw it away.
                st.session_state["pw_eps_way"] = "typed"
                st.session_state["pw_reach_note"] = reach_note(arrived,
                                                               target_now)
        else:
            place_now(arriving=True)
        # The boundaries are settled; now check the answer they give
        # against what is known about this cell type. Every component
        # carried on to the end is the model the page is set up with, but
        # a carried cube law can cover the whole of the next stretch and
        # leave the component that joins there at zero, or two orders of
        # magnitude from where the literature puts it. When that happens,
        # the mixture that measures every component and agrees with what
        # is reported is used instead, and the page says so.
        st.session_state["pw_style_note"] = None
        if piecewise_style() == "carried" and literature_for():
            found_mix = best_mixture(epsilon, force_N, model)
            everything = tuple(ORDER_COEFFICIENT[t] for t in component_order())
            as_set = next((row for row in (found_mix or {}).get("rows", [])
                           if set(row["carry"]) == set(everything)), None)
            if found_mix and as_set and (
                    (found_mix["unmeasured"], found_mix["inside"])
                    != (as_set["unmeasured"], as_set["inside"])
                    and (found_mix["unmeasured"] < as_set["unmeasured"]
                         or found_mix["inside"] > as_set["inside"])):
                st.session_state["pw_style"] = "best"
                st.session_state["pw_best_carry"] = list(found_mix["carry"])
                st.session_state["pw_mixture"] = found_mix
                st.session_state["pw_style_note"] = (
                    "ℹ️ Every component carried on to the end left "
                    + (f"{as_set['unmeasured']} of them unmeasured"
                       if as_set["unmeasured"] else
                       "the moduli away from what is reported for this cell "
                       "type")
                    + ", so **③ the best mixture** was used instead: "
                    + mixture_note(found_mix["carry"])
                    + f" (R² = {found_mix['r2']:.5f}). Choose "
                    "**① Carried on to the end** above to see it the other "
                    "way.")
        apply_board()
    applied = st.session_state["pw_applied"]["values"]

    # ---- ① ② ③ ④, then the graph beside the results --------------------
    stepper_slot = st.container()
    graph_col, results_col = st.columns([1.75, 1], gap="large")
    with graph_col:
        graph_slot = st.container()
    with results_col:
        results_slot = st.container()

    # ================================================= the control board
    # Directly under the plot, and the components are the first thing in
    # it: they are the plot, one tick per curve drawn and one bar per
    # range shown, so they are read before the rest of the parameters.
    board = st.container(border=True)
    with board:
        # The board is read top to bottom and it is four steps: what the
        # cell is made of, where its parts take over, the one button, and
        # everything else out of the way behind Advanced. Nothing on it
        # ticks or unticks a component: the ticks are the person's, and
        # every fit is made of exactly what they have ticked.
        st.markdown("#### 🎛️ Fitting")
        status_slot = st.empty()

        st.markdown("**Step 1 · What the cell is made of**")
        share_col, model_col = st.columns([1.2, 1], gap="medium")
        with share_col:
            load_sharing_control()
        with model_col:
            st.latex(r"\hat F_k(x) = \hat F_{k-1}(\varepsilon_{k-1}) + "
                     r"\theta_k\,\phi_k(x)")
        st.markdown("**The order they are met in**")
        component_order_control()
        value_slots = piecewise_components_panel(
            piecewise_boundaries(), None, None, columns=2)
        components_note_slot = st.empty()
        # What the board holds for each row, to compare with what the plot
        # is drawing: the two lists must read the same, or say they differ.
        board_off = set(piecewise_off())
        board_ranges = piecewise_ranges(piecewise_boundaries())

        st.markdown("**Step 2 · Where its parts take over**")
        st.latex(r"0 < \varepsilon_1 < \varepsilon_2 < \varepsilon_3 "
                 r"\le x_{end}")
        st.radio(
            "on ▶ Fit & plot", list(PW_EPS_WAYS),
            format_func=PW_EPS_WAYS.get, key="pw_eps_way",
            on_change=_pw_eps_way_changed, horizontal=True,
            label_visibility="collapsed",
            help="Found from this curve: ▶ Fit & plot places the boundaries "
                 "and overwrites the numbers below with what it found. Kept "
                 "as typed: the fit happens exactly at the numbers below and "
                 "nothing moves.",
        )
        e1c, e2c, e3c, endc = st.columns(4)
        with e1c:
            st.number_input("ε₁ (%)", 0.5, 99.0, step=0.5, format="%.2f",
                            key="pw_b1", help=EPS_ROLES[0])
        with e2c:
            st.number_input("ε₂ (%)", 1.0, 99.0, step=0.5, format="%.2f",
                            key="pw_b2", help=EPS_ROLES[1])
        with e3c:
            st.number_input("ε₃ (%)", 1.0, 99.5, step=0.5, format="%.2f",
                            key="pw_b3", help=EPS_ROLES[2])
        with endc:
            st.number_input("x_end (%)", 1.0, 100.0, step=0.1, format="%.1f",
                            key="pw_end", help="Last point fitted; not moved "
                            "by the routes.")
        for note in constraint_notes(piecewise_boundaries()):
            st.caption(f"⚠️ {note}.")
        if (rupture or {}).get("method") == "force-drop" and rupture.get("epsilon"):
            at = float(rupture["epsilon"]) * 100.0
            if 0.0 < at < min(float(st.session_state["pw_end"]), top):
                st.caption(rf"ℹ️ Force drop at $x = {at:.1f}\,\%$ (possible "
                           rf"rupture): set $x_{{end}} = {at:.1f}$ to exclude it.")

        st.markdown("**Step 3 · Fit**")
        nothing_ticked = len(piecewise_off()) >= len(PW_SWITCHABLE) + 1
        go1, go2, go3 = st.columns([1.1, 1.3, 1])
        with go1:
            pressed = st.button(
                "▶ Fit & plot", type="primary", key="pw_fit_plot",
                disabled=nothing_ticked,
                help="Fits the components ticked in step 1, at the "
                "boundaries step 2 gives, and redraws everything on this "
                "page from that one fit.",
                **STRETCH,
            )
        with go2:
            reaching = st.button(
                f"🎯 Reach R² ≥ {float(_pw_get('pw_target_r2', 0.999)):g}",
                key="pw_reach", disabled=nothing_ticked,
                help="Searches for boundaries that bring the whole curve to "
                "the target, opening the C2C12 constraints out a step at a "
                "time until it gets there, and says what it had to open. "
                "The boundaries it finds are written on the board as typed "
                "numbers.",
                **STRETCH,
            )
        with go3:
            refreshed = st.button(
                "🔄 Refresh graph", key="pw_refresh",
                disabled=nothing_ticked,
                help="Redraws the graph and the results at exactly what the "
                "board holds now. Moves no boundary and chooses nothing.",
                **STRETCH,
            )
        if nothing_ticked:
            st.warning("Tick at least one component in step 1 before "
                       "fitting.", icon="⚠️")
        else:
            st.caption(
                "**▶ Fit & plot** uses the board as it stands. "
                "**🎯 Reach R²** goes looking for boundaries that meet the "
                "target, widening the prior only as far as it has to. "
                "**🔄 Refresh graph** just draws the board again."
            )

        # The board holds what is decided here and nothing else. The route
        # that finds ε is chosen by the fit itself -- every route is scored
        # and the first that reaches R²★ is used -- and the table under the
        # graph says which, with a button to take any other. The guesses,
        # the bounds and the constraints on ε are the C2C12 prior's and are
        # written out under that table rather than sitting here as boxes
        # nobody sets.

    # 🎯 Reach R²: open the constraints out a step at a time until the
    # whole curve meets the target, and say what had to be opened.
    if reaching:
        with st.spinner("Looking for boundaries that reach the target…"):
            found_reach = reach_the_target(model, epsilon, force_N,
                                           float(_pw_get("pw_target_r2", 0.999)))
        st.session_state["pw_reach_note"] = reach_note(
            found_reach, float(_pw_get("pw_target_r2", 0.999)))
        if found_reach:
            st.session_state["pw_placements"] = found_reach.get("placements")
            st.session_state["pw_selected"] = {
                "key": found_reach["route"],
                "reason": f"🎯 Reach R²: {found_reach['said']}, "
                          f"R² = {found_reach['r2']:.5f}.",
            }
            rerun_keeping_settings({
                **{key: round(float(value), 2) for key, value in
                   zip(PW_BOUNDARY_KEYS, found_reach["best_pct"])},
                "pw_membrane_throughout": bool(
                    found_reach.get("throughout", True)),
                "pw_eps_way": "typed",
                "_pw_apply": True,
            })

    # 🔄 Refresh graph: draw the board as it stands, moving nothing.
    if refreshed:
        apply_board()
        applied = st.session_state["pw_applied"]["values"]

    # ▶ Fit & plot: place ε if the route says so, then apply the board.
    if pressed:
        # A new fit of its own: whatever the last target search had to say
        # was about the boundaries it found, not about these.
        st.session_state["pw_reach_note"] = None
        settle_mixture()
        if st.session_state.get("pw_eps_way") == "typed":
            apply_board()
            applied = st.session_state["pw_applied"]["values"]
        else:
            method = st.session_state.get("pw_method", "refined")
            placements = current_placements(epsilon, force_N)
            if placements is None:
                with st.spinner("Placing ε₁, ε₂, ε₃ for this cell…"):
                    placements = compute_placements(model, epsilon, force_N)
                st.session_state["pw_placements"] = placements
            key, reason = select_placement(
                placements, method, float(st.session_state.get("pw_target_r2", 0.999)))
            values = {}
            if key is not None:
                st.session_state["pw_selected"] = {"key": key, "reason": reason}
                values = {k: round(float(v), 2) for k, v in
                          zip(PW_BOUNDARY_KEYS, placements["rows"][key]["best_pct"])}
            rerun_keeping_settings({**values, "_pw_apply": True})

    waiting = pw_board_values() != applied
    with status_slot.container():
        if waiting:
            st.warning("The board has changes the plot does not show yet. "
                       "Press **▶ Fit & plot** to apply them.", icon="⚠️")
        else:
            st.success("The graph and the results show the settings on this "
                       "board.", icon="✅")
        said_reach = st.session_state.get("pw_reach_note")
        if said_reach:
            st.caption(said_reach)
        said_style = st.session_state.get("pw_style_note")
        if said_style:
            st.caption(said_style)
        if piecewise_style() == "best":
            found_mix = st.session_state.get("pw_mixture")
            if found_mix:
                st.caption(
                    "**The mixture kept:** " + mixture_note(found_mix["carry"])
                    + f" · best of {len(found_mix['rows'])} ways, "
                    f"R² = {found_mix['r2']:.5f}.")

    # ============================ the applied fit: graph, results, working
    with pw_applied_values(applied):
        target = float(_pw_get("pw_target_r2", 0.999))
        result = run_piecewise_fit(model, epsilon, force_N)
        style = current_style(force_N)
        placements = current_placements(epsilon, force_N)
        selected = st.session_state.get("pw_selected") or {}
        found = (placements or {}).get("searches", {}).get(selected.get("key"))
        geometry = piecewise_geometry(model)
        ok = bool(result.get("success"))
        source = boundary_source(result["boundaries_pct"], placements) if ok else "—"
        lamina = lamina_summary(result, geometry) if ok else None
        name = collection_name()
        collection = st.session_state.get("pw_collection") or {}
        in_collection = name in collection
        moduli = (result.get("moduli") or {}) if ok else {}
        fit = (piecewise_as_fit(result, model, found=found, placements=placements)
               if ok else None)
        fid = fit["fit_id"] if fit else ""
        off_now = piecewise_off()

        with stepper_slot:
            component_results_strip(
                fit, extra=(source if ok else "")
                + (f" · “{name}” is in the collection" if in_collection else ""))

        # What the fit made of each component, beside its row on the board,
        # and a mark on any row the plot is not drawing yet.
        if ok:
            shown = {row[0]: row[2] for row in result_rows(fit)}
            applied_ranges = result.get("ranges") or {}

            def _same(name):
                if (name in board_off) != (name in off_now):
                    return False
                a1 = tuple(round(float(v), 2) for v in board_ranges.get(name, ()))
                a2 = tuple(round(float(v), 2) for v in applied_ranges.get(name, ()))
                return a1 == a2

            for key_, term in (("K_shell", "membrane"), ("K_cyto", "interior"),
                               ("K_nucleus", "nucleus_shell"),
                               ("K_core", "nucleus")):
                slot = value_slots.get(key_)
                if slot is None:
                    continue
                symbol = next((c[2] for c in PW_COMPONENTS if c[0] == key_), "")
                waits = "" if _same(key_) else " ⏳ *not on the plot yet*"
                if key_ in off_now:
                    slot.markdown(r"$\theta = 0$ (held, off the plot)" + waits)
                else:
                    slot.markdown(
                        f"**{symbol} = {shown.get(f'modulus_{term}', '—')}**"
                        + waits)
            waiting_rows = [c[1] for c in PW_COMPONENTS if not _same(c[0])]
            components_note_slot.caption(
                "⏳ **" + ", ".join(waiting_rows) + "**: ticked or moved "
                "here, but not on the graph yet. Press **▶ Fit & plot**."
                if waiting_rows else
                "✓ Every ticked component above is fitted and drawn on the "
                "graph, over the range on its bar; nothing else is."
            )

        fitted = None
        with graph_slot:
            if not ok:
                st.error(f"The fit is not defined here: {result.get('error', '')}. "
                         "Adjust ε on the board and press ▶ Fit & plot.")
            else:
                used = result["boundaries_pct"]
                chi = result.get("chi_squared_reduced", float("nan"))
                meets = result["r_squared"] >= target
                line = (
                    f"`fit {fid}` · "
                    + rf"$R^2 = {result['r_squared']:.5f}\;"
                    + (r"\ge" if meets else "<") + rf"\;R^2_\star = {target:g}$"
                    + (rf" · $\chi^2_\nu = {chi:.3g}$" if np.isfinite(chi) else "")
                    + rf" · $(\varepsilon_1,\varepsilon_2,\varepsilon_3) = "
                    rf"({used[1]:.2f},\,{used[2]:.2f},\,{used[3]:.2f})\,\%$"
                    + f" · {source}"
                )
                if meets:
                    st.success(line)
                else:
                    regimes = [r for r in result["regimes"]
                               if r["fitted"] and r["key"] != "R1"]
                    worst = min(regimes, key=lambda r: r["r_squared"]) if regimes else None
                    st.warning(
                        line + (rf" · worst $R_{{{worst['key'][1]}}}$: "
                                rf"$R^2 = {worst['r_squared']:.4f}$"
                                if worst else "")
                        + f". Press **🎯 Reach R² ≥ {target:g}** on the board "
                        "and it will look for boundaries that meet it, "
                        "widening the C2C12 constraints only as far as it "
                        "has to.", icon="⚠️")
                log_y = bool(st.session_state.get("pw_log_y"))
                st.plotly_chart(
                    piecewise_figure(
                        epsilon, force_N, result, style, log_y=log_y,
                        off=off_now,
                        view=st.session_state.get("pw_view", "stacked"),
                        note=f"fit {fid} · R² = {result['r_squared']:.5f}",
                    ),
                    key="pw_curve", **STRETCH,
                )
                # What is on the graph right now, in one line, under it.
                piecewise_graph_note(
                    result, off=off_now,
                    view=st.session_state.get("pw_view", "stacked"),
                    log_y=log_y, fit_id_text=fid,
                    n_points=int(np.size(epsilon)),
                )
                # How the plot draws it, under the plot, where it is being
                # looked at. Both apply at once; neither changes the fit.
                v1, v2 = st.columns([2.4, 1])
                with v1:
                    st.radio("Components on the plot", list(PW_VIEWS),
                             format_func=PW_VIEWS.get, key="pw_view",
                             horizontal=True, label_visibility="collapsed")
                with v2:
                    st.checkbox("log F axis", key="pw_log_y")
                fitted = predict_piecewise(epsilon, result)

        with results_slot:
            if ok:
                section("Fitting results")
                fitting_results_rows(fit, style, send=False, compact=True)
                with st.expander("🧮 The answer, written out as mathematics",
                                 expanded=False):
                    piecewise_answer(result, fit)
                copy_the_results(fit, style.force_unit)
                adding = st.button(
                    ("↻ Update (ε, θ̂, E) in collection" if in_collection
                     else "➕ Store (ε, θ̂, E) in collection"),
                    key="pw_add", disabled=not ok, **STRETCH,
                    help="Keeps this cell (its curve, its own ε, its moduli) "
                    "for the 📚 All cells tab. Named after the cell name in "
                    "section 1, or the file.",
                )
                if adding:
                    collection = dict(collection)
                    collection[name] = collection_record(
                        name, data.get("source", ""),
                        st.session_state["cell_height_um"],
                        epsilon, force_N, result, geometry,
                        PW_ROW_LABELS.get(selected.get("key"), source),
                    )
                    st.session_state["pw_collection"] = collection
                    rerun_keeping_settings()

        # ---- how it is done, for an undergraduate, with these numbers --
        with st.expander("📘 How the fit is done — the maths, step by step",
                         expanded=False):
            fit_explainer(fit, result if ok else None)

        # ---- what decided it, from the same fit ------------------------
        t_routes, t_coef, t_model, t_set, t_work = st.tabs([
            "ε routes compared", "θ̂ by regime", "Model F(x)",
            "Settings used", "🔍 Working",
        ])
        with t_routes:
            if placements:
                piecewise_placement_table(placements, piecewise_boundaries(),
                                          target, selected)
            else:
                st.caption("Press **▶ Fit & plot** to place ε on this curve "
                           "and compare the routes.")
            st.divider()
            carried_forward_summary(piecewise_boundaries(), target)
        with t_coef:
            if ok:
                piecewise_coefficient_table(result, moduli)
        with t_model:
            if ok:
                used = result["boundaries_pct"]
                st.markdown("**Model**")
                for line in piecewise_equations_latex(used, result.get("ranges") or {},
                                                      piecewise_off()):
                    st.latex(line)
                st.markdown("**Estimator** (sequential, $k = 1\\ldots4$, "
                            "$\\varepsilon_0 = 0$, $\\varepsilon_4 = x_{end}$)")
                st.latex(r"\hat\theta_k = \arg\min_{\theta_{lo}\le\theta\le\theta_{hi}}"
                         r"\sum_{x_i \in [\varepsilon_{k-1},\varepsilon_k)}\Big[F_i - "
                         r"\hat F(\varepsilon_{k-1}) - C_k(x_i) - \sum_{j \in k}\theta_j"
                         r"\phi_j(x_i)\Big]^2")
                st.latex(r"\hat F_k(\varepsilon_{k-1}) = \hat F_{k-1}(\varepsilon_{k-1})"
                         r"\quad(C^0),\qquad C_k = \text{elements carried in from } "
                         r"j < k \text{ with their } \hat\theta_j")
                st.markdown("**Moduli** ($p_j$ = exponent, $A_j$ = prefactor)")
                st.latex(r"E_j = \frac{100^{p_j}\,K_j}{A_j},\quad A_{shell} = "
                         r"\frac{2\pi h_m R_0}{1-\nu_m},\quad A_{ne} = \frac{2\pi h_{ne}"
                         r" R_n}{1-\nu_n},\quad A_{H}(R) = \frac{\sqrt2\,R^2}{3(1-\nu^2)}"
                         r"\,c(R)")
                st.latex(r"c(R) = \left[\frac{2R^{-1/3}}{R_*^{-1/3}+R^{-1/3}}\right]^{3/2},"
                         r"\;R_* = \frac{R R_p}{R+R_p},\qquad T_{align} = "
                         r"\frac{100\,k_{align}}{2\pi R_0^2/h_0},\;E_{align} = "
                         r"\frac{T_{align}}{h_{coat}}")
                st.latex(r"W_L = \int L\,d\delta = A_L\,\frac{\varepsilon_3-"
                         r"\varepsilon_2}{2}\,\frac{h_0}{100},\qquad R^2 = 1 - "
                         r"\frac{\sum (F_i-\hat F_i)^2}{\sum (F_i-\bar F)^2}")
        with t_set:
            if ok:
                flat_table(piecewise_settings_used(result, geometry, target, source))
        with t_work:
            if ok:
                st.markdown("**Residuals** $r_i = F_i - \\hat F(x_i)$")
                st.plotly_chart(
                    piecewise_residual_figure(epsilon, force_N, fitted, style),
                    key="pw_residuals", **STRETCH,
                )
                anchors_disp = " · ".join(
                    f"$F({name.split('_')[1].replace('pct', '')}\\%) = "
                    f"{float(from_newtons(value, style.force_unit)[0]):.4g}$ "
                    f"{from_newtons(value, style.force_unit)[1]}"
                    for name, value in result["anchors"].items()
                )
                gap = max((abs(v) for v in (result.get("continuity_gaps_N") or {}).values()),
                          default=0.0)
                st.markdown("**Anchors** (C⁰): " + anchors_disp
                            + rf" · $\max|\Delta F| = {gap:.1g}$ N")
                c_cell = probe_correction(geometry.cell_radius, geometry.probe_radius)
                c_nuc = probe_correction(geometry.nucleus_radius, geometry.probe_radius)
                st.markdown(
                    rf"**Geometry**: $h_0 = {geometry.cell_height * 1e6:.2f}$ µm, "
                    rf"$R_0 = {geometry.cell_radius * 1e6:.2f}$ µm, "
                    rf"$R_n = {geometry.nucleus_radius * 1e6:.2f}$ µm, "
                    + (rf"$R_p = {geometry.probe_radius * 1e6:.1f}$ µm, "
                       rf"$c(R_0) = {c_cell:.3f}$, $c(R_n) = {c_nuc:.3f}$, "
                       if geometry.probe_radius else r"$R_p = \infty$ (flat), ")
                    + rf"$h_m = {geometry.membrane_thickness * 1e9:.1f}$ nm, "
                    rf"$h_{{ne}} = {geometry.envelope_thickness * 1e9:.0f}$ nm, "
                    rf"$\nu = ({geometry.nu_membrane:.2f}, {geometry.nu_interior:.2f}, "
                    rf"{geometry.nu_nucleus:.2f})$"
                )
                st.markdown("**Solver**: " + " · ".join(
                    f"{r['key']}: {r['engine'] or '—'}, $n = {r['n_points']}$"
                    for r in result["regimes"]))
                if found:
                    piecewise_search_maths(found)
                for warning in result.get("warnings", []):
                    st.caption(f"⚠️ {warning}")

        if not ok:
            return None, None, [], None

        used = result["boundaries_pct"]
        export = pd.DataFrame([
            {
                "coefficient": n, "regime": row["regime"], "element": row["element"],
                "K": row["K"], "K_se": row["K_se"], "power": row["power"],
                "modulus": row["symbol"], "E_Pa": row["E_Pa"], "E_se_Pa": row["E_se_Pa"],
                "prefactor_N_per_Pa": row["prefactor_N_per_Pa"],
                "probe_correction": row["probe_correction"],
                "at_bound": row["at_bound"],
                "eps1_pct": used[1], "eps2_pct": used[2], "eps3_pct": used[3],
                "end_pct": used[4],
            }
            for n, row in moduli.items()
        ] + ([{
            "coefficient": "A_lamina", "regime": "R3", "element": "Nuclear lamina (lump)",
            "K": lamina["A_N"], "K_se": lamina["A_se_N"], "power": 0,
            "modulus": "A_L", "E_Pa": float("nan"), "E_se_Pa": float("nan"),
            "eps1_pct": used[1], "eps2_pct": used[2], "eps3_pct": used[3],
            "end_pct": used[4],
        }] if lamina else []))
        with results_col:
            st.download_button(
                "📥 This cell's coefficients and moduli, one row each (CSV)",
                data=export.to_csv(index=False),
                file_name=f"{name}_4regime_coefficients.csv", mime="text/csv",
                key="pw_download",
            )
        return fit, fitted, piecewise_stage_plan(result), result


def piecewise_coefficient_table(result, moduli):
    """Every fitted and carried coefficient, regime by regime."""
    table = []
    off = piecewise_off()
    for regime in result["regimes"]:
        a, z = regime["domain_pct"]
        common = {
            "k": regime["key"],
            "[ε_{k-1}, ε_k) (%)": f"{a:.2f}–{z:.2f}",
            "n": regime["n_points"],
            "R²_k": (f"{regime['r_squared']:.4f}"
                     if np.isfinite(regime["r_squared"]) else "—"),
        }
        for n, p in regime["params"].items():
            row = moduli.get(n, {})
            value = p["value"]
            unit = ("N" if n == "C0" or p.get("shape") == "lump"
                    else f"N/%^{p['power']:g}")
            acts = ("—" if n == "C0" or "start" not in p
                    else f"{p['start']:.1f}–{p['until']:.1f}")
            table.append({
                **common, "θ_j": n, "[s, u] (%)": "off" if n in off else acts,
                "θ̂ ± SE": ("—" if not np.isfinite(value) else
                            f"{value:.4g}" + (f" ± {p['se']:.2g}"
                                              if np.isfinite(p["se"]) else ""))
                + f" {unit}",
                "p₀": "—" if not np.isfinite(p["p0"]) else f"{p['p0']:.3g}",
                "[θ_lo, θ_hi]": f"[{_pw_text(p['lower'])}, {_pw_text(p['upper'])}]",
                "E ± SE": (f"{DISPLAY_SYMBOL.get(row['symbol'], row['symbol'])} = "
                           f"{modulus_display(row['symbol'], row['E_Pa'], row['E_se_Pa'])}"
                           if row else ""),
                "flag": "at bound" if p.get("at_bound") else "",
            })
        for n, c in (regime.get("carried") or {}).items():
            if not regime["fitted"]:
                continue
            table.append({
                **common, "θ_j": f"{n} (carried)",
                "[s, u] (%)": f"{c['onset_pct']:.1f}–{c.get('until_pct', a):.1f}",
                "θ̂ ± SE": f"{c['value']:.4g} N/%^{c['power']:g}", "p₀": "—",
                "[θ_lo, θ_hi]": f"fixed ({c.get('from_regime', 'R2')})",
                "E ± SE": "", "flag": "C_k",
            })
    flat_table(pd.DataFrame(table),
               align_right=["n", "R²_k", "θ̂ ± SE", "p₀", "E ± SE"])


# ================================================================ all cells ==
#
# Cells are individuals: each has its own height, its own boundaries and
# its own moduli. The collection keeps each one with its curve, so they
# can be laid over one another, compared element by element, and saved and
# reloaded as a set.

CELL_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
               "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173")
COLLECTION_SYMBOLS = ("E_shell", "E_cyto", "E_ne", "E_core")


def refit_collection_cell(record, height_um=None):
    """Fit one kept cell again, at its own ε, with a (new) height."""
    height = float(record["height_um"] if height_um is None else height_um)
    eps, force = record["epsilon"], record["force_N"]
    model = build_model(eps, force, height_um=height)
    result = fit_piecewise(eps, force, boundaries_pct=record["bounds_pct"],
                           regimes=pw_regimes(),
                           settings=record.get("settings") or {},
                           carry=tuple(record.get("carry") or ()))
    if not result.get("success"):
        return record
    new = collection_record(record["name"], record.get("source", ""), height,
                            eps, force, result, piecewise_geometry(model),
                            record.get("route", ""))
    new["include"] = record.get("include", True)
    return new


def fit_cell_from_file(name, epsilon, force_N, height_um):
    """A new cell from a file: its own ε placed, then fitted, then kept."""
    model = build_model(epsilon, force_N, height_um=height_um)
    placements = compute_placements(model, epsilon, force_N)
    key, _reason = select_placement(
        placements, st.session_state.get("pw_method", "refined"),
        float(st.session_state.get("pw_target_r2", 0.999)))
    placement = (placements["rows"][key]["best_pct"] if key
                 else piecewise_defaults()[:3])
    bounds = (0.0,) + tuple(placement) + (
        float(st.session_state.get("pw_end", DEFAULTS["pw_end"])),)
    result = fit_piecewise(epsilon, force_N, boundaries_pct=bounds,
                           regimes=pw_regimes(),
                           settings=piecewise_model_settings(),
                           carry=piecewise_carry())
    if not result.get("success"):
        return None, result.get("error", "fit failed")
    record = collection_record(name, name, height_um, epsilon, force_N, result,
                               piecewise_geometry(model),
                               PW_ROW_LABELS.get(key, "C2C12 defaults"))
    return record, ""


def collection_frame(collection):
    """One row per cell: its ε, its fit and its numbers."""
    rows = []
    for rec in collection.values():
        b = rec["bounds_pct"]
        rows.append({
            "cell": rec["name"], "include": bool(rec.get("include", True)),
            "h₀ (µm)": float(rec["height_um"]),
            "ε₁ (%)": round(b[1], 2), "ε₂ (%)": round(b[2], 2),
            "ε₃ (%)": round(b[3], 2), "R²": round(rec["r2"], 5),
            "route": rec.get("route", ""),
            **{f"{DISPLAY_SYMBOL[s_]} ({DISPLAY_UNIT[s_][0]})":
               round(rec["moduli"].get(s_, float("nan")) / DISPLAY_UNIT[s_][1], 4)
               for s_ in COLLECTION_SYMBOLS},
        })
    return pd.DataFrame(rows)


def collection_to_json(collection):
    def plain(value):
        if isinstance(value, np.ndarray):
            return [None if not np.isfinite(v) else float(v) for v in value]
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, tuple):
            return list(value)
        return value
    return json.dumps({"cells": [{k: plain(v) for k, v in rec.items()}
                                 for rec in collection.values()]},
                      default=float)


def collection_from_json(text):
    cells = {}
    for rec in json.loads(text).get("cells", []):
        for key in ("epsilon", "force_N", "fitted_N"):
            rec[key] = np.array([np.nan if v is None else v
                                 for v in rec.get(key, [])], dtype=float)
        rec["bounds_pct"] = tuple(rec.get("bounds_pct", ()))
        cells[rec["name"]] = rec
    return cells


def collection_overlay_figure(cells, normalise=False, log_y=False,
                              show_data=True):
    """Every kept cell's curve and fit on one plot, each with its own ε."""
    fig = go.Figure()
    marks = {1: "circle", 2: "square", 3: "diamond"}
    for i, rec in enumerate(cells):
        colour = CELL_COLORS[i % len(CELL_COLORS)]
        x = rec["epsilon"] * 100.0
        f = rec["force_N"] * 1e9
        fit = np.asarray(rec["fitted_N"], dtype=float) * 1e9
        scale = 1.0
        if normalise:
            peak = np.nanmax(fit) if np.isfinite(np.nanmax(fit)) else np.nanmax(f)
            scale = 1.0 / peak if peak and np.isfinite(peak) else 1.0
        if show_data:
            fig.add_trace(go.Scatter(
                x=x, y=f * scale, mode="markers", legendgroup=rec["name"],
                marker={"color": colour, "size": 3, "opacity": 0.3},
                showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=x, y=fit * scale, mode="lines", name=rec["name"],
            legendgroup=rec["name"], line={"color": colour, "width": 2},
            hovertemplate=f"{rec['name']}<br>x = %{{x:.1f}} %<br>"
                          f"F = %{{y:.4g}}<extra></extra>",
        ))
        # This cell's own ε₁, ε₂, ε₃, on its own fitted curve.
        ex = np.array(rec["bounds_pct"][1:4], dtype=float)
        ey = np.interp(ex, x[np.isfinite(fit)], fit[np.isfinite(fit)]) * scale \
            if np.isfinite(fit).any() else np.zeros(3)
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="markers", legendgroup=rec["name"],
            marker={"color": colour, "size": 9,
                    "symbol": [marks[1], marks[2], marks[3]],
                    "line": {"color": "#000000", "width": 1}},
            showlegend=False,
            hovertemplate=[f"{rec['name']}: ε{j} = {v:.2f} %<extra></extra>"
                           for j, v in zip((1, 2, 3), ex)],
        ))
    fig.update_layout(
        height=520, template="simple_white",
        margin={"l": 70, "r": 20, "t": 20, "b": 110},
        xaxis_title="x = relative deformation (%)",
        yaxis_title="F / F_max" if normalise else "F (nN)",
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "x": 0.0},
    )
    if log_y:
        fig.update_layout(yaxis={"type": "log"})
    return fig


def collection_eps_figure(cells):
    """Each cell's ε₁, ε₂, ε₃ on one row: how the boundaries differ."""
    fig = go.Figure()
    names = [rec["name"] for rec in cells]
    for j, (symbol, colour) in enumerate(
            (("circle", "#555555"), ("square", "#1f77b4"), ("diamond", "#9467bd")), 1):
        fig.add_trace(go.Scatter(
            x=[rec["bounds_pct"][j] for rec in cells], y=names, mode="markers",
            name=f"ε{j}", marker={"symbol": symbol, "size": 11, "color": colour},
            hovertemplate="%{y}: %{x:.2f} %<extra>" + f"ε{j}</extra>",
        ))
    for rec in cells:
        fig.add_trace(go.Scatter(
            x=[0, rec["bounds_pct"][4]], y=[rec["name"]] * 2, mode="lines",
            line={"color": "#dddddd", "width": 1}, showlegend=False,
            hoverinfo="skip",
        ))
    fig.update_layout(
        height=120 + 28 * len(cells), template="simple_white",
        margin={"l": 120, "r": 20, "t": 10, "b": 60},
        xaxis_title="x (%)",
        legend={"orientation": "h", "yanchor": "top", "y": -0.25, "x": 0.0},
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def collection_strip_figure(cells, symbol, values, unit):
    """One quantity across cells: every cell a point, the median a line."""
    names = [rec["name"] for rec in cells]
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.15, 0.15, len(values))
    finite = np.array([v for v in values if np.isfinite(v) and v > 0])
    fig = go.Figure(go.Scatter(
        x=jitter, y=values, mode="markers", text=names,
        marker={"size": 10, "color": [CELL_COLORS[i % len(CELL_COLORS)]
                                      for i in range(len(values))],
                "line": {"color": "#000000", "width": 1}},
        hovertemplate="%{text}: %{y:.4g} " + unit + "<extra></extra>",
    ))
    if finite.size:
        med = float(np.median(finite))
        fig.add_hline(y=med, line_color="#000000", line_width=1.5,
                      annotation_text=f"median {med:.3g}",
                      annotation_position="top right")
    fig.update_layout(
        height=260, template="simple_white", showlegend=False,
        margin={"l": 60, "r": 10, "t": 30, "b": 20},
        title={"text": symbol, "font": {"size": 14}},
        xaxis={"visible": False, "range": [-0.5, 0.5]},
        yaxis={"title": unit, "type": "log" if finite.size and
               finite.max() / max(finite.min(), 1e-30) > 20 else "linear"},
    )
    return fig


def unit_from_header(header):
    """The force unit a column header names, e.g. 'Force (nN)', else N."""
    text = str(header)
    for key in INPUT_FORCE_UNITS:
        short = key.split(" ")[0]
        for variant in {short, short.replace("μ", "µ"), short.replace("μ", "u")}:
            if f"({variant})" in text or f"[{variant}]" in text or \
                    text.strip().endswith(" " + variant):
                return key
    return next(iter(INPUT_FORCE_UNITS))


def all_cells_tab():
    """The 📚 All cells tab."""
    st.markdown(
        r"Each cell $i$ is fitted with its own height $h_0^{(i)}$ and its own "
        r"boundaries $\boldsymbol\varepsilon^{(i)}$, placed on its own curve, "
        r"giving its own $\hat\theta^{(i)}$ and $E^{(i)}$. Nothing is "
        r"averaged before it is plotted."
    )
    collection = dict(st.session_state.get("pw_collection") or {})

    with st.expander("➕ Add cells from files (each fitted on its own)",
                     expanded=not collection):
        files = st.file_uploader(
            "Force curves (.csv or .xlsx), one cell per file",
            type=["csv", "xlsx", "xls"], accept_multiple_files=True,
            key="pw_batch_files",
        )
        u1, u2 = st.columns(2)
        with u1:
            unit_choice = st.selectbox(
                "Force unit in the files",
                ["from the column name"] + list(INPUT_FORCE_UNITS.keys()),
                key="pw_batch_unit",
                help="'from the column name' reads N, mN, µN, nN or pN out of "
                "a header such as 'Force (nN)', and takes N when there is none.")
        with u2:
            height = st.number_input(
                "h₀ for these cells (µm)", 0.1, 100.0,
                value=float(st.session_state.get("cell_height_um", 8.0)),
                step=0.01, format="%.2f", key="pw_batch_height",
                help="Each can be changed afterwards in the table below.")
        st.caption(
            "Each file: the deformation and force columns are recognised by "
            "name; a deformation above 1.5 is read as percent. ε is placed "
            "on each curve by the route and R²★ chosen on the analysis tab.")
        if files and st.button(f"Fit all {len(files)} files: per-cell ε, θ̂, E",
                               type="primary",
                               key="pw_batch_go"):
            progress = st.progress(0.0)
            problems = []
            for i, upload in enumerate(files):
                name = upload.name.rsplit(".", 1)[0]
                try:
                    frame = load_table(upload.getvalue(), upload.name)
                    cols = frame.columns.tolist()
                    e_col = cols[guess_column(cols, ("reldef", "rel def", "rel_def",
                                                     "deform", "eps", "ε", "strain"), 0)]
                    f_col = cols[guess_column(cols, ("force", "f (", "f["), 1)]
                    unit = (unit_from_header(f_col) if unit_choice ==
                            "from the column name" else unit_choice)
                    eps = pd.to_numeric(frame[e_col], errors="coerce").to_numpy(float)
                    force = to_newtons(pd.to_numeric(frame[f_col], errors="coerce")
                                       .to_numpy(float), unit)
                    good = np.isfinite(eps) & np.isfinite(force)
                    eps, force = eps[good], force[good]
                    order = np.argsort(eps, kind="stable")
                    eps, force = eps[order], force[order]
                    if eps.size and eps.max() > 1.5:
                        eps = eps / 100.0
                    record, problem = fit_cell_from_file(name, eps, force, height)
                    if record:
                        collection[name] = record
                    else:
                        problems.append(f"{upload.name}: {problem}")
                except Exception as exc:  # one bad file must not stop the rest
                    problems.append(f"{upload.name}: {exc}")
                progress.progress((i + 1) / len(files))
            st.session_state["pw_collection"] = collection
            for problem in problems:
                st.warning(problem)
            st.success(f"{len(files) - len(problems)} of {len(files)} cells "
                       "fitted and added.")

    with st.expander("📂 Load or save the collection", expanded=False):
        saved = st.file_uploader("A collection saved from here (.json)",
                                 type=["json"], key="pw_collection_file")
        if saved is not None and st.button("Load collection (.json)",
                                           key="pw_collection_load"):
            try:
                collection.update(collection_from_json(saved.getvalue().decode()))
                st.session_state["pw_collection"] = collection
                st.success(f"{len(collection)} cells in the collection.")
            except Exception as exc:
                st.error(f"Could not read it: {exc}")
        if collection:
            st.download_button(
                "💾 Save the collection (curves included, .json)",
                data=collection_to_json(collection),
                file_name=f"afm_cells_{datetime.now():%Y%m%d}.json",
                mime="application/json", key="pw_collection_save")

    if not collection:
        st.info("No cells yet. Fit a cell on 📊 Force curve analysis and press "
                "**➕ Store (ε, θ̂, E) in collection**, or add files above.")
        return

    st.markdown(f"#### {len(collection)} cells")
    frame = collection_frame(collection)
    edited = st.data_editor(
        frame, key="pw_cells_editor", hide_index=True, num_rows="fixed",
        disabled=[c for c in frame.columns if c not in ("include", "h₀ (µm)")],
        **STRETCH,
    )
    changed = False
    for _, row in edited.iterrows():
        rec = collection.get(row["cell"])
        if rec is None:
            continue
        if bool(row["include"]) != bool(rec.get("include", True)):
            rec = dict(rec, include=bool(row["include"]))
            changed = True
        if abs(float(row["h₀ (µm)"]) - float(rec["height_um"])) > 1e-9:
            # A new height changes the prefactors, so the moduli, not ε.
            rec = refit_collection_cell(rec, float(row["h₀ (µm)"]))
            rec["include"] = bool(row["include"])
            changed = True
        collection[row["cell"]] = rec
    if changed:
        st.session_state["pw_collection"] = collection
        rerun_keeping_settings()
    st.caption(r"Only $h_0$ and *include* are editable: a new $h_0^{(i)}$ "
               r"refits that cell's moduli at its own $\boldsymbol"
               r"\varepsilon^{(i)}$.")

    cells = [rec for rec in collection.values() if rec.get("include", True)]
    if not cells:
        st.info("Tick *include* for at least one cell to plot it.")
        return

    o1, o2, o3 = st.columns(3)
    with o1:
        normalise = st.checkbox("F / F_max", key="pw_cells_norm")
    with o2:
        log_y = st.checkbox("log F", key="pw_cells_log")
    with o3:
        show_data = st.checkbox("show data points", value=True,
                                key="pw_cells_data")
    st.plotly_chart(collection_overlay_figure(cells, normalise, log_y, show_data),
                    key="pw_cells_overlay", **STRETCH)
    st.caption("Lines: each cell's fitted F(x). Markers on each line: that "
               "cell's own ε₁ (●), ε₂ (■), ε₃ (◆).")

    st.markdown(r"**Boundaries per cell** $\boldsymbol\varepsilon^{(i)}$")
    st.plotly_chart(collection_eps_figure(cells), key="pw_cells_eps", **STRETCH)

    st.markdown(r"**Moduli per cell**: one point per cell, line = median")
    quantities = [(DISPLAY_SYMBOL[s_],
                   [rec["moduli"].get(s_, float("nan")) / DISPLAY_UNIT[s_][1]
                    for rec in cells],
                   DISPLAY_UNIT[s_][0]) for s_ in COLLECTION_SYMBOLS]
    for start in range(0, len(quantities), 4):
        cols = st.columns(4)
        for col, (symbol, values, unit_) in zip(cols, quantities[start:start + 4]):
            with col:
                st.plotly_chart(collection_strip_figure(cells, symbol, values, unit_),
                                key=f"pw_cells_strip_{symbol}", **STRETCH)

    stats = []
    for symbol, values, unit_ in quantities:
        v = np.array([x for x in values if np.isfinite(x)])
        if not v.size:
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        stats.append({"quantity": symbol, "unit": unit_, "n": int(v.size),
                      "median": f"{med:.4g}", "IQR": f"{q1:.4g}–{q3:.4g}",
                      "mean ± sd": f"{v.mean():.4g} ± {v.std(ddof=1) if v.size > 1 else 0:.2g}"})
    flat_table(pd.DataFrame(stats), align_right=["n", "median", "IQR", "mean ± sd"],
               caption="Across the included cells.")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("📥 Table of all cells (CSV)",
                           data=collection_frame(collection).to_csv(index=False),
                           file_name=f"afm_cells_{datetime.now():%Y%m%d}.csv",
                           mime="text/csv", key="pw_cells_csv", **STRETCH)
    with d2:
        gone = st.multiselect("Remove cells", list(collection), key="pw_cells_remove")
        if gone and st.button("Remove selected cells", key="pw_cells_remove_go"):
            for n in gone:
                collection.pop(n, None)
            st.session_state["pw_collection"] = collection
            rerun_keeping_settings(forget=("pw_cells_remove",))
    with d3:
        if st.button("Clear collection", key="pw_cells_clear", **STRETCH):
            st.session_state["pw_collection"] = {}
            rerun_keeping_settings()


def share_of_force_plot(fit, model, style):
    """
    What the log curve's slope means, drawn: who is carrying the load, where.

    The exponent plot says the curve changes its power law. This says why:
    the share of the total force each element carries, against ε. A slope
    leaving 3 and settling near 3/2 is the shell's share falling and the
    cytoplasm's rising, and here that is the same picture as a line
    crossing.
    """
    if not (fit and fit.get("success")):
        return
    terms = [t for t in ALL_TERMS if t in (fit.get("terms") or ())]
    if not terms:
        return
    lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
    grid = np.linspace(max(lo, 1e-4), hi, 200)
    try:
        basis = model.composition_basis(
            grid, float(fit.get("break_1", 0.15)),
            float(fit.get("break_2", 0.40)),
            fit.get("membrane", "freeze"), fit.get("cyto_start", "break"),
            term_windows=fit.get("term_windows"),
        )
    except Exception:  # pragma: no cover - defensive
        return
    pieces = {}
    for term in terms:
        key = MODULUS_FIELDS[term][0]
        raw = {"tension": "T0", "membrane": "Em", "cortex": "Ecx",
               "interior": "Ei", "nucleus_shell": "Ene", "nucleus": "En"}[term]
        column = np.asarray(basis.get(term), dtype=float)
        if column.size != grid.size:
            continue
        pieces[term] = column * float(fit.get(raw, 0.0) or 0.0)
    total = np.sum(list(pieces.values()), axis=0) if pieces else None
    if total is None or not np.any(total > 0):
        return

    figure = go.Figure()
    for term, force in pieces.items():
        share = 100.0 * np.divide(force, np.maximum(total, 1e-30))
        icon = SHAPE_MARKS.get(term, ("",))[0]
        figure.add_trace(go.Scatter(
            x=grid, y=share, mode="lines", stackgroup="one",
            name=f"{icon} {plain_name(term).lower()}",
            hovertemplate="ε = %{x:.3f}<br>%{y:.0f} % of the force"
                          "<extra>" + plain_name(term).lower() + "</extra>",
        ))
    for value, name in ((fit.get("break_1"), "ε₁"),
                        (fit.get("break_2"), "ε₂")):
        if value is not None and lo < float(value) < hi:
            figure.add_vline(
                x=float(value), line=dict(color="#333333", width=1.5,
                                          dash="dot"),
                annotation_text=name, annotation_position="top",
            )
    figure.update_layout(
        height=max(260, int(style.height * 0.5)),
        margin=dict(l=60, r=20, t=40, b=50),
        title="Who is carrying the force, and where",
        yaxis=dict(title="share of the total force (%)", range=[0, 100]),
        xaxis=dict(title="Relative deformation, ε"),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(figure, key="share_of_force", **STRETCH)
    st.caption(
        "Read the two together: the exponent above is the force-weighted "
        "average of the element exponents, and this is the weighting. Where "
        "one band fills the plot the measured slope is that element's own "
        "power; where two bands share it, the slope sits between theirs."
    )


def stage_rows(fit):
    """
    The stretches the boundaries make, and who is carrying load in each.

    One place decides this: the bands drawn behind the curve, the algebra
    printed under the log plot and the sentence describing each stretch all
    come from here, so a band on the figure cannot disagree with the
    equation under it. It used to be hard-coded as membrane, then
    cytoskeleton, then cytoskeleton plus nucleus, which stopped being true
    the moment an element was given a range of its own.

    Returns [(from, to, rising, held)] where ``rising`` are the elements
    still taking more load over that stretch and ``held`` those carrying
    what they already reached.
    """
    if not (fit and fit.get("success")):
        return []
    terms = [t for t in ALL_TERMS if t in (fit.get("terms") or ())]
    if not terms:
        return []
    lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
    windows = fit.get("term_windows") or {}

    def support(term):
        window = windows.get(term)
        if window:
            return float(window[0]), float(window[1])
        return lo, hi

    edges = {round(lo, 6), round(hi, 6)}
    for term in terms:
        a, b = support(term)
        for value in (a, b):
            if lo < value < hi:
                edges.add(round(float(value), 6))
    rows = []
    for a, b in zip(sorted(edges)[:-1], sorted(edges)[1:]):
        if b - a < 1e-6:
            continue
        middle = 0.5 * (a + b)
        rising, held = [], []
        for term in terms:
            start, stop = support(term)
            if middle < start:
                continue
            (rising if middle < stop else held).append(term)
        rows.append((a, b, rising, held))
    return rows


def stage_bands(fit):
    """The stretches as bands for the plot, labelled by who carries them."""
    bands = []
    for index, (a, b, rising, held) in enumerate(stage_rows(fit)):
        carrying = rising or held
        label = " + ".join(plain_name(term).lower() for term in carrying)
        if held and rising:
            label += " (" + ", ".join(
                plain_name(term).lower() for term in held
            ) + " holding)"
        bands.append({
            "range": (a, b),
            "label": label or "nothing carrying",
            "color": STAGE_COLORS[index % len(STAGE_COLORS)],
        })
    return bands


def stage_algebra(fit):
    """
    Each stretch of the squash, written as the sum of terms carrying it.

    The boundaries are where the number of terms changes, so the useful
    description of a boundary is the algebra either side of it: one term or
    two, which two, and what exponent their sum therefore shows. Said in
    words it is a paragraph; said as equations it is three lines that can be
    checked against the slope plotted above.
    """
    if not (fit and fit.get("success")):
        st.caption("Fit the curve and the algebra of each stretch appears here.")
        return
    q = float(fit.get("confinement", 0.0) or 0.0)
    rows = stage_rows(fit)
    if not rows:
        return

    lines = []
    for a, b, rising, held in rows:
        pieces = []
        for term in rising:
            a_tex, e_tex = EQUATION_TERMS[term][2], EQUATION_TERMS[term][3]
            pieces.append(f"{a_tex} {e_tex}\\," + _basis_latex(term, fit))
        for term in held:
            a_tex, e_tex = EQUATION_TERMS[term][2], EQUATION_TERMS[term][3]
            pieces.append(
                r"\underbrace{" + f"{a_tex} {e_tex}" + r"\,\mathrm{const}}"
                r"_{\text{held}}"
            )
        body = " + ".join(pieces) if pieces else r"0"
        if q:
            body = r"(1-\varepsilon)^{-" + f"{q:g}" + r"}\left[" + body + r"\right]"
        lines.append(
            f"{a:.3f} \\le \\varepsilon < {b:.3f}:&\\quad F = " + body
        )
    st.latex(r"\begin{aligned}" + r"\\[2pt]".join(lines) + r"\end{aligned}")

    described = []
    for a, b, rising, held in rows:
        n = len(rising)
        names = " and ".join(plain_name(t).lower() for t in rising) or "nothing"
        exponents = sorted({_exponent_of(t, fit) for t in rising})
        if n == 0:
            said = "no element is still taking load"
        elif n == 1:
            said = (f"one term: {names} alone, so the slope here is its own "
                    f"exponent, {exponents[0]:g}")
        else:
            said = (
                f"{n} terms carrying together: {names}. Their exponents are "
                + ", ".join(f"{value:g}" for value in exponents)
                + ", so the measured slope sits between them, at whichever "
                "the force is weighted towards"
            )
        if held:
            said += (
                ", with "
                + " and ".join(plain_name(t).lower() for t in held)
                + " holding what "
                + ("it" if len(held) == 1 else "they")
                + " reached and taking no more"
            )
        described.append(f"**ε {a:.3f} to {b:.3f}** — {said}.")
    st.markdown("\n\n".join(described))
    if q:
        st.caption(
            f"Every stretch is multiplied by the confinement factor "
            f"(1−ε)^−{q:g}, which adds qε/(1−ε) to the slope everywhere and "
            "is why the measured exponent climbs inside a stretch where the "
            "algebra has not changed."
        )


def _exponent_of(term, fit):
    """The power of ε this term rises with."""
    if term == "tension":
        return 1.0
    if term in ("membrane", "nucleus_shell"):
        return 3.0
    return 1.5


# ---------------------------------------------------------------- layers --
#
# What the plot carries beyond the data and the fit is chosen one piece at a
# time, by sending it there from wherever that piece is computed. A figure
# built from a panel of switches is a figure whose contents somebody has to
# remember; a figure built by sending things to it says what is on it,
# because the sending left a row behind.

LAYER_KINDS = ("boundaries", "range", "equation", "moduli", "quality", "note")


def plot_layers():
    """Everything currently sent to the plot, oldest first."""
    return list(st.session_state.get("plot_layers") or [])


def send_to_plot(kind, label, payload):
    """Put one piece on the plot, replacing any earlier piece of that kind."""
    layers = [row for row in plot_layers() if row["kind"] != kind]
    layers.append({"kind": kind, "label": label, "payload": payload})
    st.session_state["plot_layers"] = layers


def send_to_plot_button(kind, label, payload, key, help_text=None):
    """
    A 📤 Send to plot button for one piece of the results.

    Every layer is live: what it draws is recomputed from the page each
    time the figure is built, so the list under the figure is a history of
    what was sent and never a set of stale numbers. Sending is therefore a
    yes/no, and the button says which it currently is.
    """
    on_plot = any(row["kind"] == kind for row in plot_layers())
    if on_plot:
        st.button("✅ On the plot", key=f"send_{key}", disabled=True,
                  help="It follows the numbers on this page, so it is "
                       "already up to date. Take it off under the figure.",
                  **STRETCH)
        return
    if st.button("📤 Send to plot", key=f"send_{key}", help=help_text,
                 **STRETCH):
        send_to_plot(kind, label, payload)
        rerun_keeping_settings()


def fit_id(fit):
    """
    A short fingerprint of one fit, the same wherever that fit is shown.

    Printed on the curve, above the results and in the block copied into a
    spreadsheet, so three places that should be showing one fit can be seen
    to be: if the fingerprints match, the numbers are the same numbers.
    """
    import hashlib
    if not (fit and fit.get("success")):
        return ""
    keys = ("coupling", "break_1", "break_2", "epsilon_range", "r_squared",
            "Em", "Ei", "En", "T0", "Ene", "Ecx", "confinement", "terms",
            "term_windows", "boundaries_pct", "coefficients")
    text = repr([(k, fit.get(k)) for k in keys])
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:6]


def fit_edges(fit):
    """
    Every place where an element of the fit starts or stops taking load.

    The same edges as the bands behind the curve (both come from
    stage_rows), each named ε₁ or ε₂ where it is one, so a line on the plot
    and the start of a component curve are one number. (value, name).
    """
    pw = (fit or {}).get("piecewise") if fit else None
    if pw and pw.get("boundaries_pct"):
        # Regime by regime, the boundaries are ε₁, ε₂, ε₃ and nothing else:
        # the three lines on its curve.
        b = pw["boundaries_pct"]
        return [(float(v) / 100.0, name) for v, name in zip(b[1:4], EPS_NAMES)]
    rows = stage_rows(fit)
    if not rows:
        return []
    lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
    edges = sorted({round(float(a), 6) for a, _b, _r, _h in rows}
                   | {round(float(b), 6) for _a, b, _r, _h in rows})
    named = []
    for value in edges:
        if not (lo + 1e-6 < value < hi - 1e-6):
            continue
        name = None
        for key, label in (("break_1", "ε₁"), ("break_2", "ε₂")):
            if fit.get(key) is not None and abs(float(fit[key]) - value) < 5e-4:
                name = label
        named.append([value, name])
    # Anything else the components start at is the next ε in the sequence,
    # named rather than left as a bare "ε": with one component to a stretch
    # there is a third and a fourth boundary and they have names.
    spare = [n for n in EPS_NAMES if n not in {row[1] for row in named}]
    for row in named:
        if row[1] is None:
            row[1] = spare.pop(0) if spare else "ε"
    return [(value, name) for value, name in named]


def decorate_curve_figure(figure, style, fit=None, epsilon=None, bands=(),
                          rupture_eps=None, boundaries=True, show_range=False,
                          extra_tags=(), artefact=True):
    """
    Everything drawn on the curve besides the data and the fit, placed so
    that nothing sits on anything else.

    Lines (the end of the contact artefact, the boundaries, the end of the
    fitted range, a rupture) are tagged above the plot on as many rows as
    they need. The bands behind the curve are named in the legend, and the
    legend is under the plot, so neither is written over the data. Every
    line comes from ``fit``, the fit the figure is drawing, never from the
    page's controls, so a line cannot mark a boundary the curve was not
    fitted with.
    """
    log_mode = bool(getattr(style, "log_scale", False))
    eps = np.asarray(epsilon if epsilon is not None else [], dtype=float)
    eps = eps[np.isfinite(eps)]
    x_lo = float(style.x_range[0]) if style.x_range else (
        min(0.0, float(eps.min())) if eps.size else 0.0)
    x_hi = float(style.x_range[1]) if style.x_range else (
        float(eps.max()) if eps.size else 1.0)
    span = max(x_hi - x_lo, 1e-6) * (1.0 if style.x_range else 1.04)
    x_max = x_hi + (0.0 if style.x_range else 0.02 * span)
    tags = []
    legend_extra = []
    ok = bool(fit and fit.get("success"))
    lo = hi = None
    if ok:
        lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
    if not log_mode:
        for band in bands or ():
            a, b = (float(v) for v in band["range"])
            colour = band.get("color", "#2ca02c")
            figure.add_vrect(x0=a, x1=b, fillcolor=colour,
                             opacity=band.get("opacity", 0.12),
                             layer="below", line_width=0)
            name = f"▮ {band.get('label', 'stretch')} · ε {a:.3f}–{b:.3f}"
            legend_extra.append(name)
            figure.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=name,
                marker={"symbol": "square", "size": 13, "color": colour,
                        "opacity": 0.45},
                hoverinfo="skip",
            ))
        if show_range and ok:
            figure.add_vrect(x0=lo, x1=hi, layer="below", line_width=0,
                             fillcolor="#7f7f7f", opacity=0.07)
            name = f"▮ fitted range · ε {lo:.3f}–{hi:.3f}"
            legend_extra.append(name)
            figure.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=name,
                marker={"symbol": "square", "size": 13, "color": "#7f7f7f",
                        "opacity": 0.35},
                hoverinfo="skip",
            ))
        # The first few percent are the probe settling onto the cell: drawn
        # as a stretch of its own, and fitted all the same.
        edge = ARTEFACT_PCT / 100.0
        if artefact and x_hi > edge:
            figure.add_vrect(x0=max(x_lo, 0.0), x1=edge, layer="below",
                             line_width=0, fillcolor=ARTEFACT_FILL)
            tags.append({"x": edge, "text": f"contact artefact ≤ {edge:.2f}",
                         "kind": "artefact"})
        if ok and boundaries:
            for value, name in fit_edges(fit):
                tags.append({"x": value, "text": f"{name} = {value:.3f}",
                             "bold": name, "kind": "boundary"})
        if ok and hi is not None and eps.size and hi < float(eps.max()) - 1e-3:
            tags.append({"x": hi, "text": f"ε_max = {hi:.3f}", "kind": "end"})
        if rupture_eps is not None and getattr(style, "show_rupture_marker", True):
            tags.append({"x": float(rupture_eps),
                         "text": f"rupture {float(rupture_eps):.3f}",
                         "kind": "rupture"})
        tags.extend(extra_tags or ())
    rows = add_tag_lines(figure, tags, span, x_max) if tags else 0
    finish_legend_below(figure, style, rows, legend_extra)
    return rows


def finish_legend_below(figure, style, tag_rows=0, labels=()):
    """Title at the top, tags under it, the legend under the axis."""
    base_top = max(70, int(style.title_size * 3))
    bottom = 50 + int(2.4 * style.axis_title_size)
    names = [getattr(t, "name", None) for t in getattr(figure, "data", ())]
    names = [n for n in names if isinstance(n, str) and n] or list(labels)
    rows = legend_rows(names) if getattr(style, "show_legend", True) else 0
    figure.update_layout(
        title={"y": 0.99, "yref": "container", "yanchor": "top"},
        margin={"t": base_top + TAG_ROW_PX * tag_rows,
                "b": bottom + 26 * rows + 8},
        legend={"orientation": "h", "yref": "container", "y": 0.005,
                "yanchor": "bottom", "xanchor": "left", "x": 0.0,
                "bgcolor": "rgba(255,255,255,0)", "borderwidth": 0},
        height=int(style.height) + TAG_ROW_PX * tag_rows + 26 * rows,
    )


def apply_plot_layers(figure, style, fit=None):
    """
    Draw whatever has been sent to the plot onto the figure.

    Done here rather than inside the figure builder because a layer is a
    decision made on the page, not a property of the curve: the builder
    draws the measurement, this draws what somebody chose to say about it.
    Every box is written from ``fit``, the fit the figure is drawing, so a
    number on the curve is the number in the results beside it.
    """
    boxes = []
    if fit is None:
        fit = st.session_state.get("_last_fit")
    unit = getattr(style, "force_unit", "nN")
    for row in plot_layers():
        kind, payload = row["kind"], dict(row["payload"] or {})
        if kind in ("boundaries", "range"):
            # Drawn by decorate_curve_figure, from the same fit.
            continue
        if kind not in ("note",):
            # A results box: rebuilt from the fit on the page, so a number
            # on the figure is never a number the page has moved on from.
            payload = {"text": row_text(payload.get("key", kind), fit, unit)}
        text = str(payload.get("text", "")).strip()
        if text:
            boxes.append(text)
    # Text boxes stack down the left, inside the axes, where a rising curve
    # leaves the plot empty; the legend is under the axis, out of their way.
    for index, text in enumerate(boxes):
        figure.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98 - 0.16 * index,
            xanchor="left", yanchor="top", text=text, showarrow=False,
            align="left", font=dict(size=max(10, style.tick_size - 7)),
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#8c8c8c",
            borderwidth=1, borderpad=4,
        )
    return figure


def layer_on(kind):
    """Whether a live layer of this kind is on the plot."""
    return any(row["kind"] == kind for row in plot_layers())


LEGACY_EPS_MODES = {
    "as_set": "As set on the board",
    "estimate": "📈 Estimate from the log curve",
}


def layer_checkbox(kind, label, key):
    """A live plot layer as a switch on the board rather than a button."""
    on = any(row["kind"] == kind for row in plot_layers())
    st.session_state[key] = on

    def _toggle():
        layers = [row for row in plot_layers() if row["kind"] != kind]
        if st.session_state.get(key):
            layers.append({"kind": kind, "label": kind,
                           "payload": live_payload(kind)})
        st.session_state["plot_layers"] = layers

    st.checkbox(label, key=key, on_change=_toggle,
                help="Drawn from the fit on the graph, and kept up to date.")


def plot_boxes_control(fit):
    """Which results are written in boxes on the graph: a switch per row."""
    rows = result_rows(fit)
    options = ["equation"] + [row[0] for row in rows]
    labels = {"equation": "The fitted equation",
              **{row[0]: row[1] for row in rows}}
    current = [row["kind"] if row["kind"] == "equation"
               else (row.get("payload") or {}).get("key", row["kind"])
               for row in plot_layers()
               if row["kind"] not in ("boundaries", "range", "note")]
    st.session_state["plot_boxes"] = [k for k in current if k in options]

    def _changed():
        keep = [row for row in plot_layers()
                if row["kind"] in ("boundaries", "range", "note")]
        for key in st.session_state.get("plot_boxes") or []:
            kind = "equation" if key == "equation" else key
            keep.append({"kind": kind, "label": labels.get(key, key),
                         "payload": {"key": key}})
        st.session_state["plot_layers"] = keep

    st.multiselect("Write on the graph", options, format_func=labels.get,
                   key="plot_boxes", on_change=_changed,
                   help="Each is a box on the graph, written from the fit "
                   "shown and kept up to date.")


def plot_layer_table():
    """What is on the plot, with a way to take each piece off again."""
    layers = plot_layers()
    st.markdown("**On the plot**")
    st.caption(
        "The measured points and the fitted curve are always drawn. "
        "Everything else here was sent from somewhere on the page, and "
        "comes off again with ✕."
    )
    if not layers:
        st.caption("Nothing else sent yet.")
        return
    for index, row in enumerate(layers):
        left, right = st.columns([5, 1])
        label = live_label_for(row)
        left.markdown(label)
        if right.button("✕", key=f"drop_layer_{index}",
                        help="Take this off the plot"):
            remaining = plot_layers()
            del remaining[index]
            st.session_state["plot_layers"] = remaining
            rerun_keeping_settings()
    if st.button("Clear the plot", key="clear_plot_layers", **STRETCH):
        st.session_state["plot_layers"] = []
        rerun_keeping_settings()


# What each element is, as a shape: a balloon is a thin shell around
# something that does not compress, and its force rises as ε³; a spring is a
# material squeezed between two plates, a Hertzian contact, rising as ε³ᐟ².
# The mark is the same in the panel, the component list and the plot legend,
# so "the spring" is one claim everywhere.
SHAPE_MARKS = {
    "tension": ("🪢", "taut network", "ε"),
    "membrane": ("🎈", "balloon", "ε³"),
    "cortex": ("🕸️", "spring", "ε³ᐟ²"),
    "interior": ("🕸️", "spring", "ε³ᐟ²"),
    "nucleus_shell": ("🎈", "balloon", "⟨ε−ε₂⟩³"),
    "nucleus": ("🕸️", "spring", "⟨ε−ε₂⟩³ᐟ²"),
}


def shape_mark(term):
    """The balloon-or-spring mark and law for one element, as one line."""
    icon, kind, law = SHAPE_MARKS.get(term, ("", "", "ε"))
    return f"{icon} {kind} · {law}"


def share_of_load_maths():
    """
    How the elements share the load, as algebra rather than as adjectives.

    Two ways of fitting the same four components over the same four
    stretches, one unknown at a time, and they are different equations
    rather than different wordings. What separates them is whether a
    component that has already been fitted goes on carrying load into the
    stretches after it.
    """
    st.caption("① Carried forward — each modulus fitted on its own stretch, "
               "then held while the next is fitted, every stretch starting "
               "from the force the one before it ended on:")
    st.latex(SHARING_MATHS[PW_SHARE])
    st.latex(r"\hat E_1 \;\text{on}\; [0,\varepsilon_1);\quad "
             r"\hat E_2 \;\text{on}\; [\varepsilon_1,\varepsilon_2)"
             r"\;\text{with}\; E_1 = \hat E_1 \;\text{fixed};\quad "
             r"\hat E_3 \;\text{on}\; [\varepsilon_2,\varepsilon_3)"
             r"\;\text{with}\; E_1, E_2\;\text{fixed};\;\ldots")
    st.caption("② Each on its own stretch — every component fitted inside "
               "its own range and acting nowhere else:")
    st.latex(SHARING_MATHS[OWN_STRETCH])
    marks = [
        f"- {plain_name(term)} — {shape_mark(term)}"
        for term in terms_for(st.session_state.get("cell_type"))
    ]
    st.markdown(
        "with p = 3 for a 🎈 balloon, a thin shell around something that "
        "does not compress, and p = 3/2 for a 🕸️ spring, a material "
        "squeezed between two plates. In this cell:\n\n" + "\n".join(marks)
    )


def result_rows(fit):
    """
    Every number the fit produced, one per row, with its uncertainty.

    A row is (key, label, value string, ± string, interval string). The key
    is what a "send to plot" button stores, so a row sent to the figure is
    recomputed from the current fit every time it is drawn rather than
    frozen at the moment somebody pressed it.
    """
    if not (fit and fit.get("success")):
        return []
    rows = []
    here = terms_for(st.session_state.get("cell_type"))
    fitted_terms = set(fit.get("terms") or ())
    for term in ALL_TERMS:
        if term not in here:
            continue
        key, unit_name, std_key = MODULUS_FIELDS[term]
        label = f"{TERM_SYMBOLS.get(term, term)} {plain_name(term).lower()}"
        if term not in fitted_terms:
            rows.append((f"modulus_{term}", label, "not in this model",
                         "—", "—", "not in this model"))
            continue
        try:
            value = float(fit.get(key, 0.0))
        except (TypeError, ValueError):
            value = float("nan")
        try:
            error = float(fit.get(std_key, float("nan")))
        except (TypeError, ValueError):
            error = float("nan")
        rows.append((
            f"modulus_{term}",
            label,
            # The value with its ± beside it, the way it is quoted.
            (f"{value:.4g} ± {error:.3g} {unit_name}" if np.isfinite(error)
             else f"{value:.4g} {unit_name}"),
            (f"± {error:.3g} {unit_name}" if np.isfinite(error) else "—"),
            (f"{max(value - 1.96 * error, 0.0):.4g} to "
             f"{value + 1.96 * error:.4g} {unit_name}"
             if np.isfinite(error) else "—"),
            element_support(term, fit),
        ))
    if fit.get("piecewise"):
        # Anything the regime-by-regime sharing reports on top of the four
        # components every way of sharing has.
        for term, symbol, key, unit_name in EXTRA_PIECEWISE_ROWS:
            try:
                value = float(fit.get(key, float("nan")))
                error = float(fit.get(f"{key}_std", float("nan")))
            except (TypeError, ValueError):
                value = error = float("nan")
            rows.append((
                f"modulus_{term}",
                f"{symbol} {EXTRA_NAMES[term].lower()}",
                ("—" if not np.isfinite(value) else
                 f"{value:.4g} ± {error:.3g} {unit_name}" if np.isfinite(error)
                 else f"{value:.4g} {unit_name}"),
                (f"± {error:.3g} {unit_name}" if np.isfinite(error) else "—"),
                (f"{max(value - 1.96 * error, 0.0):.4g} to "
                 f"{value + 1.96 * error:.4g} {unit_name}"
                 if np.isfinite(error) and np.isfinite(value) else "—"),
                element_support(term, fit),
            ))
    chi = fit.get("chi_squared_reduced", float("nan"))
    lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
    rows.append((
        "quality", "Goodness of fit",
        f"R² = {float(fit.get('r_squared', float('nan'))):.5f}",
        (f"χ²/dof = {float(chi):.3g}" if np.isfinite(chi) else "—"),
        f"RMSE = {float(fit.get('rmse', float('nan'))):.4g} N",
        f"{int(fit.get('n_points', 0))} points over {eps_text(lo, fit)} to "
        f"{eps_text(hi, fit, bare=True)}",
    ))
    # The boundaries this fit has: the lines on the curve and the ε columns
    # of the copied row, never a boundary the model does not use.
    edges = {name: value for value, name in fit_edges(fit)}
    if edges:
        def said(name):
            if name not in edges:
                return "—"
            return f"{name} = {eps_text(edges[name], fit, bare=True)}"
        third = (said("ε₃") if "ε₃" in edges else
                 (f"q = {float(fit.get('confinement', 0.0) or 0.0):g}"
                  if fit.get("confinement") else "—"))
        rows.append((
            "boundaries", "Boundaries", said("ε₁"), said("ε₂"), third,
            "where the elements take over from one another",
        ))
    return rows


# The rows only the regime-by-regime sharing has, after the four it shares
# with every other way: (support key, symbol, fit field, unit).
EXTRA_PIECEWISE_ROWS = ()
EXTRA_NAMES = {
    "perinuclear": "Perinuclear cytoskeleton",
    "lamina": "Nuclear lamina (lump)",
}


def _pm(value, error, unit="", digits=4):
    """value ± error, both in the same unit, as the page quotes them."""
    if value is None or not np.isfinite(float(value)):
        return "—"
    text = f"{float(value):.{digits}g}"
    if error is not None and np.isfinite(float(error)):
        text += f" ± {float(error):.3g}"
    return text + (f" {unit}" if unit else "")


def fit_explainer(fit, result=None):
    """
    How the fit on the page was done, for an undergraduate, with this
    cell's own numbers in every step.

    The same eight steps whichever way the load is shared: what was
    measured, the model, why the powers are 3 and 3/2, why the fit is
    linear, least squares, the uncertainty, from stiffness to modulus, and
    how good the fit is. The piecewise sharing adds the regime-by-regime
    steps and where ε comes from.
    """
    if not (fit and fit.get("success")):
        st.caption("Fit the curve and the working appears here, with its numbers.")
        return
    piecewise = bool(fit.get("piecewise")) and result is not None
    n = int(fit.get("n_points", 0))
    h0 = float(st.session_state.get("cell_height_um", float("nan")))
    lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))

    st.markdown("#### 1 · What was measured")
    st.markdown(
        "The probe squeezes the cell by a distance $\\delta$. Dividing by the "
        f"cell's height $h_0 = {h0:.3g}$ µm gives the **relative deformation** "
        "$\\varepsilon = \\delta / h_0$"
        + (" (written as a percentage, $x = 100\\,\\varepsilon$)" if piecewise else "")
        + f". The curve is $n = {n}$ pairs $(\\varepsilon_i, F_i)$, from "
        f"{eps_text(lo, fit)} to {eps_text(hi, fit, bare=True)}."
    )

    st.markdown("#### 2 · The model: a sum of springs")
    st.markdown(
        "The cell is treated as a few parts pushing back at once. Each part "
        "has a **shape** $\\phi_j$ (how its force grows with the squeeze, "
        "fixed by physics) and a **stiffness** $\\theta_j$ (how strong it is, "
        "the unknown the fit finds). The force is their sum:"
    )
    st.latex(r"\hat F(x) \;=\; \sum_j \theta_j\,\phi_j(x)")
    if piecewise:
        b = fit["piecewise"]["boundaries_pct"]
        st.latex(
            r"\hat F(x) = C_0 + k\,\min(x,\varepsilon_1)"
            r" + K_m\,[x-\varepsilon_1]_+^{3} + K_c\,[x-\varepsilon_1]_+^{3/2}"
            r" + K_{ne}\,[x-\varepsilon_2]_+^{3} + K_n\,[x-\varepsilon_3]_+^{3/2}"
        )
        st.markdown(
            "$[y]_+ = \\max(y, 0)$: a part adds nothing until the squeeze "
            "reaches the boundary where it starts, "
            f"$\\varepsilon_1 = {b[1]:.1f}\\,\\%$ (membrane and cytoskeleton), "
            f"$\\varepsilon_2 = {b[2]:.1f}\\,\\%$ (nuclear envelope), "
            f"$\\varepsilon_3 = {b[3]:.1f}\\,\\%$ (inside the nucleus). "
            "Before $\\varepsilon_1$ only the contact line $C_0 + kx$ acts: the "
            "probe settling onto the cell."
        )
    else:
        pieces = equation_pieces(fit)
        if pieces:
            st.latex(r"\hat F(\varepsilon) = " + " + ".join(
                f"{p['symbols']}\\,{p['basis']}" for p in pieces))
        st.markdown(
            "$\\langle y \\rangle = \\max(y, 0)$: a part adds nothing until "
            "the squeeze reaches the boundary where it starts. Each $a$ is a "
            "number fixed by the cell's geometry, so the unknowns are the "
            "Young's moduli $E$ themselves."
        )

    st.markdown("#### 3 · Why the powers are 3 and 3/2")
    st.markdown(
        "**A solid ball pressed by a flat plate** (the cytoskeleton, the inside "
        "of the nucleus) follows Hertz's law: the contact area grows as it is "
        "pressed, and"
    )
    st.latex(r"F = \tfrac{4}{3}\,E^*\sqrt{R}\;\delta^{3/2}"
             r"\quad\Longrightarrow\quad F \propto \varepsilon^{3/2}")
    st.markdown(
        "**A thin skin around a filled cell** (the membrane, the nuclear "
        "envelope) must stretch as the cell is flattened. The stretch of its "
        "area grows as $\\varepsilon^2$, the tension it carries grows with the "
        "stretch, and the force on the plate is tension times the curvature "
        "it pushes through, which gives one more power of $\\varepsilon$:"
    )
    st.latex(r"F \propto E\,h\,\varepsilon^{2}\cdot\varepsilon"
             r"\quad\Longrightarrow\quad F \propto \varepsilon^{3}")
    st.markdown(
        "So on log–log axes each part is a straight line: slope 3 for a skin, "
        "3/2 for a solid (see the 📈 Log curve and boundaries tab)."
    )

    st.markdown("#### 4 · Why the fit is a straight-line problem")
    st.markdown(
        "Once the boundaries are fixed, the shapes $\\phi_j$ are known "
        "numbers at every measured point. Only the stiffnesses are unknown, "
        "and they enter **linearly**, exactly as the slope and intercept of a "
        "straight line do. Stacking one equation per point gives a matrix "
        "problem:"
    )
    st.latex(r"\begin{pmatrix}F_1\\ \vdots\\ F_n\end{pmatrix} \approx "
             r"\underbrace{\begin{pmatrix}\phi_1(x_1) & \cdots & \phi_m(x_1)\\"
             r"\vdots & & \vdots\\ \phi_1(x_n) & \cdots & \phi_m(x_n)\end{pmatrix}}"
             r"_{X}\begin{pmatrix}\theta_1\\ \vdots\\ \theta_m\end{pmatrix},"
             r"\qquad \mathbf F \approx X\boldsymbol\theta")

    st.markdown("#### 5 · Least squares: the best stiffnesses")
    st.markdown(
        "The fit chooses the $\\theta_j$ that make the total squared miss "
        "between the curve and the model as small as possible:"
    )
    st.latex(r"S(\boldsymbol\theta) = \sum_{i=1}^{n}\bigl(F_i - \hat F(x_i)\bigr)^2"
             r" = \lVert \mathbf F - X\boldsymbol\theta\rVert^2")
    st.markdown(
        "Setting $\\partial S/\\partial\\theta_j = 0$ for every $j$ gives the "
        "**normal equations**, whose solution is the best fit:"
    )
    st.latex(r"X^{\mathsf T}X\,\hat{\boldsymbol\theta} = X^{\mathsf T}\mathbf F"
             r"\quad\Longrightarrow\quad \hat{\boldsymbol\theta} = "
             r"(X^{\mathsf T}X)^{-1}X^{\mathsf T}\mathbf F")
    st.markdown(
        "A stiffness cannot be negative, so the solver looks for the best "
        "$\\boldsymbol\\theta$ with every $\\theta_j \\ge 0$ (bounded least "
        "squares). When a part is not needed it lands on 0, and the page "
        "says so."
    )

    if piecewise:
        st.markdown("#### 5b · Regime by regime")
        st.markdown(
            "The piecewise way of sharing the load fits the curve one stretch "
            "at a time, from the contact inwards. Regime $k$ only fits the "
            "parts that **start** in it; the parts that started earlier keep "
            "the stiffness already found ($C_k$), and the regime starts from "
            "the force the one before it ended on, so the curve has no jumps "
            "($C^0$ continuity):"
        )
        st.latex(r"\hat F_k(x) = \hat F_{k-1}(\varepsilon_{k-1}) + C_k(x) + "
                 r"\sum_{j\,\in\,k}\theta_j\,\phi_j(x),\qquad "
                 r"x\in[\varepsilon_{k-1},\varepsilon_k)")
        flat_table(pd.DataFrame([
            {"regime": r["key"],
             "x (%)": f"{r['domain_pct'][0]:.1f}–{r['domain_pct'][1]:.1f}",
             "points": r["n_points"],
             "fitted here": ", ".join(
                 next((c[1].split(" ", 1)[1].lower() for c in PW_COMPONENTS
                       if c[0] == n_), n_)
                 for n_ in r["params"]
                 if n_ not in ("C0", "k_align")) or "C₀ alone",
             "R²_k": (f"{r['r_squared']:.4f}" if np.isfinite(r["r_squared"]) else "—")}
            for r in result["regimes"]
        ]), align_right=["points", "R²_k"])
        st.markdown(
            "**Where ε₁, ε₂, ε₃ come from.** For a single power law "
            "$F = a x^{p}$, $\\ln F = \\ln a + p\\ln x$: the slope on log–log "
            "axes *is* the power. The local slope"
        )
        st.latex(r"p(x) = \frac{d\ln F}{d\ln x}")
        st.markdown(
            "jumps where a new part joins in, so the boundaries are placed where "
            "$p$ bends, inside the C2C12 bands, and then every nearby "
            "placement is fitted and the one with the smallest $S$ is kept. "
            f"Here: **{fit['piecewise'].get('boundary_source', '')}**."
        )

    st.markdown("#### 6 · The ± : how sure is each number?")
    st.markdown(
        "The misses $r_i = F_i - \\hat F(x_i)$ that are left measure the noise. "
        "With $m$ fitted numbers,"
    )
    st.latex(r"\hat\sigma^2 = \frac{S_{\min}}{n-m},\qquad "
             r"\operatorname{Cov}(\hat{\boldsymbol\theta}) = "
             r"\hat\sigma^2\,(X^{\mathsf T}X)^{-1},\qquad "
             r"\mathrm{SE}(\hat\theta_j) = \sqrt{\operatorname{Cov}_{jj}}")
    st.markdown(
        "Every value on this page is quoted as **best fit ± SE**. About 95 % "
        "of the time the true value lies within $\\hat\\theta \\pm 1.96\\,"
        "\\mathrm{SE}$, which is the interval in the results table. A large SE "
        "means the curve cannot tell that part apart from the others."
    )

    st.markdown("#### 7 · From stiffness to Young's modulus")
    if piecewise:
        st.markdown(
            "Each stiffness is the part's Young's modulus times a number set "
            "by the geometry, $A_j$ (in N/Pa). Because $x$ is in % and the "
            "formulas use $\\varepsilon = x/100$, a power $p$ brings a factor "
            "$100^{p}$:"
        )
        st.latex(r"E_j = \frac{100^{\,p_j}\,K_j}{A_j},\qquad "
                 r"\mathrm{SE}(E_j) = \frac{100^{\,p_j}\,\mathrm{SE}(K_j)}{A_j}")
        st.latex(r"A_{\text{skin}} = \frac{2\pi h\,R}{1-\nu},\qquad "
                 r"A_{\text{Hertz}} = \frac{\sqrt2\,R^2}{3(1-\nu^2)}\,c(R)")
        moduli = result.get("moduli") or {}
        rows = []
        for key_, label, symbol, _c, _law in PW_COMPONENTS:
            m = moduli.get(key_)
            if not m:
                continue
            unit, scale = DISPLAY_UNIT.get(m["symbol"], ("kPa", 1e3))
            rows.append({
                "part": label.split(" ", 1)[1],
                "p": f"{m['power']:g}",
                "K ± SE (N/%ᵖ)": _pm(m.get("K"), m.get("K_se"), digits=4),
                "A (N/Pa)": (f"{m['prefactor_N_per_Pa']:.4g}"
                             if np.isfinite(m.get("prefactor_N_per_Pa", float("nan")))
                             else "—"),
                "E ± SE": (modulus_display(m["symbol"], m["E_Pa"], m["E_se_Pa"])
                           if key_ not in (fit["piecewise"].get("components_off") or ())
                           else "off"),
            })
        if rows:
            flat_table(pd.DataFrame(rows),
                       align_right=["p", "K ± SE (N/%ᵖ)", "A (N/Pa)", "E ± SE"],
                       caption="Every component goes through E = 100ᵖK/A. The "
                       "first stretch, [0, ε₁), has no component at all: it is "
                       "the probe settling onto the cell, and it is fitted by "
                       "the constant C₀, which the curve is measured from.")
    else:
        st.markdown(
            "Here the model is written with the moduli themselves: each term is "
            "$a_k E_k$ times its shape, and $a_k$ (N/Pa) is fixed by the "
            "geometry before the fit. So the fit returns $E_k$ directly, and "
            "its SE with it."
        )
        rows = []
        for piece in equation_pieces(fit):
            key, unit_name, std_key = MODULUS_FIELDS[piece["term"]]
            rows.append({
                "part": piece["name"],
                "shape": SHAPE_MARKS.get(piece["term"], ("", "", ""))[2],
                "a (N/Pa)": f"{piece['prefactor']:.4g}",
                "E ± SE": _pm(fit.get(key), fit.get(std_key), unit_name),
            })
        if rows:
            flat_table(pd.DataFrame(rows), align_right=["a (N/Pa)", "E ± SE"])

    st.markdown("#### 8 · How good is the fit?")
    r2 = float(fit.get("r_squared", float("nan")))
    chi = float(fit.get("chi_squared_reduced", float("nan")))
    st.latex(r"R^2 = 1 - \frac{\sum_i (F_i-\hat F_i)^2}{\sum_i (F_i-\bar F)^2}"
             + (f" = {r2:.5f}" if np.isfinite(r2) else ""))
    st.latex(r"\chi^2_\nu = \frac{1}{n-m}\sum_i\frac{(F_i-\hat F_i)^2}{\sigma^2}"
             + (f" = {chi:.3g}" if np.isfinite(chi) else ""))
    st.markdown(
        "$R^2$ is the share of the curve's variation the model explains (1 is "
        "perfect). $\\chi^2_\\nu$ compares the misses with the noise "
        "$\\sigma$ measured on the curve: about 1 means the model is as close "
        "as the noise allows; much more than 1 means it is missing something "
        "real, even when $R^2$ looks excellent."
    )


def results_tab_metrics(fit):
    """The moduli and R² as tiles, each value with its ± beside it."""
    tiles = [(row[1].split(" ", 1)[0], row[2]) for row in result_rows(fit)
             if row[0].startswith("modulus_") and row[2] != "not in this model"]
    tiles.append(("R²", f"{float(fit.get('r_squared', float('nan'))):.5f}"))
    for start in range(0, len(tiles), 3):
        cols = st.columns(3)
        for col, (label, value) in zip(cols, tiles[start:start + 3]):
            col.metric(label, value)


def row_text(key, fit, unit="nN"):
    """One results row as the text a plot box should carry."""
    if key == "equation":
        return equation_text(fit, unit)
    for row in result_rows(fit):
        if row[0] != key:
            continue
        pieces = [f"<b>{row[1]}</b>", row[2]]
        if row[3] not in ("—", "") and "±" not in row[2]:
            pieces.append(row[3])
        if row[4] not in ("—", ""):
            pieces.append(row[4])
        return "<br>".join(pieces)
    return ""


def fitting_results_rows(fit, style, send=True, compact=False):
    """
    The results, one per row, each with a way onto the figure.

    Tiles put four numbers side by side and ran out of width at the fourth,
    so the uncertainty went grey and small. A row has room for the value,
    the ±, the interval and where it was measured, and room for the button
    that puts it on the plot.
    """
    rows = result_rows(fit)
    if not rows:
        st.caption("Fit the curve and the results appear here.")
        return
    if compact:
        # Beside the graph: one table, the same rows and the same strings.
        def quoted(value, error):
            # Moduli already carry their ± beside the value; the quality
            # and boundary rows put their second number beside the first.
            if "±" in value or error in ("—", ""):
                return value
            return f"{value} · {error}"

        flat_table(
            pd.DataFrame([
                {"": label, "value ± SE": quoted(value, error),
                 "95 % interval": interval}
                for _key, label, value, error, interval, _note in rows
            ]),
            align_right=["value ± SE", "95 % interval"],
            caption="Each value is quoted as best fit ± one standard error, "
                    "SE = √diag[σ̂²(XᵀX)⁻¹]; the interval is ±1.96 SE, cut "
                    "at zero. How these are worked out is under 📘 How the fit "
                    "is done.",
        )
        return
    widths = [1.5, 1.3, 1.2, 1.7, 1.6] + ([0.9] if send else [])
    head = st.columns(widths)
    for column, title in zip(
        head,
        ("", "value", "±", "95 % interval", "where / how", ""),
    ):
        column.markdown(
            f"<span style='font-size:0.8em;opacity:0.7'>{title}</span>",
            unsafe_allow_html=True,
        )
    for key, label, value, error, interval, note in rows:
        c = st.columns(widths)
        c[0].markdown(f"**{label}**")
        c[1].markdown(value)
        c[2].markdown(error)
        c[3].markdown(interval)
        c[4].markdown(
            f"<span style='font-size:0.85em;opacity:0.8'>{note}</span>",
            unsafe_allow_html=True,
        )
        if not send:
            continue
        with c[5]:
            send_to_plot_button(
                key, f"{label} · {value}", {"key": key}, key=f"row_{key}",
                help_text="Writes this row in a box on the curve, and keeps "
                          "it up to date as the fit changes.",
            )
    st.caption(
        "The ± is one standard error from the covariance of the fit, "
        "σ²(XᵀWX)⁻¹"
        + (" of the regime the element was fitted in" if not send else "")
        + ", and the interval is ±1.96σ around the value, cut at "
        "zero because a modulus cannot be negative."
        + (" Every row can go on the figure, and stays current there."
           if send else "")
    )


def moduli_text(fit):
    """The fitted moduli with their uncertainties, as one block of text."""
    lines = []
    for term in ALL_TERMS:
        if term not in (fit.get("terms") or ()):
            continue
        key, unit_name, std_key = MODULUS_FIELDS[term]
        value = fit.get(key)
        error = fit.get(std_key, float("nan"))
        if value is None:
            continue
        line = f"{TERM_SYMBOLS.get(term, term)} = {float(value):.4g}"
        if error is not None and np.isfinite(error):
            line += f" ± {float(error):.2g}"
        lines.append(line + f" {unit_name}")
    return "<br>".join(lines)


def quality_text(fit):
    """R², χ²/dof and the range, as one block of text."""
    chi = fit.get("chi_squared_reduced", float("nan"))
    lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
    parts = [f"R² = {float(fit.get('r_squared', float('nan'))):.5f}"]
    if np.isfinite(chi):
        parts.append(f"χ²/dof = {chi:.3g}")
    parts.append(f"ε {lo:.3f}–{hi:.3f}, {int(fit.get('n_points', 0))} points")
    return "<br>".join(parts)


def equation_text(fit, unit="nN"):
    """The fitted equation with this cell's numbers, as one line of text."""
    pieces = equation_pieces(fit)
    if not pieces:
        return ""
    factor, unit_label = FORCE_UNITS.get(unit, (1e9, "nN"))
    q = float(fit.get("confinement", 0.0) or 0.0)
    shapes = {
        "tension": "ε", "membrane": "ε³", "cortex": "ε³ᐟ²",
        "interior": "⟨ε−ε₁⟩³ᐟ²", "nucleus_shell": "⟨ε−ε₂⟩³",
        "nucleus": "⟨ε−ε₂⟩³ᐟ²",
    }
    body = " + ".join(
        f"{piece['coefficient_N'] * factor:.3g}·{shapes.get(piece['term'], 'ε')}"
        for piece in pieces
    )
    if q:
        body = f"(1−ε)^−{q:g} [ {body} ]"
    return f"F/{unit_label} = {body}"


def live_payload(kind):
    """What a live layer should draw, read from the page as it stands now."""
    if kind == "boundaries":
        return boundaries_payload()
    return {
        "lo": round(float(st.session_state.get("window_start", 0.0)), 4),
        "hi": round(float(st.session_state.get("window_end", 1.0)), 4),
    }


def live_label_for(row):
    """One row of the history table, read from the page as it stands."""
    kind = row["kind"]
    if kind in ("boundaries", "range"):
        return live_label(kind)
    fit = st.session_state.get("_last_fit")
    if kind == "equation":
        return "The fitted equation · " + (equation_text(fit) or "not fitted")
    for key, label, value, _error, _interval, _note in result_rows(fit):
        if key == (row["payload"] or {}).get("key", kind):
            return f"{label} · {value}"
    return row["label"]


def live_label(kind):
    """How a live layer reads in the list of what is on the plot."""
    fit = st.session_state.get("_last_fit")
    if fit and fit.get("success"):
        # What the plot is drawing, which is the fit, not the controls.
        if kind == "boundaries":
            edges = fit_edges(fit)
            return ("Boundaries · " + (", ".join(f"{n} = {v:.3f}" for v, n in edges)
                                       or "none in this model")
                    + " · from the fit shown")
        lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
        return f"Fitted range · ε {lo:.3f} to {hi:.3f} · from the fit shown"
    payload = live_payload(kind)
    if kind == "boundaries":
        return (f"Boundaries · ε₁ = {payload['e1']:.3f}, "
                f"ε₂ = {payload['e2']:.3f} · follows the page")
    return (f"Fitted range · ε {payload['lo']:.3f} to {payload['hi']:.3f}"
            " · follows the page")


def boundaries_payload():
    """The boundaries as they stand, ready to send to the plot."""
    return {
        "e1": round(float(st.session_state["segment_break_1"]), 4),
        "e2": round(float(st.session_state["segment_break_2"]), 4),
    }


def boundaries_label():
    """How a boundary layer reads in the list of what is on the plot."""
    payload = boundaries_payload()
    return (f"Boundaries · ε₁ = {payload['e1']:.3f}, "
            f"ε₂ = {payload['e2']:.3f}")


def where_the_power_law_changes(fit, model, style):
    """
    The curve read on log-log axes, with what changes where written on it.

    This replaces a section of prose about exponents. The exponent is the
    one part of this model a person can check by hand off their own data,
    and a plot of it against ε with the boundaries drawn on says everything
    that prose said, in the place where it can be disagreed with.
    """
    if exponent_profile_figure is None:
        return
    if not (fit and fit.get("success")):
        # No fit yet: the curve's own slope still says everything it says,
        # over the range chosen on the analysis tab, with no boundaries.
        fit = None
        eps_all = np.asarray(model.epsilon, dtype=float)
        lo = float(st.session_state.get("window_start", 0.0))
        hi = float(st.session_state.get("window_end", float(np.nanmax(eps_all))))
        lo, hi = max(lo, 0.0), min(hi, float(np.nanmax(eps_all)))
    else:
        lo, hi = (float(v) for v in fit.get("epsilon_range", (0.0, 1.0)))
    profile = st.session_state.get("_slope_profile")
    if not (profile and len(profile.get("epsilon", []))
            and tuple(profile.get("_range", ())) == (round(lo, 4), round(hi, 4))):
        profile = log_slope_profile(model, lo, hi)
        profile["_range"] = (round(lo, 4), round(hi, 4))
        st.session_state["_slope_profile"] = profile
    eps = np.asarray(profile.get("epsilon", []), dtype=float)
    p = np.asarray(profile.get("exponent", []), dtype=float)
    if eps.size < 4:
        st.caption(
            "The slope of the log curve cannot be measured over enough of "
            "this range to plot: too few points clear of the noise."
        )
        return

    # The boundaries the fit on the analysis tab has: the lines on its
    # curve, whichever way it shares the load.
    named = dict((name, value) for value, name in fit_edges(fit)) if fit else {}
    piecewise = bool(fit and fit.get("piecewise"))
    e1 = named.get("ε₁")
    e2 = named.get("ε₂")
    e3 = named.get("ε₃")
    terms = set((fit or {}).get("terms") or ())
    edges = [lo]
    for value in (e1, e2, e3):
        if value is not None and lo < float(value) < hi:
            edges.append(float(value))
    edges.append(hi)
    edges = sorted(set(round(v, 6) for v in edges))

    def mean_slope(a, b):
        inside = (eps >= a) & (eps <= b) & np.isfinite(p)
        return float(np.mean(p[inside])) if inside.sum() >= 2 else float("nan")

    stages, notes = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        measured = mean_slope(a, b)
        stages.append({
            "from": a, "to": b,
            "label": (f"slope ≈ {measured:.2f}"
                      if np.isfinite(measured) else "slope not measurable"),
        })
    if e1 is not None and lo < float(e1) < hi:
        before, after = mean_slope(lo, float(e1)), mean_slope(float(e1), hi)
        notes.append({
            "epsilon": float(e1),
            "y": 0.9,
            "ay": -46,
            "text": (
                f"ε₁ = {float(e1):.3f}<br>"
                + (f"slope {before:.2f} → {after:.2f}"
                   if np.isfinite(before) and np.isfinite(after)
                   else "the shell hands over")
                + "<br>" + (
                    "membrane and cytoskeleton start" if piecewise else
                    "the shell stops taking more; the cytoplasm carries on"
                    if "interior" in terms else "the outer law ends"
                )
            ),
        })
    if e2 is not None and lo < float(e2) < hi and (
            "nucleus" in terms or "nucleus_shell" in terms):
        before, after = mean_slope(float(e1 or lo), float(e2)), mean_slope(float(e2), hi)
        notes.append({
            "epsilon": float(e2),
            "y": 0.45,
            "ax": 40,
            "ay": -40,
            "text": (
                f"ε₂ = {float(e2):.3f}<br>"
                + (f"slope {before:.2f} → {after:.2f}"
                   if np.isfinite(before) and np.isfinite(after)
                   else "the deep element is met")
                + "<br>" + plain_name(
                    "nucleus_shell" if "nucleus_shell" in terms else "nucleus"
                ).lower() + " first carries load"
            ),
        })

    if e3 is not None and lo < float(e3) < hi:
        before, after = mean_slope(float(e2 or lo), float(e3)), mean_slope(float(e3), hi)
        notes.append({
            "epsilon": float(e3),
            "y": 0.2,
            "ax": 40,
            "ay": -30,
            "text": (
                f"ε₃ = {float(e3):.3f}<br>"
                + (f"slope {before:.2f} → {after:.2f}"
                   if np.isfinite(before) and np.isfinite(after) else "")
                + "<br>the inside of the nucleus carries load"
            ),
        })
    figure = exponent_profile_figure(
        profile, style, e1, e2,
        title="Where the curve changes its power law",
        notes=notes, stages=stages,
    )
    if e3 is not None and lo < float(e3) < hi:
        figure.add_vline(x=float(e3), line=dict(color="#9467bd", width=2,
                                                dash="dot"))
    st.plotly_chart(figure, key="power_law_profile", **STRETCH)
    st.caption(
        "The slope of ln F against ln ε, measured on the data alone with no "
        "fit in it: 3 while a fluid-filled shell is being stretched, 3/2 "
        "for a Hertzian contact, 1 for a network already under tension, and "
        "the force-weighted mean of those wherever more than one is "
        "carrying. Confinement adds qε/(1−ε) to all of them, which is why "
        "the measured slope climbs along the curve rather than sitting on a "
        "constant"
        + (f"; here q = {float(fit.get('confinement', 0.0) or 0.0):g}. "
           if fit and fit.get("confinement") else ". ")
        + ("The vertical lines are the boundaries the fit used, so this is "
           "the one panel where the model can be held to the data by eye."
           if fit else "Fit the curve on the analysis tab and its boundaries "
           "are drawn on this.")
    )


def shown_fit():
    """The fit the analysis tab is showing now, whichever way it shares."""
    if piecewise_on():
        return (st.session_state.get("results") or {}).get("fit")
    return st.session_state.get("_last_fit")


# The straight lines drawn on the log-log plot to hold the curve against.
# Each is F = F(ε_a)·(ε/ε_a)^p: one exponent, pivoting on one point of the
# data, so turning the exponent rotates the line about that point and the
# eye can match it to a stretch of the curve.
LOG_GUIDES = (("log_guide_p1", "#2ca02c"), ("log_guide_p2", "#9467bd"))
GUIDE_NAMES = {3.0: "a stretching shell", 1.5: "a Hertzian contact",
               1.0: "a network under tension"}


def guide_line_name(power):
    """What a line of this slope would mean, when it means something."""
    for value, said in GUIDE_NAMES.items():
        if abs(power - value) < 1e-9:
            return said
    return ""


def log_guide_settings():
    """The guide lines as the page has them set."""
    return {
        "on": bool(st.session_state.get("log_guides", True)),
        "powers": [float(st.session_state.get(key, default))
                   for key, default in zip([k for k, _c in LOG_GUIDES],
                                           (3.0, 1.5))],
        "anchor_pct": float(st.session_state.get("log_guide_anchor_pct", 30.0)),
        "span": float(st.session_state.get("log_guide_span", 0.4)),
    }


def log_guide_controls():
    """
    Turn the straight lines, and say where they pivot.

    The exponent is the only thing a straight line on log-log axes has, so
    it is the only thing to set: p is typed, the line turns about the point
    of the data at ε_a, and its tag on the plot reads the p it is at.
    """
    st.checkbox("Draw straight guide lines on the plot", key="log_guides",
                help="Each is F = F(ε_a)·(ε/ε_a)^p, a straight line of slope "
                     "p through the data at ε_a.")
    st.latex(r"\log F = \log F(\varepsilon_a) + p\,"
             r"\big[\log\varepsilon - \log\varepsilon_a\big]")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.number_input("p₁", min_value=0.1, max_value=6.0, step=0.05,
                        format="%.2f", key="log_guide_p1",
                        help="Exponent of the first line. 3 is a stretching "
                             "shell.")
    with c2:
        st.number_input("p₂", min_value=0.1, max_value=6.0, step=0.05,
                        format="%.2f", key="log_guide_p2",
                        help="Exponent of the second line. 3/2 is a Hertzian "
                             "contact, 1 a network already under tension.")
    with c3:
        st.slider("ε_a (%)", min_value=0.5, max_value=99.0, step=0.5,
                  key="log_guide_anchor_pct",
                  help="Where the lines touch the data. Both turn about this "
                       "point, so slide it along the curve and turn p until a "
                       "line lies on the stretch you are reading.")
    with c4:
        st.slider("half-length (decades)", min_value=0.2, max_value=2.0,
                  step=0.1, format="%.1f", key="log_guide_span",
                  help="How far each line runs either side of ε_a, in decades "
                       "of ε. The plot always shows the whole of both lines.")


def log_log_figure(epsilon, force_N, fitted_N, fit, style, guides=None):
    """
    The curve itself on log-log axes: the data, the fit, the boundaries.

    Every law in the model is a straight line here, of slope 3 or 3/2, so a
    stretch that runs straight is one law carrying the load and a bend is a
    boundary. The boundaries are drawn as lines of the plot's own, in data
    coordinates, so they sit where they belong on a log axis.

    The straight guide lines pivot on one point of the data, ε_a, and each
    carries its exponent as a tag at its end. The force axis is set to hold
    the data **and** the whole of every guide line, so a line turned steep
    is not cut off at the top of the plot.
    """
    x = np.asarray(epsilon, dtype=float)
    y, unit = from_newtons(np.asarray(force_N, dtype=float), style.force_unit)
    keep = (x > 0) & (y > 0) & np.isfinite(y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x[keep], y=y[keep], mode="markers", name="Experimental data",
        marker={"color": style.data_color, "size": max(3, int(style.marker_size * 0.55))},
    ))
    if fitted_N is not None:
        fy, _ = from_newtons(np.asarray(fitted_N, dtype=float), style.force_unit)
        good = (x > 0) & np.isfinite(fy) & (fy > 0)
        fig.add_trace(go.Scatter(
            x=x[good], y=fy[good], mode="lines", name="Fitted curve",
            line={"color": style.fit_color or "#000000", "width": 2.5},
        ))
    if keep.any():
        y_lo, y_hi = float(np.min(y[keep])), float(np.max(y[keep]))
        x_lo, x_hi = float(np.min(x[keep])), float(np.max(x[keep]))
        # Everything drawn, so the axis can be set to hold all of it.
        shown_lo, shown_hi = y_lo, y_hi
        for value, name in (fit_edges(fit) if fit else []):
            if value <= 0:
                continue
            fig.add_trace(go.Scatter(
                x=[value, value], y=[y_lo, y_hi], mode="lines",
                name=f"{name} = {eps_text(value, fit, bare=True)}",
                line={"color": "#222222", "width": 1.5, "dash": "dash"},
                hoverinfo="skip",
            ))
        settings = guides if guides is not None else log_guide_settings()
        if settings.get("on", True):
            # The pivot: a point of the data itself, so the lines are held
            # against the curve rather than floating beside it.
            order = np.argsort(x[keep])
            xs_data, ys_data = x[keep][order], y[keep][order]
            x_a = float(np.clip(settings["anchor_pct"] / 100.0, x_lo, x_hi))
            y_a = float(np.interp(x_a, xs_data, ys_data))
            span = float(settings["span"])
            ends = np.array([x_a * 10.0 ** -span, x_a * 10.0 ** span])
            for power, (_key, colour) in zip(settings["powers"], LOG_GUIDES):
                line_y = y_a * (ends / x_a) ** power
                shown_lo = min(shown_lo, float(np.min(line_y)))
                shown_hi = max(shown_hi, float(np.max(line_y)))
                said = guide_line_name(power)
                fig.add_trace(go.Scatter(
                    x=ends, y=line_y, mode="lines+text",
                    name=f"slope p = {power:g}" + (f" ({said})" if said else ""),
                    text=["", f"p = {power:g}"], textposition="top left",
                    textfont={"color": colour, "size": 12},
                    line={"color": colour, "width": 1.5, "dash": "dot"},
                    hoverinfo="skip",
                ))
            fig.add_trace(go.Scatter(
                x=[x_a], y=[y_a], mode="markers",
                name=f"pivot ε_a = {100.0 * x_a:.1f} %",
                marker={"symbol": "circle-open", "size": 12, "color": "#000000",
                        "line": {"width": 2}},
                hovertemplate=f"the lines turn about ε = {100.0 * x_a:.1f} %"
                              "<extra></extra>",
            ))
            # Room for the tags at the ends of the lines.
            shown_hi *= 1.6
            shown_lo /= 1.6
        fig.update_yaxes(range=[np.log10(max(shown_lo, 1e-30)) - 0.1,
                                np.log10(max(shown_hi, 1e-29)) + 0.1])
    fig.update_layout(
        height=460, template="simple_white",
        margin={"l": 70, "r": 20, "t": 30, "b": 110},
        legend={"orientation": "h", "yref": "container", "y": 0.005,
                "yanchor": "bottom", "x": 0.0, "xanchor": "left"},
        xaxis={"title": "Relative deformation ε (log)", "type": "log"},
        yaxis={"title": f"Force ({unit}, log)", "type": "log"},
    )
    return fig


def piecewise_share_figure(result, style):
    """Who carries the load where: each component's share of F̂, stacked."""
    b = result["boundaries_pct"]
    x = np.linspace(max(float(b[0]), 0.2), float(b[-1]), 400)
    total = predict_piecewise(x / 100.0, result)
    fig = go.Figure()
    off = piecewise_off()
    for name, label, _symbol, colour, _law in PW_COMPONENTS:
        if component_force is None:
            break
        layer = component_force(result, name, x)
        if layer is None or name in off:
            continue
        share = np.where(np.abs(total) > 0, layer / total, np.nan)
        fig.add_trace(go.Scatter(
            x=x, y=100.0 * share, mode="lines", stackgroup="share",
            name=label, line={"color": colour, "width": 0.8},
            fillcolor=_rgba(colour, 0.35),
        ))
    for value, name in zip(b[1:4], EPS_NAMES):
        fig.add_trace(go.Scatter(
            x=[value, value], y=[0, 100], mode="lines", name=f"{name} = {value:.1f} %",
            line={"color": "#222222", "width": 1.2, "dash": "dash"},
            hoverinfo="skip",
        ))
    fig.update_layout(
        height=360, template="simple_white",
        margin={"l": 70, "r": 20, "t": 30, "b": 90},
        legend={"orientation": "h", "yref": "container", "y": 0.005,
                "yanchor": "bottom", "x": 0.0, "xanchor": "left"},
        xaxis={"title": "Relative deformation x (%)"},
        yaxis={"title": "share of F̂ (%)", "range": [0, 100]},
    )
    return fig


def log_slope_profile(model, lo, hi):
    """
    The curve's local power law, ε by ε: d log F / d log ε.

    This is the curve read as a physicist reads it. Each element's law is a
    power of ε, so the slope on log-log axes is the force-weighted mean of
    the exponents in play, and a boundary is a place where that mean moves
    from one law to the next. Measuring it is independent of any fit, which
    is what makes it evidence rather than a restatement of the model.
    """
    if not hasattr(model, "local_exponent"):
        return {"epsilon": np.array([]), "exponent": np.array([])}
    try:
        eps, exponent = model.local_exponent()
    except Exception:  # pragma: no cover - defensive
        return {"epsilon": np.array([]), "exponent": np.array([])}
    eps = np.asarray(eps, dtype=float)
    exponent = np.asarray(exponent, dtype=float)
    keep = (eps >= float(lo)) & (eps <= float(hi)) & np.isfinite(exponent)
    return {"epsilon": eps[keep], "exponent": exponent[keep]}


def boundaries_from_log_slope(profile, band1, band2):
    """
    Where the measured exponent changes, by least squares on the slope curve.

    Two changepoints are placed so the exponent is as nearly constant as it
    can be inside each of the three stretches they make. That is the
    definition of a boundary written down directly: not where the residual
    of a fitted force happens to be smallest, but where the curve stops
    obeying one power law and starts obeying another.

    A band of zero width means that boundary is not free -- ε₁ does nothing
    when every outer element carries load throughout -- so it is pinned to
    the value the band names and only the other one is searched.

    Returns (ε₁, ε₂, within-segment scatter) or None when the slope cannot
    be measured over enough of the curve to say.
    """
    eps = np.asarray(profile.get("epsilon", []), dtype=float)
    p = np.asarray(profile.get("exponent", []), dtype=float)
    if eps.size < 9:
        return None

    def allowed(band):
        low, high = float(band[0]), float(band[1])
        if high - low < 1e-6:
            return [int(np.argmin(np.abs(eps - low)))], low
        inside = np.nonzero((eps >= low - 1e-9) & (eps <= high + 1e-9))[0]
        return [int(i) for i in inside], None

    first, pinned_1 = allowed(band1)
    second, pinned_2 = allowed(band2)
    if not first or not second:
        return None

    def scatter(values):
        return (float(np.sum((values - values.mean()) ** 2))
                if values.size else 0.0)

    best = None
    for i in first:
        if i < 2 or i > eps.size - 5:
            continue
        for j in second:
            if j < i + 2 or j > eps.size - 2:
                continue
            cost = scatter(p[:i]) + scatter(p[i:j]) + scatter(p[j:])
            if best is None or cost < best[2]:
                best = (
                    float(pinned_1 if pinned_1 is not None else eps[i]),
                    float(pinned_2 if pinned_2 is not None else eps[j]),
                    cost,
                )
    return best


def evaluate_boundaries(model, lo, hi, terms, e1, e2, membrane, cyto_start):
    """One fit at one pair of boundaries, for comparing candidates fairly."""
    flags = {
        "use_membrane": "membrane" in terms,
        "use_interior": "interior" in terms,
        "use_nucleus": "nucleus" in terms,
        "use_tension": "tension" in terms,
        "use_nucleus_shell": "nucleus_shell" in terms,
        "use_cortex": "cortex" in terms,
    }
    try:
        return model.fit_composition(
            lo, hi, float(e1), float(e2), membrane, cyto_start,
            weighting=st.session_state["weighting"],
            fit_offset=st.session_state["fit_offset"], **flags
        )
    except Exception:  # pragma: no cover - defensive
        return {"success": False}


def boundary_grid(band, n):
    """A grid over a band, or the single point when the band is one."""
    a, b = float(band[0]), float(band[1])
    if b - a < 1e-6:
        return np.array([a])
    return np.linspace(a, b, int(n))


def epsilon_1_is_free(terms, membrane, cyto_start):
    """
    Whether ε₁ changes the model at all, as it is currently arranged.

    With every outer element carrying load throughout, ε₁ is a number the
    basis functions never read. Searching it then produces a different
    answer every time from noise alone, which is exactly what "the
    boundaries move a lot each press" was.
    """
    if any(term in terms and not carries_throughout(term)
           for term in ("membrane", "tension")) and membrane in ("freeze", "late"):
        return True
    if ("interior" in terms and not carries_throughout("interior")
            and cyto_start == "break"):
        return True
    return False


def refine_boundaries_control(model, lo, hi, terms):
    """
    Move the boundaries within the interval the biology allows, and no further.

    The free search was the problem: pressing it moved every boundary and
    every element's range across the whole curve, so two presses on one
    cell gave two different cells. What is actually being estimated is
    narrow. A C2C12 is met by its sarcolemma and cytoskeleton from first
    contact, so their ranges are not in question at all, and its nucleus is
    met between 44 % and 70 %, so ε₂ is a number inside a band a third of
    the curve wide. Constraining the profile to that band is not a
    convenience: an estimate allowed outside it would be reporting a cell
    nobody has ever measured.
    """
    if not terms:
        return
    cell_type, prior, membrane, cyto_start, e1_now, e2_now, band1, band2 = (
        _boundary_search_setup(lo, hi, terms))
    moving = []
    if band1[1] > band1[0]:
        moving.append(f"ε₁ within {band1[0]:.2f} to {band1[1]:.2f}")
    if band2[1] > band2[0]:
        moving.append(f"ε₂ within {band2[0]:.2f} to {band2[1]:.2f}")

    pressed = st.button(
        "📈 Estimate ε₁, ε₂: d ln F / d ln ε changepoints",
        key="refine_boundaries_button", type="primary",
        disabled=not terms, **STRETCH,
    )
    st.caption(
        "Boundaries found from the curve rather than assumed: it reads the "
        "curve on log-log axes, where every one of these laws is a straight "
        "line, and puts them where its slope stops obeying one power and "
        "starts obeying the next"
        + ((", searching " + " and ".join(moving)) if moving else "")
        + ". Then it fits at that placement and at the ones the residual "
        "prefers, and keeps whichever predicts best. "
        + (prior.get("why", "").capitalize() + ", so nothing is looked for "
           "outside that band. " if prior.get("why") else "")
        + "The components ticked above are not touched, and ranges you moved "
        "by hand are left alone."
    )

    show_boundary_candidates(model, lo, hi, terms, membrane, cyto_start)

    if not pressed:
        return
    estimate_boundaries_now(model, lo, hi, terms)


def _boundary_search_setup(lo, hi, terms):
    """The cell type's bands for ε₁ and ε₂, and the arrangement on the page."""
    cell_type = st.session_state.get("cell_type")
    prior = component_prior(cell_type)
    membrane = MEMBRANE_CHOICES.get(
        st.session_state["membrane_after_break"], "freeze")
    cyto_start = CYTO_CHOICES.get(st.session_state["cyto_starts_at"], "break")
    e1_now = float(st.session_state["segment_break_1"])
    e2_now = float(st.session_state["segment_break_2"])

    span = max(float(hi) - float(lo), 1e-6)
    deep = [t for t in terms if t in ("nucleus", "nucleus_shell")]
    declared = deep_onset_band(cell_type, lo, hi)
    if not deep:
        # No deep element in the model: ε₂ is not in the design at all, so
        # every value of it gives the same fit.
        band2 = (e2_now, e2_now)
    elif declared != (float(lo), float(hi)):
        band2 = declared
    else:
        # Nothing declared for this cell type, so the deep onset is looked
        # for over the stretch of the curve where one could be met at all.
        band2 = (lo + 0.30 * span, lo + 0.92 * span)
    free1 = epsilon_1_is_free(terms, membrane, cyto_start)
    band1 = (
        (lo + 0.06 * span, min(lo + 0.50 * span, band2[1] - 0.02 * span))
        if free1 else (e1_now, e1_now)
    )
    if band1[1] <= band1[0]:
        band1 = (e1_now, e1_now)

    return cell_type, prior, membrane, cyto_start, e1_now, e2_now, band1, band2


def estimate_boundaries_now(model, lo, hi, terms, then_fit=False):
    """
    ε₁, ε₂ read off the log curve inside the cell type's bands, the
    placement that predicts best kept, and (with ``then_fit``) the curve
    fitted there on the next pass. What ▶ Fit & plot does when the board
    says to estimate the boundaries.
    """
    if not terms:
        return
    cell_type, prior, membrane, cyto_start, e1_now, e2_now, band1, band2 = (
        _boundary_search_setup(lo, hi, terms))

    fixed = {t for t in terms if st.session_state.get(f"_window_touched_{t}")}
    picks = hypotheses_for(cell_type, terms=terms, exact=True)
    candidates = []
    with st.spinner("Reading the slope of the log curve, then fitting at "
                    "every placement it and the residual suggest…"):
        # 1. What the log curve says, on its own, with no fit involved.
        profile = log_slope_profile(model, lo, hi)
        st.session_state["_slope_profile"] = profile
        from_slope = boundaries_from_log_slope(profile, band1, band2)
        if from_slope:
            candidates.append({
                "why": "where the log slope changes",
                "break_1": from_slope[0], "break_2": from_slope[1],
                "membrane": membrane, "cyto_start": cyto_start,
            })
        # 2. Where the boundaries stand now, so a candidate is never worse
        #    than what it replaces without saying so.
        candidates.append({
            "why": "on the page now",
            "break_1": e1_now, "break_2": e2_now,
            "membrane": membrane, "cyto_start": cyto_start,
        })
        # 3. This cell type's stated default.
        stated = default_boundaries(cell_type)
        candidates.append({
            "why": "this cell type's default",
            "break_1": float(stated["segment_break_1"]),
            "break_2": float(stated["segment_break_2"]),
            "membrane": membrane, "cyto_start": cyto_start,
        })
        # 4. The full profile over the allowed band, hand-over and all,
        #    scored on points it was not fitted to.
        outcome = analyse_curve(
            model, lo, hi, picks,
            weighting=st.session_state["weighting"],
            measure_q=wants_confinement(),
            terms_hint=terms,
            n_grid=12, cv_repeats=3, passes=3,
            band_1=band1 if band1[1] > band1[0] else None,
            band_2=band2 if band2[1] > band2[0] else None,
        )
        found = outcome["hypotheses"]
        if found and found.get("success"):
            st.session_state["hypothesis_search"] = found
            if outcome["q_scan"] is not None:
                st.session_state["confinement_scan"] = outcome["q_scan"]
            winner = found["best"]
            candidates.append({
                "why": "smallest held-out error",
                "break_1": float(winner["break_1"]),
                "break_2": float(winner["break_2"]),
                "membrane": winner["membrane"],
                "cyto_start": winner["cyto_start"],
                "q": outcome["q"],
                "cv_rmse": float(winner.get("cv_rmse", float("nan"))),
            })

        # Every placement is fitted at each hand-over the cell type allows
        # and reported at its best one. Judging a placement at whichever
        # hand-over happened to be on the page compares boundaries by a
        # difference that is not about the boundaries.
        arrangements = [
            (spec.get("membrane", "continue"), spec.get("cyto_start", "zero"))
            for spec in picks
        ] or [(membrane, cyto_start)]
        rows = []
        for row in candidates:
            best_fit, best_pair = None, None
            for pair in arrangements:
                fitted = evaluate_boundaries(
                    model, lo, hi, terms, row["break_1"], row["break_2"],
                    pair[0], pair[1],
                )
                if not fitted.get("success"):
                    continue
                if best_fit is None or fitted["r_squared"] > best_fit["r_squared"]:
                    best_fit, best_pair = fitted, pair
            if best_fit is None:
                continue
            row["membrane"], row["cyto_start"] = best_pair
            row["r_squared"] = float(best_fit["r_squared"])
            row["chi"] = float(
                best_fit.get("chi_squared_reduced", float("nan"))
            )
            rows.append(row)

    if not rows:
        st.error("No placement in the allowed band fitted this curve.")
        if then_fit:
            rerun_keeping_settings({"_fit_from_curve": True})
        return
    # The one applied is the one that predicts best where that was measured,
    # and otherwise the closest fit. Every other row stays one click away.
    rows.sort(key=lambda r: (-r["r_squared"],))
    best = min(
        rows,
        key=lambda r: (0 if np.isfinite(r.get("cv_rmse", float("nan"))) else 1,
                       r.get("cv_rmse", float("inf")), -r["r_squared"]),
    )
    st.session_state["boundary_candidates"] = rows
    st.session_state["_last_search"] = "boundaries"
    if fixed:
        st.session_state["_boundaries_left_alone"] = sorted(fixed)
    rerun_keeping_settings({**settings_for_boundaries(best),
                            **({"_fit_from_curve": True} if then_fit else {})})


def settings_for_boundaries(row):
    """The widget values one candidate placement stands for."""
    pending = {
        "segment_break_1": round(float(row["break_1"]), 4),
        "segment_break_2": round(float(row["break_2"]), 4),
        "model_kind": "Segmented (each part takes over in turn)",
        "membrane_after_break": next(
            k for k, v in MEMBRANE_CHOICES.items() if v == row["membrane"]
        ),
        "cyto_starts_at": next(
            k for k, v in CYTO_CHOICES.items() if v == row["cyto_start"]
        ),
    }
    if row.get("q") is not None and np.isfinite(row.get("q", float("nan"))):
        pending["confinement"] = round(float(row["q"]), 2)
    return pending


def show_boundary_candidates(model, lo, hi, terms, membrane, cyto_start):
    """
    The placements that were tried, what each came to, and a way to take one.

    A search that hands back one number asks to be trusted. A search that
    shows the four placements it weighed, and what the curve looks like at
    each, can be argued with, which is the only useful kind.
    """
    rows = st.session_state.get("boundary_candidates")
    if not rows:
        return
    here = (round(float(st.session_state["segment_break_1"]), 4),
            round(float(st.session_state["segment_break_2"]), 4))
    st.markdown("**Placements tried**")
    for index, row in enumerate(rows):
        mine = (round(float(row["break_1"]), 4),
                round(float(row["break_2"]), 4)) == here
        c1, c2, c3, c4 = st.columns([1.5, 1.1, 1.6, 0.9])
        c1.markdown(("**" if mine else "") + row["why"] + ("**" if mine else ""))
        c2.markdown(
            f"ε₁ {float(row['break_1']):.3f} · ε₂ {float(row['break_2']):.3f}"
            + "<br><span style='font-size:0.78em;opacity:0.75'>"
            + next(k for k, v in MEMBRANE_CHOICES.items()
                   if v == row["membrane"])
            + ", cytoskeleton "
            + next(k for k, v in CYTO_CHOICES.items()
                   if v == row["cyto_start"])
            + "</span>",
            unsafe_allow_html=True,
        )
        chi = row.get("chi", float("nan"))
        c3.markdown(
            f"R² {row['r_squared']:.5f}"
            + (f" · χ²/dof {chi:.3g}" if np.isfinite(chi) else "")
            + (f" · CV {row['cv_rmse']:.3g} N"
               if np.isfinite(row.get("cv_rmse", float("nan"))) else "")
        )
        if mine:
            c4.markdown("✅ in use")
        elif c4.button("✓ Apply this row", key=f"use_boundary_{index}", **STRETCH):
            rerun_keeping_settings(settings_for_boundaries(row))
    st.caption(
        "Every row is fitted the same way over the same range, so the "
        "columns compare like with like. R² cannot choose between them on "
        "its own, which is why the row that was applied is the one with the "
        "smallest held-out error where that was measured."
    )


def _scan_q_at(model, lo, hi, terms, e1, e2, membrane, cyto_start):
    """The confinement exponent that fits best at one pair of boundaries."""
    try:
        return model.scan_confinement(
            lo, hi, e1=float(e1), e2=float(e2),
            membrane=membrane, cyto_start=cyto_start,
            use_nucleus="nucleus" in terms,
            use_tension="tension" in terms,
            use_nucleus_shell="nucleus_shell" in terms,
            use_cortex="cortex" in terms,
            weighting=st.session_state["weighting"],
        )
    except Exception:  # pragma: no cover - defensive
        return None


def profile_boundaries(model, lo, hi, terms, band1, band2,
                       membrane="freeze", cyto_start="break", n=13, rounds=2):
    """
    The residual surface over the allowed boundary pairs, and its minimum.

    One exact bounded least-squares solve per node, which is what makes an
    exhaustive profile affordable and removes every question of starting
    values: there is nothing to start from.

    A single grid can only ever land on its own points, and half a step on
    a 13-point grid is enough to move a modulus, so the coarse pass is
    followed by refinement rounds that re-grid inside one step either side
    of the incumbent, staying inside the allowed band. Two rounds narrow
    each boundary by about 25 times for a fraction of the cost of a fine
    grid over the whole band.
    """
    flags = {
        "use_membrane": "membrane" in terms,
        "use_interior": "interior" in terms,
        "use_nucleus": "nucleus" in terms,
        "use_tension": "tension" in terms,
        "use_nucleus_shell": "nucleus_shell" in terms,
        "use_cortex": "cortex" in terms,
    }
    trials, best = [], None
    first, second = boundary_grid(band1, n), boundary_grid(band2, n)
    for round_index in range(int(rounds) + 1):
        for e1 in first:
            for e2 in second:
                if float(e2) <= float(e1):
                    continue
                try:
                    got = model.fit_composition(
                        lo, hi, float(e1), float(e2), membrane, cyto_start,
                        weighting=st.session_state["weighting"],
                        fit_offset=st.session_state["fit_offset"], **flags
                    )
                except Exception:  # pragma: no cover - defensive
                    continue
                if not got.get("success"):
                    continue
                trials.append({
                    "e1": float(e1), "e2": float(e2),
                    "r_squared": float(got["r_squared"]),
                })
                if best is None or got["r_squared"] > best["r_squared"]:
                    best = got
        if best is None or round_index == int(rounds):
            break
        # Re-grid inside one step either side of the winner, clipped to the
        # band: the prior is not relaxed by refining.
        def _around(grid, centre, band):
            if grid.size < 2:
                return grid
            step = (grid[-1] - grid[0]) / max(grid.size - 1, 1)
            low = max(float(band[0]), float(centre) - step)
            high = min(float(band[1]), float(centre) + step)
            if high - low < 1e-9:
                return np.array([float(np.clip(centre, band[0], band[1]))])
            return np.linspace(low, high, max(5, int(n) // 2))

        first = _around(first, best["break_1"], band1)
        second = _around(second, best["break_2"], band2)
    if best is None:
        return {"success": False,
                "error": "No usable fit anywhere in the allowed band."}
    return {
        "success": True,
        "best": best,
        "best_break_1": float(best["break_1"]),
        "best_break_2": float(best["break_2"]),
        "trials": trials,
        "band_1": tuple(float(v) for v in band1),
        "band_2": tuple(float(v) for v in band2),
    }


def free_placement_control(model, lo, hi, terms):
    """
    The unconstrained placement search, kept but moved out of the way.

    It searches both edges of every element's range against held-out error,
    which is the right tool when a cell type has no prior worth the name
    and the wrong one when it does: on a C2C12 it will happily move the
    sarcolemma off first contact, which is not a thing a sarcolemma does.
    So it lives in the settings, behind a sentence saying what it will do.
    """
    if search_term_windows is None or not terms:
        return
    st.markdown("**🧪 Search every element's range freely**")
    st.caption(
        "Moves both edges of every element's range, scoring each placement "
        "on held-out points, with no prior about where an element belongs. "
        "It overwrites every range at once, including ones you set, so it "
        "is for a cell type this app has no expectations about rather than "
        "for tidying up a C2C12."
    )
    if not st.button("Unconstrained window search: argmin S", key="free_placement",
                     **STRETCH):
        return
    with st.spinner("Moving each range and scoring every placement…"):
        try:
            found = search_term_windows(
                model, lo, hi, terms,
                membrane=MEMBRANE_CHOICES.get(
                    st.session_state["membrane_after_break"], "freeze"),
                cyto_start=CYTO_CHOICES.get(
                    st.session_state["cyto_starts_at"], "break"),
                e1=float(st.session_state["segment_break_1"]),
                e2=float(st.session_state["segment_break_2"]),
                weighting=st.session_state["weighting"],
                fit_offset=st.session_state["fit_offset"],
                start_from=element_windows(terms, lo, hi),
            )
        except Exception as exc:  # pragma: no cover - defensive
            found = {"success": False, "error": str(exc)}
    st.session_state["element_window_search"] = found
    st.session_state["_last_search"] = "windows"
    if found.get("success"):
        pending = {
            element_window_key(term): tuple(window)
            for term, window in found["windows"].items()
        }
        pending.update({
            f"_window_touched_{term}": True for term in found["windows"]
        })
        rerun_keeping_settings(pending)
    st.error(found.get("error", "The placement search failed."))


def optimisation_controls(model, lo, hi, terms):
    """The optimisation step: what it decided, and the argument for how."""
    if not terms:
        return
    b1, b2 = st.columns([2.2, 1])
    with b1:
        # Filled once the fit of this pass exists (boundaries_note), so it
        # names the boundaries drawn on the plot, not the fit before.
        slot = st.container()
        st.caption(
            "They are read off the log curve on the **📈 Log curve and "
            "boundaries** tab, which is also where the placements tried and "
            "the algebra of each stretch live."
        )
    with b2:
        layer_checkbox("boundaries", "Draw on the plot", key="layer_boundaries")
    # What every curve of this type starts from stays here: it is a setting
    # for the whole experiment, not an analysis of this cell.
    set_default_boundaries_control()
    return slot


def components_note(fit):
    """What the plot is drawing, under the bars that say what to draw next."""
    if not (fit and fit.get("success")):
        st.caption("The plot shows nothing yet: press **▶ Fit & plot**.")
        return
    drawn = []
    for term in ALL_TERMS:
        if term not in (fit.get("terms") or ()):
            continue
        drawn.append(f"{plain_name(term).lower()} ({element_support(term, fit)})")
    ticked = {t for t in terms_for(st.session_state.get("cell_type"))
              if st.session_state.get(f"use_{t}")}
    fitted = set(fit.get("terms") or ())
    st.caption("**On the plot:** " + ("; ".join(drawn) if drawn else "nothing")
               + ".")
    if ticked != fitted:
        missing = ", ".join(plain_name(t).lower() for t in sorted(ticked - fitted))
        extra = ", ".join(plain_name(t).lower() for t in sorted(fitted - ticked))
        st.caption(
            "⚠️ The ticks and the plot differ: "
            + (f"ticked but not drawn: {missing}. " if missing else "")
            + (f"drawn but no longer ticked: {extra}. " if extra else "")
            + "Press **▶ Fit & plot** to make the plot match the board."
        )


def legacy_graph_note(fit, epsilon=None):
    """
    What is on the curve above, in one line, under it.

    Written from the fit that was drawn, so it describes the picture on
    screen rather than what the board would draw next.
    """
    n = int(np.size(epsilon)) if epsilon is not None else 0
    if not (fit and fit.get("success")):
        st.caption(
            "**On the graph now:** the measured curve"
            + (f" ({n} points)" if n else "")
            + " only. Press **▶ Fit & plot** and the fitted curve, its "
            "components and its boundaries are drawn on it."
        )
        return
    drawn = [f"{plain_name(term).lower()} ({element_support(term, fit)})"
             for term in ALL_TERMS if term in (fit.get("terms") or ())]
    lo, hi = fit.get("epsilon_range", (float("nan"), float("nan")))
    parts = [
        "**On the graph now:** the data"
        + (f" ({n} points)" if n else "")
        + " and the fitted curve"
        + (f" of `fit {fit['fit_id']}`" if fit.get("fit_id") else "")
        + (f", R² = {fit['r_squared']:.5f}"
           if np.isfinite(fit.get("r_squared", float("nan"))) else ""),
        ("components, each over its own stretch: " + "; ".join(drawn))
        if drawn else "no separate components",
    ]
    edges = fit_edges(fit)
    if edges:
        parts.append("boundaries " + ", ".join(f"{name} = {value:.3f}"
                                               for value, name in edges)
                     + (" (drawn)" if layer_on("boundaries")
                        else " (lines hidden)"))
    if np.isfinite(lo) and np.isfinite(hi):
        parts.append(f"fitted over ε {lo:.3f} to {hi:.3f}"
                     + (", shaded" if layer_on("range") else ""))
    st.caption(" · ".join(parts) + ".")


def boundaries_note(fit):
    """The boundaries of the fit on the plot, and any the next fit will move."""
    if not (fit and fit.get("success")):
        st.markdown(
            f"ε₁ = {float(st.session_state['segment_break_1']):.3f}, "
            f"ε₂ = {float(st.session_state['segment_break_2']):.3f} "
            "(not fitted yet)."
        )
        return
    edges = fit_edges(fit)
    shown = ", ".join(f"**{name} = {value:.3f}**" for value, name in edges)
    st.markdown(
        (f"On the plot: {shown}" if shown else "This model has no boundaries")
        + (f" · `fit {fit['fit_id']}`" if fit.get("fit_id") else "")
    )
    used = {name for _v, name in edges}
    notes = []
    for key, name in (("segment_break_1", "ε₁"), ("segment_break_2", "ε₂")):
        set_to = float(st.session_state.get(key, float("nan")))
        was = fit.get("break_1" if name == "ε₁" else "break_2")
        if name not in used:
            notes.append(f"{name} = {set_to:.3f} is set but not used by these "
                         "components")
        elif was is not None and abs(set_to - float(was)) > 5e-4:
            notes.append(f"{name} is set to {set_to:.3f}; ▶ Fit & plot moves the "
                         "line")
    if notes:
        st.caption("; ".join(notes) + ".")


def _term_symbol(term):
    """This element's modulus symbol, as LaTeX."""
    return EQUATION_TERMS.get(term, (None, None, None, term))[3]


def search_results_as_equations():
    """
    What the last search decided, written as the arithmetic it decided on.

    A table invited the reader to compare the candidates by eye, which is
    exactly the comparison the arithmetic was run to replace. What settled
    it is the residual at the nodes that were tried, so that is what is
    printed.
    """
    which = st.session_state.get("_last_search")
    if which == "boundaries":
        found = st.session_state.get("hypothesis_search")
        if not (found and found.get("success")):
            return
        rows = sorted(found["candidates"], key=lambda r: r["cv_rmse"])
        best = found["best"]
        tau = float(found.get("tie_tolerance", float("nan")))
        lines = []
        for row in rows[:5]:
            mark = ""
            if row is best or row.get("chosen"):
                mark = r"&& \leftarrow \text{kept}"
            elif row.get("tied_with_best"):
                mark = r"&& \text{(within } \tau \text{)}"
            lines.append(
                r"\mathrm{CV}(\varepsilon_1=" + f"{row['break_1']:.3f}"
                + r",\,\varepsilon_2=" + f"{row['break_2']:.3f}" + r") &= "
                + _sci_latex(row["cv_rmse"]) + r"\ \mathrm{N}" + mark
            )
        st.latex(r"\begin{aligned}" + r"\\".join(lines) + r"\end{aligned}")
        if np.isfinite(tau):
            st.latex(r"\tau = " + _sci_latex(tau) + r"\ \mathrm{N}")
        st.caption(
            f"Kept: ε₁ = {float(best['break_1']):.3f}, "
            f"ε₂ = {float(best['break_2']):.3f}"
            + (f", q = {float(best['confinement']):.2f}"
               if np.isfinite(best.get("confinement", float("nan"))) else "")
            + f" · {best['label'].lower()}. Each row is one hand-over "
            "profiled over the allowed boundary grid at its own confinement, "
            "then scored on points it was not fitted to; the moduli at every "
            "node are the exact bounded least-squares solution there, which "
            "is what makes an exhaustive profile affordable."
            + ("" if found.get("clear_cut") else
               " The best two are inside τ, so this curve cannot separate "
               "them and what the cell type is known to do broke the tie.")
        )
        return

    if which == "windows":
        found = st.session_state.get("element_window_search")
        if not (found and found.get("success")):
            return
        lines = []
        for term, window in found["windows"].items():
            lines.append(
                _term_symbol(term) + r"&: \varepsilon \in ["
                + f"{float(window[0]):.3f}" + r",\,"
                + f"{float(window[1]):.3f}" + r"]"
            )
        st.latex(r"\begin{aligned}" + r"\\".join(lines) + r"\end{aligned}")
        cv = float(found.get("cv_rmse", float("nan")))
        if np.isfinite(cv):
            st.latex(r"\mathrm{CV} = " + _sci_latex(cv) + r"\ \mathrm{N}")
        st.caption(
            "Each interval is that element's support: it begins carrying "
            "load at the lower edge and stops taking more at the upper one, "
            "holding what it reached past it, so the total force has no "
            "step in it. The placement was scored on held-out points, the "
            "same criterion as the element search."
        )


def why_this_search(model=None):
    """
    The mathematics the search button follows, written out.

    Not decoration. Every choice below is forced by a property of this
    model, and someone quoting a modulus from this app is entitled to the
    argument for why the boundary it was measured at is the boundary the
    curve has.
    """
    with st.expander(
        "∑ Model selection and boundary estimation, in full", expanded=False
    ):
        st.caption(
            "The two buttons above solve one joint problem, separated "
            "because the two halves are answered on different criteria. "
            "Written out:"
        )
        st.latex(
            r"\bigl(\mathcal{S}^{*},\varepsilon_1^{*},\varepsilon_2^{*}"
            r"\bigr) \;=\; \operatorname*{arg\,min}_{\mathcal{S}\subseteq"
            r"\mathcal{C},\;(\varepsilon_1,\varepsilon_2)\in\mathcal{G}}"
            r"\;\mathrm{CV}\bigl(\mathcal{S},\varepsilon_1,\varepsilon_2"
            r"\bigr) \quad\text{s.t.}\quad |\mathcal{S}| \;\text{minimal "
            r"within}\; \tau"
        )
        st.caption(
            "𝒞 is the admissible element set for this cell type, 𝒢 the "
            "boundary grid, CV the generalisation error, τ the resolution "
            "of that estimate. Support recovery (which elements) and "
            "changepoint estimation (where they hand over) are separated "
            "because a subset chosen at the wrong changepoint is being "
            "scored on the wrong design matrix, and a changepoint estimated "
            "under a misspecified subset absorbs the missing term."
        )

        st.markdown("**0 · Initial values, and the order the steps run in**")
        st.caption(
            "There is no initial guess for the moduli, and none is "
            "possible to get wrong: at fixed changepoints the problem is "
            "convex with a unique global minimiser, so the solver returns "
            "the same θ̂ from anywhere. What does need starting values is "
            "the small set of parameters that enter non-linearly, and "
            "those come from prior knowledge of the cell type rather than "
            "from the curve: "
            + (component_prior().get("why", "")
               or "no prior is declared for this cell type, so the "
                  "changepoints start at the stored defaults")
            + ". The estimate is then a block coordinate descent, each "
            "block solved exactly rather than stepped:"
        )
        st.latex(
            r"\begin{aligned}"
            r"\text{(i)}\quad & \hat{\theta}^{(m)} = \arg\min_{\theta\ge0}"
            r"\lVert W(F - X(\varepsilon^{(m)})\theta)\rVert^{2} "
            r"&& \text{exact, BVLS}\\"
            r"\text{(ii)}\quad & \varepsilon^{(m+1)} = \arg\min_{\varepsilon"
            r"\in\mathcal{G}} S(\varepsilon) && \text{grid profile, then "
            r"two refinements}\\"
            r"\text{(iii)}\quad & q^{(m+1)} = \arg\min_{q} S(\varepsilon^{"
            r"(m+1)};q) && \text{scanned at the incumbent boundaries}\\"
            r"\text{(iv)}\quad & \varepsilon^{(m+2)} = \arg\min_{\varepsilon"
            r"\in\mathcal{G}} S(\varepsilon;q^{(m+1)}) && \text{re-profiled "
            r"at the new } q\\"
            r"\text{(v)}\quad & \mathcal{S}^{*} = \arg\min_{\mathcal{S}}"
            r"\mathrm{CV}(\mathcal{S}) && \text{support, on held-out error}"
            r"\end{aligned}"
        )
        st.caption(
            "Steps (ii) to (iv) are one descent, not a pipeline: the "
            "confinement exponent q multiplies every basis function, so "
            "changepoints estimated at the wrong q are estimated on the "
            "wrong design, and q measured at the wrong changepoints absorbs "
            "the mismatch. Each candidate arrangement carries its own q "
            "through the comparison for the same reason, which is also what "
            "stops the answer depending on which candidate was fitted "
            "first. S is non-increasing along the sequence and each block "
            "is solved to its exact optimum, so the descent terminates; in "
            "practice three passes move χ²/dof by less than the "
            "fold-to-fold spread. Every quantity that is not solved exactly "
            "is profiled on a closed interval, so nothing here can diverge "
            "or fail to converge."
        )

        st.markdown("**1 · Conditional linearity of the constitutive model**")
        st.caption(
            "The elements act in parallel under a common areal strain, so "
            "their tractions superpose and the model is linear in the "
            "moduli once the changepoints are held fixed:"
        )
        st.latex(
            r"F(\varepsilon) \;=\; \sum_k a_k E_k\, g_k(\varepsilon;"
            r"\varepsilon_1,\varepsilon_2) \;=\; X(\varepsilon_1,"
            r"\varepsilon_2)\,\theta, \qquad \theta_k = a_k E_k \ge 0"
        )
        st.latex(
            r"\hat{\theta} \;=\; \arg\min_{\theta \ge 0} \;\bigl\| W\bigl("
            r"F - X(\varepsilon_1,\varepsilon_2)\theta\bigr) \bigr\|^{2}"
        )
        st.caption(
            "The prefactors a_k are pure geometry, fixed before fitting from "
            "the cell's height, radius and shell thickness, so the free "
            "parameters are moduli and nothing else. This is a "
            "bounded-variable least-squares programme: convex, with a unique "
            "global solution and KKT conditions that put an inactive element "
            "exactly on its bound. The non-negativity is thermodynamic, not "
            "numerical: a passive element cannot do negative work on the "
            "indenter, so an element returned at θ = 0 is an estimate of "
            "zero stiffness contribution over that support, not a solver "
            "failure."
        )

        st.markdown("**2 · Changepoint estimation by profiling**")
        st.latex(
            r"S(\varepsilon_1,\varepsilon_2) \;=\; \min_{\theta \ge 0} "
            r"\bigl\| W\bigl(F - X(\varepsilon_1,\varepsilon_2)\theta\bigr) "
            r"\bigr\|^{2}, \qquad (\hat{\varepsilon}_1,\hat{\varepsilon}_2) "
            r"= \arg\min S"
        )
        st.caption(
            "The changepoints enter through min(·) and the Macaulay bracket "
            "⟨·⟩, so S is continuous but only piecewise differentiable: its "
            "gradient jumps each time a boundary crosses a sample, and "
            "quasi-Newton steps walk through the minimum rather than into "
            "it. The nuisance parameters are therefore concentrated out "
            "analytically and the two-dimensional profile is evaluated "
            "exhaustively on 𝒢, then re-gridded within one step of the "
            "incumbent for two rounds, contracting the bracket by about 25×. "
            "Cost is one BVLS solve per node, which is milliseconds."
        )

        st.markdown("**3 · The physical meaning of a changepoint**")
        st.caption(
            "Each constitutive law is a power law in ε, so the local "
            "logarithmic stiffness is the traction-weighted mean of the "
            "exponents in play:"
        )
        st.latex(
            r"\frac{d\log F}{d\log \varepsilon} \;=\; \sum_k w_k\,p_k, "
            r"\qquad w_k = \frac{a_k E_k g_k}{\sum_j a_j E_j g_j}"
        )
        st.caption(
            "p = 3 for area-strain stretching of a fluid-filled shell "
            "(Lulevich eq 3), p = 3/2 for Hertzian compression of the "
            "cytoplasmic continuum and of the deep organelle (eq 6), p = 1 "
            "for pre-existing cortical tension resisting the area gained on "
            "flattening. A changepoint is where the weight transfers from "
            "one law to the next, so the apparent exponent relaxes from one "
            "value towards another; S is minimised there because that is "
            "the only place a single kinked basis can reproduce the "
            "curvature of the transition. The exponent read directly off "
            "the data by local regression is reported below the fit, which "
            "makes the estimate falsifiable independently of the fit."
        )

        st.markdown("**4 · Support recovery cannot be done on the residual**")
        st.caption(
            "The candidate models are nested, so the residual sum of "
            "squares is monotone non-increasing in |𝒮|: RSS, R² and χ² all "
            "improve on adding an element that contributes nothing, and an "
            "information criterion only trades that off against a penalty "
            "chosen a priori. Out-of-sample prediction error estimates the "
            "quantity that is actually at issue."
        )
        st.latex(
            r"\mathrm{CV}(\mathcal{S}) = \frac{1}{R}\sum_{r=1}^{R}"
            r"\frac{1}{K}\sum_{k=1}^{K} \sqrt{\frac{1}{|f_k|}"
            r"\sum_{i \in f_k}\bigl(F_i - \hat{F}^{(-k)}_{\mathcal{S}}"
            r"(\varepsilon_i)\bigr)^{2}}"
        )
        st.caption(
            "Every 𝒮 ⊆ 𝒞 is refitted on each training partition at the same "
            "changepoints and evaluated on the fold it never saw: the "
            "criterion is how well a model predicts points it was not "
            "fitted to. K = 5, R = 3 "
            "independent permutations, because the fold-to-fold variance on "
            "a few hundred correlated samples is itself large enough to "
            "reorder candidates that are not distinguishable."
        )

        st.markdown("**5 · Identifiability sets the tie rule**")
        st.latex(
            r"\tau = \max\bigl(0.05\,\mathrm{CV}_{\min},\; s_{\min} + "
            r"\max_j s_j\bigr)"
        )
        st.caption(
            "s is the between-permutation standard deviation of a "
            "candidate's score, so τ is the resolution of the estimator "
            "itself; differences below it carry no information. Among "
            "candidates inside τ the minimal support is retained, and the "
            "reason is collinearity of the design, not Occam:"
        )
        st.latex(
            r"\operatorname{Var}(\hat{\theta}_k) \;=\; \frac{\sigma^{2}}"
            r"{(1-\rho^{2})\,\lVert x_k \rVert^{2}}, \qquad "
            r"\kappa(X) = \frac{\sigma_{\max}(X)}{\sigma_{\min}(X)}"
        )
        st.caption(
            "Two basis functions with correlation ρ leave their sum sharply "
            "determined and their partition variance-inflated by "
            "1/(1 − ρ²). Retaining a component the design cannot separate "
            "does not bias the total traction; it makes the partition of "
            "that traction between neighbours drift from cell to cell, "
            "which is precisely the variance that destroys a population "
            "comparison. The realised ρ and the condition number κ(X) are "
            "both reported with the fit."
        )

        st.markdown("**6 · Admissibility of the basis**")
        st.caption(
            "Two elements are separable only if their laws differ in "
            "exponent or in support. Identical p with identical onset gives "
            "a rank-deficient design and an unidentifiable partition, no "
            "matter how many samples are taken, which is why each element "
            "in 𝒞 carries either its own exponent or its own changepoint. "
            "It is also why fixation collapses the model to a set of "
            "Hertzian terms separated only by onset: cross-linking removes "
            "the fluid-filled shell that made the cube law distinct."
        )
        share = (model.bending_share(0.30)
                 if hasattr(model, "bending_share") else None)
        st.caption(
            "**Bending.** Lulevich eq 1 carries a bending contribution "
            "F_bend = π Eₘ h² ε^½ / 2√2 alongside the stretching term, with "
            "exponent ½, so it is a distinct shape rather than a "
            "reparametrisation. Eq 2 gives the ratio of bending to "
            "stretching as (h/R)/ε^{5/2}"
            + (f", which on this cell is {float(share):.3g} at ε = 0.30"
               if share is not None and np.isfinite(float(share)) else "")
            + ". A column with that little leverage has an ill-determined "
            "coefficient and, being non-orthogonal to the rest, acts as a "
            "sink for model error elsewhere, so it is excluded from 𝒞 "
            "rather than offered and regularised."
        )


def set_default_boundaries_control():
    """
    Set the boundaries every new curve of this cell type starts from.

    A lab settles on a placement and then wants it back on every cell, not
    typed again for each one. What is saved here is what the app fills in
    the moment a curve of this type loads, and what the sidebar shows as
    this type's boundaries.
    """
    cell_type = st.session_state.get("cell_type", "")
    with st.expander("⚙️ Set the default boundaries for this cell type"):
        now = default_boundaries(cell_type)
        d1, d2, d3 = st.columns([1, 1, 1.2])
        with d1:
            first = st.number_input(
                "Default ε₁", min_value=0.0, max_value=0.99, step=0.005,
                format="%.3f", value=float(now["segment_break_1"]),
                key="default_break_1_box",
            )
        with d2:
            second = st.number_input(
                "Default ε₂", min_value=0.01, max_value=1.0, step=0.005,
                format="%.3f", value=float(now["segment_break_2"]),
                key="default_break_2_box",
            )
        with d3:
            st.markdown("<div style='height:1.7rem'></div>",
                        unsafe_allow_html=True)
            saved = st.button("💾 Save as the default", key="save_defaults",
                              **STRETCH)
        st.caption(
            f"Every new **{cell_type}** curve will start here, and the "
            "component ranges with it. Pressing this also applies them to "
            "the curve on screen."
        )
        stored = (st.session_state.get("learned_boundaries") or {}).get(cell_type)
        if stored:
            st.caption(
                f"Your own defaults are in force: ε₁ = "
                f"{stored['break_1']:.3f}, ε₂ = {stored['break_2']:.3f}."
            )
            if st.button("↺ Back to the built-in ones", key="forget_defaults",
                         **STRETCH):
                store = dict(st.session_state.get("learned_boundaries", {}))
                store.pop(cell_type, None)
                st.session_state["learned_boundaries"] = store
                for term in ALL_TERMS:
                    st.session_state.pop(f"_window_touched_{term}", None)
                rerun_keeping_settings(default_boundaries(cell_type))
        if not saved:
            return
        if second <= first:
            st.error("ε₂ has to be past ε₁.")
            return
        store = dict(st.session_state.get("learned_boundaries", {}))
        store[cell_type] = {
            "n": 0, "break_1": float(first), "break_2": float(second),
            "spread_1": None, "spread_2": None,
        }
        st.session_state["learned_boundaries"] = store
        # Setting a default means the ranges should follow it again.
        for term in ALL_TERMS:
            st.session_state.pop(f"_window_touched_{term}", None)
        rerun_keeping_settings({
            "segment_break_1": round(float(first), 3),
            "segment_break_2": round(float(second), 3),
        })


def element_window_controls(terms, lo, hi, step, e1, e2, membrane="freeze",
                            cyto_start="break", heading=None):
    """
    One bar per element: the stretch of the squash it is allowed to act on.

    The bar is not a view of the fit, it is part of the model. An element's
    window is where its own law starts and where it stops taking more, so
    moving a handle changes the curve and every modulus in it. That is why
    the whole panel is behind a tick: the boundaries ε₁ and ε₂ describe the
    same thing more simply, and are right for most cells.
    """
    if not terms:
        return None
    if heading:
        st.markdown(f"##### {heading}")
    own = st.checkbox(
        "Set these myself",
        key="use_element_windows",
        help="Off, the bars show where the fit will place each material: "
        "the shell from first contact, the scaffolding from ε₁, whatever is "
        "deeper from ε₂, with ε₁ and ε₂ found from the curve. On, each bar "
        "is yours to move and the fit uses exactly what you set.",
    )

    names = components_for(st.session_state["cell_type"])
    columns = st.columns(min(len(terms), 2))
    for index, term in enumerate(terms):
        key = element_window_key(term)
        automatic = default_element_window(term, lo, hi, e1, e2, membrane,
                                           cyto_start)
        # While the bars are not yours, they follow the fit rather than
        # holding an old position: they are showing you where each material
        # will act, and a bar that shows the wrong place is worse than none.
        if not own or not st.session_state.get(key):
            st.session_state[key] = automatic
        with columns[index % len(columns)]:
            bar = f"{key}__bar"
            pair = st.session_state[key]
            a = float(np.clip(pair[0], lo, hi - step))
            b = float(np.clip(pair[1], a + step, hi))
            st.session_state[bar] = (a, b)

            def _store(term=term, key=key, bar=bar):
                got = st.session_state.get(bar)
                if got:
                    st.session_state[key] = (
                        round(float(got[0]), 4), round(float(got[1]), 4)
                    )

            st.slider(
                f"{names[term][0]} · {TERM_SYMBOLS.get(term, term)}",
                min_value=float(lo), max_value=float(hi), step=float(step),
                key=bar, on_change=_store, disabled=not own,
                help="Where this material starts carrying load and where it "
                     "stops taking more. Past the far end it holds what it "
                     "reached rather than vanishing, so the curve has no "
                     "step in it.",
            )
    if not own:
        st.caption(
            "These follow the fit: ε₁ and ε₂ are found from the curve, and "
            "the bars move with them. Tick **Set these myself** to place a "
            "material by hand."
            + ("  A C2C12 usually goes membrane first, then the "
               "cytoskeleton — or both together from contact — then the "
               "nuclear envelope and what it contains, met at the same "
               "deformation."
               if st.session_state.get("cell_type", "").startswith("Myoblast")
               else "")
        )
    else:
        automatic = {
            term: default_element_window(term, lo, hi, e1, e2, membrane,
                                         cyto_start)
            for term in terms
        }
        drifted = [
            term for term in terms
            if st.session_state.get(element_window_key(term)) != automatic[term]
        ]
        if drifted:
            st.caption(
                "These are yours now, so a fit that moves ε₁ or ε₂ no longer "
                "moves them. "
                + ", ".join(plain_name(t) for t in drifted)
                + (" is" if len(drifted) == 1 else " are")
                + " away from where the fit would put "
                + ("it." if len(drifted) == 1 else "them.")
            )
            if st.button("↺ Put them back where the fit wants them",
                         key="reset_element_windows", **STRETCH):
                st.session_state["_pending_settings"] = {
                    element_window_key(term): window
                    for term, window in automatic.items()
                }
                st.rerun()
    return element_windows(terms, lo, hi)


def boundary_control(key, label, lower, upper, step, help_text=None,
                     disabled=False):
    """
    One boundary, set on its own by a bar or by typing.

    ``key`` is a plain session key rather than a widget key, so the search
    buttons and the winning picture can write it at any point and these
    widgets follow on the next pass. The two boundaries are kept apart by
    the caller's bounds: ε₁ cannot be pushed past ε₂, and neither can leave
    the fitted range.
    """
    lower, upper, step = float(lower), float(upper), float(step)
    if upper - lower < step:
        upper = lower + step
    bar, box = f"{key}__bar", f"{key}__box"
    value = float(np.clip(round(_as_float(st.session_state.get(key), lower), 4),
                          lower, upper))

    def _store(new_value):
        st.session_state[key] = float(
            np.clip(round(float(new_value), 4), lower, upper)
        )

    def _from_bar():
        _store(st.session_state.get(bar, value))

    def _from_box():
        _store(st.session_state.get(box, value))

    st.session_state[bar] = value
    st.session_state[box] = value
    st.slider(label, min_value=lower, max_value=upper, step=step, key=bar,
              on_change=_from_bar, help=help_text, disabled=disabled)
    st.number_input(
        f"{label} — type it", min_value=lower, max_value=upper, step=step,
        format="%.3f", key=box, on_change=_from_box, disabled=disabled,
        label_visibility="collapsed",
    )
    return value


# ==================================================== the archive database ==


def current_fit_settings():
    """Everything needed to reproduce the fit currently configured."""
    # A piecewise cell is recorded with the settings of the fit on its plot,
    # not with whatever is waiting on the control board.
    applied = (st.session_state.get("pw_applied") or {}).get("values")
    if applied and _PW_SOURCE[0] is None and piecewise_on():
        with pw_applied_values(applied):
            return current_fit_settings()
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
        "protein_coat_nm": float(st.session_state["protein_coat_nm"]),
        "sarcomere_nm": float(st.session_state["sarcomere_nm"]),
        "sarcomere_spread": float(st.session_state["sarcomere_spread"]),
        "cell_shape": st.session_state["cell_shape"],
        "confinement": float(st.session_state["confinement"]),
        "probe_diameter_um": float(st.session_state["probe_diameter_um"]),
        "approach_speed_um_s": float(st.session_state["approach_speed_um_s"]),
        "poisson_membrane": float(st.session_state["poisson_membrane"]),
        "poisson_interior": float(st.session_state["poisson_interior"]),
        "term_windows": {
            term: list(st.session_state.get(f"window_term_{term}", (0.0, 1.0)))
            for term in ALL_TERMS
        },
        "combined_window": list(st.session_state.get("window_combined", (0.0, 1.0))),
        # The four-regime C2C12 fit: which fit was on, where its regimes
        # were, and any initial guess or bound changed from the spec.
        "fit_mode": (
            st.session_state.get("c2c12_fit_mode", "")
            if piecewise_offered() else ""
        ),
        "piecewise_boundaries_pct": list(piecewise_boundaries()),
        "piecewise_settings": piecewise_settings(),
        "piecewise_membrane_throughout": bool(
            _pw_get("pw_membrane_throughout", True)),
        "piecewise_until": piecewise_until(),
        "piecewise_off": list(piecewise_off()),
        "piecewise_method": st.session_state.get("pw_method", "refined"),
        "piecewise_target_r2": float(_pw_get("pw_target_r2", 0.999)),
    }


def adopt_video(uploaded, widget_key):
    """
    Take an uploaded video as *the* video for this cell, from either uploader.

    There are two upload boxes on purpose: one beside the database controls,
    where you are already standing when you finish a cell, and one in the
    video tab, where the analysis happens. They are two doors into one room.
    Both call this, so both write the same file, and whichever you used, the
    video tab is analysing what you uploaded.

    The subtlety is knowing when a box has something new in it. Comparing
    against the loaded video's name is not enough: upload A here, then B
    there, and on the next rerun this box still holds A, sees that the loaded
    name is B, and helpfully replaces it. The two boxes then fight, one
    winning per rerun. So each box remembers what it last handed over, and
    only speaks up when its own contents change.
    """
    seen = st.session_state.setdefault("_video_seen", {})
    signature = None if uploaded is None else (uploaded.name, uploaded.size)
    if signature == seen.get(widget_key):
        return False
    seen[widget_key] = signature
    if uploaded is None:
        return False

    destination = os.path.join(tempfile.gettempdir(), f"afm_video_{uploaded.name}")
    with open(destination, "wb") as handle:
        handle.write(uploaded.getvalue())
    st.session_state["video_path"] = destination
    st.session_state["video_name"] = uploaded.name
    st.session_state["video_track"] = None
    # A frame pinned from the previous video is not a frame of this one.
    st.session_state["video_saved_frame"] = None
    st.session_state["video_saved_frame_index"] = None
    try:
        st.session_state["video_info"] = va.probe(destination)
    except Exception as exc:
        st.session_state["video_info"] = None
        st.error(f"Could not open that video: {exc}")
    return True


def video_loaded():
    """True when there is a video on disk this session can read."""
    path = st.session_state.get("video_path")
    return bool(
        not VIDEO_IMPORT_ERROR
        and path
        and st.session_state.get("video_info")
        and os.path.exists(path)
    )


def detection_at(frame_index):
    """
    The cell in one frame, as the video tab has it set up.

    Every reader of a frame goes through here, so the box drawn by hand and
    the scale taken from the probe apply everywhere, not only on the tab
    where they were set. The side panel next to the force curve was reading
    the raw detector and quietly ignoring both.

    Returns ``(frame, detection, nucleus, probe_box, scale_um_px)``.
    """
    frame, det, nucleus, probe_box = cached_detection(
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
    if frame is None:
        return None, None, None, None, None

    if st.session_state.get("video_manual_cell"):
        bx = st.session_state["video_cell_box_x"]
        by = st.session_state["video_cell_box_y"]
        det = va.manual_detection(frame, (bx[0], by[0], bx[1], by[1]))

    scale_um_px = None
    if st.session_state.get("video_use_probe_scale"):
        px = st.session_state["video_probe_box_x"]
        scale_um_px, _ = va.scale_from_probe(
            (px[0], 0.0, px[1], 0.0), frame.shape,
            float(st.session_state["video_probe_width_um"]),
        )
    return frame, det, nucleus, probe_box, scale_um_px


def morphology_frame_png():
    """
    The picture stored with the cell.

    A frame the person chose in the video tab wins over one recomputed here.
    They looked at that frame and approved it; regenerating it later from
    settings that have since changed can quietly store a different picture.
    """
    pinned = st.session_state.get("video_saved_frame")
    if pinned:
        return pinned
    if VIDEO_IMPORT_ERROR or not st.session_state.get("video_path"):
        return None
    path = st.session_state["video_path"]
    if not os.path.exists(path):
        return None
    try:
        index = int(st.session_state.get("video_preview_frame", 0))
        frame, det, nucleus, probe_box, _ = detection_at(index)
        if frame is None:
            return None
        annotated = va.annotate(frame, det, nucleus=nucleus, probe=probe_box)
        pad = float(st.session_state.get("video_crop_pad", 0.35))
        return png_bytes(
            va.crop(annotated, det, pad_frac=pad)
            if det and det.get("found") else annotated
        )
    except Exception:
        return None


def cell_record(fit, epsilon, force_N, fitted, date_acquired, stage_plan):
    """
    The record and the curve, built once.

    The archive and the download both go through here, so the file you keep
    on disk is the same record the archive holds rather than a near-copy that
    drifts from it.
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
        # The four-regime fit's own numbers: every coefficient, modulus and
        # anchor. Absent for the spring-network models.
        "piecewise": fit.get("piecewise"),
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


def send_cell_to_store(store, fit, epsilon, force_N, fitted, date_acquired,
                       model, stage_plan):
    """
    Package this analysis and write it into the cell's folder.

    Written against a store's save_cell rather than against one service, so
    the packaging stays in one place if another archive is added later.
    """
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


# The old name, kept so nothing that still calls it breaks.
send_cell_to_archive = send_cell_to_store


def send_cell_to_sheet(manager, fit, date_acquired):
    """
    Write one row into the Google Sheet.

    The sheet is the flat summary: one row per cell, the numbers you would
    plot across a population. The curve itself is too big for a cell and
    lives in OneDrive or in the downloaded zip.
    """
    terms = set(fit.get("terms") or ())
    lo, hi = fit.get("epsilon_range", (0.0, 0.0))
    e1 = fit.get("break_1")
    e2 = fit.get("break_2")
    segmented = fit.get("coupling") == "segmented"

    combination = ""
    if segmented:
        combination = "{}, {}".format(
            "membrane keeps stiffening" if fit.get("membrane") == "continue"
            else "membrane holds after ε₁",
            "cytoskeleton from zero" if fit.get("cyto_start") == "zero"
            else "cytoskeleton from ε₁",
        )

    # Blank for a cell type without sarcomeres: a number in these columns
    # for a myoblast would claim a measurement of something it does not have.
    sarcomere = {"relaxed": "", "at_max": ""}
    if st.session_state["cell_type"] in HAS_SARCOMERES:
        relaxed = float(st.session_state["sarcomere_nm"])
        stretch = (1.0 - min(float(hi), 0.999)) ** (
            -0.5 * float(st.session_state["sarcomere_spread"])
        )
        sarcomere = {
            "relaxed": round(relaxed, 1),
            "at_max": round(relaxed * stretch, 1),
        }

    piecewise = fit.get("piecewise") or {}
    if piecewise:
        edges = list(piecewise.get("boundaries_pct", []))
        combination = (
            "4 regimes, " + ", ".join(
                f"{n} = {float(v):.2f} %" for n, v in zip(EPS_NAMES, edges[1:4]))
            + (f", end {float(edges[4]):.1f} %" if len(edges) > 4 else "")
            + f" ({piecewise.get('boundary_source', '')}), C0-anchored"
            + (", membrane throughout" if piecewise.get("membrane_throughout")
               else "")
        )

    def span(term):
        """The stretch of deformation this modulus was actually measured on."""
        if term not in terms:
            return "not fitted"
        if piecewise:
            lo_hi = (piecewise.get("spans") or {}).get(term)
            if lo_hi:
                return f"{float(lo_hi[0]):.3f} to {float(lo_hi[1]):.3f}"
        if not segmented or e1 is None or e2 is None:
            return f"{lo:.3f} to {hi:.3f}"
        if term in ("membrane", "tension"):
            end = hi if fit.get("membrane") == "continue" else e1
            return f"{lo:.3f} to {end:.3f}"
        if term == "interior":
            begin = lo if fit.get("cyto_start") == "zero" else e1
            return f"{begin:.3f} to {hi:.3f}"
        return f"{e2:.3f} to {hi:.3f}"

    def modulus(key, term):
        # A term that was not fitted is written as 0, not blank: blank in a
        # spreadsheet column is ambiguous between zero and not measured, and
        # those mean different things once the rows are averaged.
        return round(float(fit.get(key, 0.0)), 6) if term in terms else 0

    return manager.append_cell_data(
        {
            "cell_id": st.session_state["cell_name"].strip(),
            "experiment_date": str(date_acquired),
            "cell_height": round(float(st.session_state["cell_height_um"]), 3),
            # Spring constant in N/m, which is the unit it is calibrated in.
            "spring_constant": round(float(st.session_state["spring_constant"]), 6),
            # Blank, not zero, where the cell type has no tension spring at
            # all: a zero in this column would read as "measured and found to
            # be nothing", which is a different claim.
            "T0": (
                modulus("T0_mN_m", "tension")
                if "tension" in terms_for(st.session_state["cell_type"]) else ""
            ),
            "T0_range": (
                span("tension")
                if "tension" in terms_for(st.session_state["cell_type"]) else ""
            ),
            "Em": modulus("Em_MPa", "membrane"),
            "Em_range": span("membrane"),
            "Ecx": (
                modulus("Ecx_kPa", "cortex")
                if "cortex" in terms_for(st.session_state["cell_type"]) else ""
            ),
            "Ei": modulus("Ei_kPa", "interior"),
            "Ei_range": span("interior"),
            "Ene": (
                modulus("Ene_MPa", "nucleus_shell")
                if "nucleus_shell" in terms_for(st.session_state["cell_type"])
                else ""
            ),
            "En": modulus("En_kPa", "nucleus"),
            "En_range": span("nucleus"),
            "membrane_areal": round(
                float(fit.get("membrane_areal_modulus", 0.0)) * 1e3, 5
            ),
            "model": fit.get("model_label") or st.session_state["model_kind"],
            "combination": combination,
            # Only the piecewise sharing has the apparent contact modulus of
            # regime 1. The perinuclear cytoskeleton is no longer one of the
            # components, so its column stays blank.
            "E_nc": "",
            "E_align": (round(float(fit.get("E_align_kPa", 0.0)), 6)
                        if piecewise else ""),
            "piecewise": (
                json.dumps(piecewise.get("flat", {}), default=float)
                if piecewise else ""
            ),
            "break_1": round(float(e1), 4) if e1 is not None else "",
            "break_2": round(float(e2), 4) if e2 is not None else "",
            "fit_range": f"{lo:.3f} to {hi:.3f}",
            "fit_quality": round(float(fit.get("r_squared", float("nan"))), 5),
            "adj_r_squared": round(float(fit.get("adj_r_squared", float("nan"))), 5),
            "chi_squared": float(f"{float(fit.get('chi_squared', float('nan'))):.5g}"),
            "chi_squared_reduced": round(
                float(fit.get("chi_squared_reduced", float("nan"))), 4
            ),
            "noise_sigma": float(f"{float(fit.get('noise_sigma', float('nan'))):.4g}"),
            "rmse_N": float(f"{float(fit.get('rmse', float('nan'))):.4g}"),
            "n_points": int(fit.get("n_points", 0)),
            "weighting": fit.get("weighting", st.session_state["weighting"]),
            "cell_radius": round(float(fit.get("R0", 0.0)) * 1e6, 4),
            "nucleus_radius": round(float(fit.get("R_nucleus", 0.0)) * 1e6, 4),
            "membrane_thickness": round(
                float(st.session_state["membrane_thickness_nm"]), 3
            ),
            "protein_coat": (
                round(float(st.session_state["protein_coat_nm"]), 2)
                if "tension" in terms_for(st.session_state["cell_type"]) else ""
            ),
            # Only for cell types that have sarcomeres. A number here for a
            # myoblast would be a measurement of something it does not have.
            "sarcomere_relaxed": sarcomere["relaxed"],
            "sarcomere_at_max": sarcomere["at_max"],
            "poisson": "{:.2f} / {:.2f}".format(
                float(st.session_state["poisson_membrane"]),
                float(st.session_state["poisson_interior"]),
            ),
            "force_curve_created": "Yes",
            "analysis_status": "Complete",
            "notes": st.session_state["cell_notes"],
            "video_link": st.session_state.get("video_link", ""),
            # Blank rather than 0 when it was never measured: a zero in a
            # height column reads as a measurement, and it is not one.
            "video_height_um": (
                round(float(st.session_state.get("video_height_um", 0.0)), 3)
                if float(st.session_state.get("video_height_um", 0.0)) > 0 else ""
            ),
            "video_comment": st.session_state.get("video_comment", ""),
        }
    )


def cell_bundle_zip(fit, epsilon, force_N, fitted, date_acquired, stage_plan):
    """The same record as the archive, as a zip the browser can download."""
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
        shell_thickness=float(settings.get("protein_coat_nm", 200.0)) * 1e-9,
        deep_uses_cell_radius=settings.get("cell_type") in DEEP_USES_CELL_RADIUS,
        sarcomere_length=float(settings.get("sarcomere_nm", 2100.0)) * 1e-9,
        confinement=float(settings.get("confinement", 0.0)),
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
        raise ArchiveError(f"{cell_id}: no stored curve to refit.")
    curve = pd.read_csv(io.StringIO(record["curve_csv"]))

    merged = dict(record.get("settings") or {})
    merged.update(settings)
    record["settings"] = merged
    model = model_from_record(record, curve)

    if (
        HAS_PIECEWISE
        and merged.get("fit_mode") == PIECEWISE_MODE
        and merged.get("cell_type") in PIECEWISE_CELL_TYPES
    ):
        # A cell stored from the four-regime fit is refitted with it, at
        # its own boundaries, guesses, bounds and probe.
        probe_um = float(merged.get("probe_diameter_um", 0.0) or 0.0)
        result = fit_piecewise(
            model.epsilon, model.force,
            boundaries_pct=merged.get("piecewise_boundaries_pct")
            or C2C12_BOUNDARIES_PCT,
            regimes=pw_regimes(),
            settings=effective_piecewise_settings(
                merged.get("piecewise_settings") or {},
                merged.get("piecewise_until") or {},
                merged.get("piecewise_off") or (),
            ),
            carry=(("K_shell",)
                   if merged.get("piecewise_membrane_throughout", True) else ()),
        )
        if not result.get("success"):
            raise ArchiveError(f"{cell_id}: {result.get('error', 'fit failed')}")
        result["moduli"] = piecewise_moduli(
            result,
            piecewise_geometry(model, probe_um=probe_um,
                               coat_nm=merged.get("protein_coat_nm", 200.0)),
            regimes=pw_regimes(),
        )
        fit = piecewise_as_fit(result, model, probe_um=probe_um,
                               settings=merged.get("piecewise_settings") or {},
                               off=merged.get("piecewise_off") or ())
        record.update({
            "coupling": "piecewise",
            "Em_MPa": float(fit["Em_MPa"]),
            "Ec_kPa": float(fit["Ei_kPa"]),
            "En_kPa": float(fit["En_kPa"]),
            "r_squared": float(fit["r_squared"]),
            "epsilon_min": float(fit["epsilon_range"][0]),
            "epsilon_max": float(fit["epsilon_range"][1]),
            "n_points": int(fit["n_points"]),
            "piecewise": fit["piecewise"],
            "refit_at": datetime.now().isoformat(timespec="seconds"),
        })
        record.pop("curve_csv", None)
        store.save_cell(record)
        return record

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
            use_tension="tension" in terms,
            use_nucleus_shell="nucleus_shell" in terms,
            use_cortex="cortex" in terms,
            weighting=settings.get("weighting", "uniform"),
            fit_offset=bool(settings.get("fit_offset", False)),
        )
    elif coupling == "series":
        fit = model.fit_series(lo, hi, terms=terms, weighting=settings.get("weighting", "uniform"))
    elif coupling in ("hybrid_ps", "hybrid_sp"):
        order = "parallel-then-series" if coupling == "hybrid_ps" else "series-then-parallel"
        scan = model.scan_crossover(lo, hi, terms=terms, order=order)
        if not scan.get("success"):
            raise ArchiveError(f"{cell_id}: hybrid fit failed.")
        fit = scan["best"]
    elif coupling == "auto":
        comparison = compare_couplings(model, lo, hi, terms=terms)
        if not comparison.get("success"):
            raise ArchiveError(f"{cell_id}: {comparison.get('error')}")
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
        raise ArchiveError(f"{cell_id}: {fit.get('error', 'fit failed')}")

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



def onedrive_instructions():
    """
    How to get OneDrive working, written out once and shown in two places.

    Every step here has been the one that stopped somebody. The account type
    at registration cannot be changed afterwards without re-registering, the
    public-client switch is off by default and its error names neither
    itself nor the switch, and the sheet-style mistake of skipping the
    consent step produces a message about something else entirely.
    """
    st.markdown(
        "**What you need first.** A Microsoft account with a OneDrive: a "
        "personal one (outlook.com, hotmail, live) or a work or school one "
        "(your university). Nothing is installed and nothing is paid for.\n\n"
        "**1 · Register the app, once.** Go to "
        "[portal.azure.com](https://portal.azure.com) → **Microsoft Entra "
        "ID** → **App registrations** → **New registration**.\n"
        "- Name: anything, for example `AFM cell analyzer`.\n"
        "- Supported account types: **Accounts in any organizational "
        "directory and personal Microsoft accounts**. This one matters. "
        "Registering it single-tenant is what produces `AADSTS50020`, whose "
        "wording blames the account rather than the registration.\n"
        "- Redirect URI: leave it empty.\n\n"
        "**2 · Copy the client id.** It is on the app's Overview page, "
        "called **Application (client) ID**. It is not a secret.\n\n"
        "**3 · Allow the sign-in this app uses.** **Authentication** → "
        "scroll to **Advanced settings** → **Allow public client flows** → "
        "**Yes** → Save. Without it the sign-in fails with "
        "`AADSTS7000218`.\n\n"
        "**4 · Ask for the permissions.** **API permissions** → **Add a "
        "permission** → **Microsoft Graph** → **Delegated permissions** → "
        "tick `offline_access`, `User.Read` and `Files.ReadWrite` for a "
        "personal account, or `Files.ReadWrite.All` for a work or school "
        "one. On a university tenant an administrator may have to press "
        "**Grant admin consent**; many tenants let you consent yourself the "
        "first time you sign in.\n\n"
        "**5 · Put the client id in this app's secrets.** In Streamlit Cloud: "
        "**⋮ → Settings → Secrets**. Running locally: "
        "`.streamlit/secrets.toml`. Paste this and reboot the app:"
    )
    st.code(
        '[onedrive]\n'
        'account = "personal"        # or "work"\n'
        'client_id = "the id from step 2"\n'
        'root_folder = "AFM cells"\n',
        language="toml",
    )
    st.markdown(
        "**6 · Sign in once, in the sidebar.** Open **☁️ OneDrive database** "
        "→ pick the account type → **Sign in to get a refresh token** → "
        "**Start sign-in**. Open the address it shows, type the code, sign "
        "in, then press **I have signed in**. It hands you a block with a "
        "`refresh_token` in it.\n\n"
        "**7 · Paste that block into Secrets**, replacing the one from step "
        "5, and reboot. Treat the refresh token like a password: it is a "
        "standing key to that OneDrive. Never put it in the repository, only "
        "in the Secrets box.\n\n"
        "**8 · Press Connect.** The sidebar says which account it reached. "
        "Cells you send arrive in the folder from step 5, one folder each, "
        "holding `record.json`, `curve.csv`, the morphology frame and the "
        "video if there is one, with an `index.csv` at the root so this tab "
        "can list them without opening every folder."
    )
    with st.expander("If it will not connect"):
        st.markdown(
            "- **AADSTS50020** · the registration is single-tenant, or a "
            "personal account is being signed in against a tenant id. "
            "Re-register with *any organizational directory and personal "
            "Microsoft accounts*, and remove `tenant` from the secrets for a "
            "personal account, which does not belong to a directory.\n"
            "- **AADSTS7000218** · step 3 was skipped. Allow public client "
            "flows.\n"
            "- **AADSTS65001** · nobody has consented yet. Sign in again, or "
            "ask an administrator for consent on a university tenant.\n"
            "- **The scope was refused** · a personal account takes "
            "`Files.ReadWrite`, not `Files.ReadWrite.All`. The account-type "
            "dropdown in the sidebar sets both the endpoint and the scopes, "
            "so change it there rather than in secrets.\n"
            "- **It worked and then stopped** · a refresh token expires "
            "after about 90 days unused. Sign in again and paste the new "
            "block.\n"
            "- **No [onedrive] section in secrets** · the app cannot see "
            "step 5. On Streamlit Cloud the secrets box needs a reboot "
            "before the app reads it."
        )


# ================================================================ sidebar ==

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    hint("These apply everywhere in the app.")

    previous_type = st.session_state.get("_applied_cell_type")
    st.selectbox(
        "Cell type",
        SELECTABLE_CELL_TYPES,
        key="cell_type",
        help="Sets geometry, bilayer thickness, the plausibility bands used for "
        "warnings, and the starting fit windows. Everything stays editable.",
    )
    if previous_type != st.session_state["cell_type"]:
        apply_cell_type(st.session_state["cell_type"])
        st.session_state["_applied_cell_type"] = st.session_state["cell_type"]
        st.session_state["_cell_type_applied_note"] = True

    # What changing the cell type actually did. It changes the geometry, the
    # materials and the whole fitting story at once, and a change that large
    # happening silently behind one dropdown is how a cell ends up fitted as
    # something it is not.
    _here = st.session_state["cell_type"]
    _materials = "  ·  ".join(plain_name(term, _here)
                              for term in terms_for(_here))
    if st.session_state.pop("_cell_type_applied_note", False):
        st.success(
            f"**Set up for {_here.split(' (')[0]}.**  Height "
            f"{st.session_state['cell_height_um']:.1f} µm, membrane "
            f"{st.session_state['membrane_thickness_nm']:.0f} nm, "
            f"{len(terms_for(_here))} materials: {_materials}. Any curve "
            f"already loaded is refitted.",
            icon="✅",
        )
    else:
        st.caption(
            f"**{_here.split(' (')[0]}:** {_materials}.  Height "
            f"{st.session_state['cell_height_um']:.1f} µm."
        )
    if st.button("↻ Re-apply this cell type's defaults", **STRETCH):
        apply_cell_type(_here)
        st.session_state["_cell_type_applied_note"] = True
        st.session_state.pop("_auto_picked", None)
        st.rerun()

    st.caption(
        "**Set once per session.** Everything in this sidebar describes the "
        "cell and the instrument, not this particular fit. What changes from "
        "curve to curve lives on the page, under the fit button."
    )

    with st.expander("📐 Cell geometry", expanded=True):
        # The height itself is on the page, in Cell information, because it
        # is a measurement of this cell rather than a setting for the
        # session. What is left here is what follows from it.
        st.caption(
            f"Cell height h₀ = {st.session_state['cell_height_um']:.2f} μm, "
            "set with the rest of the cell's details on the page."
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

    with st.expander(
        "📐 Cell shape and the incompressible interior", expanded=False
    ):
        st.selectbox(
            "What shape is the cell",
            ["Sphere (a rounded cell)", "Belt cylinder (a rod lying down)"],
            key="cell_shape",
            help="A cell rounded up on the dish is a sphere. An adult "
            "cardiomyocyte is a rod lying on its side, which is a different "
            "problem: it has less room to spread as it is flattened, so it "
            "stiffens faster.",
        )
        st.number_input(
            "Confinement exponent q",
            min_value=0.0, max_value=3.0, step=0.05, format="%.2f",
            key="confinement",
            help="Every term is multiplied by (1−ε)^−q. q = 0 is the classic "
            "Lulevich model, right near contact and too soft deep in. "
            "q = 0.5 is a sphere spreading freely at constant volume. q = 3 "
            "is a rod whose cross-section cannot deform at all. A real "
            "attached cell sits between, because it sheds some volume as it "
            "is squeezed. Measure it rather than assume it.",
        )
        st.caption(
            "**What q is.** It is the interior refusing to be compressed. "
            "Every spring in the model is a small-strain law, and past about "
            "half the cell's height a real cell stiffens faster than any of "
            "them can: on the WT cardiomyocyte curve the measured exponent "
            "climbs from 3 near ε = 0.2 to 5.3 by ε = 0.7, and nothing built "
            "from ε, ε³ᐟ² and ε³ follows that. The cause is not a new "
            "material. The cell keeps its volume, so what the plates take out "
            "of its height has to go somewhere sideways, and q says how "
            "easily it can go. Fitting with q free instead of q = 0 improves "
            "χ²/dof about 200-fold."
        )
        st.caption(
            "**Why it replaces a deep spring.** A stiff layer switching on "
            "part-way and an interior running out of room look alike on a "
            "curve, and only one of them is a material. Fitting the four "
            "measured WT curves with the deep spring removed and q measured "
            "instead changes χ²/dof by less than a tenth. So the deep spring "
            "is not in the cardiomyocyte model: it was describing "
            "incompressibility under another name."
        )
        if st.session_state.get("data") is not None:
            found = st.session_state.get("confinement_scan")
            if st.button("📐 Estimate q: argmin χ²(q)", **STRETCH):
                st.session_state["_want_confinement_scan"] = True
                st.rerun()
            if found and found.get("success"):
                st.success(
                    f"q = {found['q']:.2f} "
                    f"(anything from {found['q_low']:.2f} to "
                    f"{found['q_high']:.2f} fits about as well)"
                )
                st.caption(
                    f"R² {found['r_squared']:.6f} against "
                    f"{found['baseline']['r_squared']:.6f} at q = 0; "
                    f"χ²/dof {found['chi_squared_reduced']:.4g} against "
                    f"{found['baseline']['chi_squared_reduced']:.4g}."
                )
                if st.button("Apply q", **STRETCH):
                    st.session_state["_pending_settings"] = {
                        "confinement": round(float(found["q"]), 2)
                    }
                    st.rerun()

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
        if st.session_state["cell_type"] in HAS_SARCOMERES:
            # The in-plane membrane spring and its genotype used to be set
            # here. They are gone on purpose: the plain cardiomyocyte, four
            # materials and no extras, has to be settled before a membrane
            # protein is added on top of it. Nothing is lost — the term and
            # the prefactor are still in the model layer, so putting the
            # control back is one entry in OPTIONAL_TERMS.
            st.number_input(
                "Relaxed sarcomere length (nm)",
                min_value=500.0,
                max_value=5000.0,
                step=50.0,
                key="sarcomere_nm",
                help="2100 nm for cardiac muscle. Nothing in the fit uses "
                "it. It converts the deformation into a sarcomere length, "
                "which is what says whether the myofibrils were still in "
                "their working range over the stretch you fitted.",
            )
            st.slider(
                "How freely the cell spreads sideways", 0.0, 1.0, step=0.05,
                key="sarcomere_spread",
                help="A squashed cell keeps its volume, so it spreads. How "
                "much of that spreading runs along the myofibrils decides "
                "how far the sarcomeres are pulled out. 1 is a cell free to "
                "spread in every direction, the most they can lengthen; 0 "
                "is a cell held at its ends, where they do not lengthen at "
                "all. An attached cell is somewhere between, so read the "
                "two ends as bounds.",
            )
        st.slider("Poisson ratio, membrane νₘ", 0.0, 0.5, step=0.01, key="poisson_membrane")
        st.slider("Poisson ratio, cytoskeleton νc", 0.0, 0.5, step=0.01, key="poisson_interior")
        st.caption("0.5 = incompressible, the usual choice for living cells.")

    deep_name = plain_name("nucleus")
    # A cell type with no deep element has nothing to set here, and a panel
    # of settings for a term that is not in the model is not a harmless
    # extra: it is four numbers a reader will assume were measured.
    with st.expander(
        f"🟣 {deep_name}" if has_deep_term()
        else "🫗 One incompressible interior (no deep element)"
    ):
        if not has_deep_term():
            st.caption(
                f"A {st.session_state['cell_type'].split(' (')[0].lower()} is "
                f"modelled as a shell around a single incompressible "
                f"interior, so there is no deep spring and nothing to set "
                f"here. What a deep spring would have absorbed is the cell "
                f"running out of room, and that is the confinement exponent "
                f"q under **Cell shape**, measured from the curve rather "
                f"than assumed."
            )
        if has_deep_term():
            st.radio(
                f"{deep_name} radius",
                ["From cell radius", "Manual"],
                key="nucleus_radius_mode",
                horizontal=True,
            )
            if st.session_state["nucleus_radius_mode"] == "From cell radius":
                st.slider(
                    f"{deep_name} radius / R₀", 0.10, 0.90, step=0.01,
                    key="nucleus_fraction"
                )
            else:
                st.number_input(
                    f"{deep_name} radius (μm)", min_value=0.1, max_value=50.0, step=0.05,
                    format="%.2f", key="nucleus_radius_um",
                )
            st.slider(f"Poisson ratio, {deep_name.lower()} νₙ", 0.0, 0.5, step=0.01,
                      key="poisson_nucleus")
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

    st.caption(
        "Below here is presentation and plumbing, not physics."
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
            st.color_picker(
                "Fit", key="fit_color",
                disabled=bool(st.session_state.get("recolour_on_range", True)),
            )
        st.checkbox(
            "New fit colour each time the range changes",
            key="recolour_on_range",
            help="On, the fitted line steps through a set of colours as you "
            "move the fitted range, so two plots made from different ranges "
            "are never the same colour. Off, it stays the colour picked "
            "above.",
        )
        if st.session_state.get("recolour_on_range", True):
            st.caption(
                f"Fit is drawn in `{fit_line_colour()}` for ε = "
                f"{float(st.session_state.get('window_start', 0.0)):.3f} to "
                f"{float(st.session_state.get('window_end', 1.0)):.3f}."
            )
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
        st.caption(
            "The data and the fitted curve are always drawn. Anything else "
            "on the figure was sent there from the page, and is listed "
            "under it."
        )

        st.markdown("**Panels beside the curve**")
        st.checkbox(
            "Cell diagram",
            key="show_schematic",
            help="The model drawn as a mechanics schematic at the deformation "
            "you select: a fixed support, the cantilever, and the elements "
            "between them.",
        )
        st.radio(
            "Draw it as",
            ["Mechanics schematic", "Balloon with a spring inside"],
            key="schematic_style", horizontal=True,
            disabled=not st.session_state["show_schematic"],
            help="The schematic says what the model computes: a support, a "
            "cantilever, and the elements between them. The balloon says "
            "what it is about: a taut skin holding an interior, flattened "
            "between two plates and spreading sideways because it keeps its "
            "volume. Same numbers, same deformation. For a cardiomyocyte "
            "the balloon holds fluid rather than a spring, and the "
            "myofibrils lie along the cell, because nothing in that model "
            "singles out a nucleus.",
        )
        st.checkbox(
            "Moduli under the diagram", key="show_schematic_moduli",
            help="The Eₘ, E_c and Eₙ values printed under the cell diagram. The "
            "same numbers are in the results table.",
        )

    with st.expander("☁️ OneDrive database", expanded=False):
        if ONEDRIVE_IMPORT_ERROR:
            st.error(onedrive_load_problem())
            st.caption(f"Python said: {ONEDRIVE_IMPORT_ERROR}")
            with st.expander("📖 The setup steps, for when it does load"):
                onedrive_instructions()
        else:
            st.caption(
                "Storage you already have, and you can authorise it yourself. "
                "Files go to a folder per cell."
            )
            with st.expander("📖 How to set this up, step by step"):
                onedrive_instructions()
            st.selectbox(
                "Account type",
                list(onedrive_store.ACCOUNTS.keys()),
                key="onedrive_account",
                format_func=lambda key: onedrive_store.ACCOUNTS[key]["label"],
                help="This decides both the sign-in endpoint and the "
                "permissions asked for, and the two are not interchangeable. "
                "A personal account does not live in a directory, so it "
                "cannot sign in against a tenant id, and it uses "
                "Files.ReadWrite rather than the work-account "
                "Files.ReadWrite.All.",
            )
            st.text_input("Folder for the cells", key="onedrive_root")
            # Which half of the setup is done. A client_id with no token is
            # the normal halfway state, and it deserves an instruction
            # rather than the raw error the token request would raise.
            try:
                _od = dict(st.secrets.get("onedrive", {}))
            except Exception:
                _od = {}
            _has_id = bool(_od.get("client_id"))
            _has_key = bool(_od.get("refresh_token") or _od.get("client_secret"))
            if _has_id and not _has_key:
                st.info(
                    "**One step left.** The client id is in secrets and "
                    "nobody has signed in with it yet, so there is nothing "
                    "to connect with. Open **Sign in to get a refresh "
                    "token** below, press **Start sign-in**, and follow it "
                    "through; it ends by handing you a block to paste back "
                    "into Secrets. Pressing Connect before that will always "
                    "say there are no credentials."
                )

            if st.button("🔌 Connect to OneDrive", **STRETCH):
                store = onedrive_store.store_from_secrets(
                    st, st.session_state["onedrive_root"] or None
                )
                if store is not None:
                    # The picker wins over whatever is in secrets, so the fix
                    # is one dropdown rather than an edit and a reboot.
                    store.account = st.session_state["onedrive_account"]
                    chosen = onedrive_store.settings_for(store.account)
                    store.tenant = chosen["tenant"]
                    store.scopes = chosen["scopes"]
                if store is None:
                    st.error(
                        "No [onedrive] section in secrets. Add client_id and "
                        "tenant, then sign in below to get a refresh token."
                    )
                else:
                    status = store.check()
                    if status["ok"]:
                        st.session_state["onedrive_store"] = store
                        st.success(f"Connected: {status['detail']}")
                    elif "No OneDrive credentials" in str(status["detail"]):
                        st.warning(
                            "Nothing to connect with yet: secrets have the "
                            "client id but no refresh token. That token "
                            "comes from **Sign in to get a refresh token** "
                            "just below — do that first, paste the block it "
                            "gives you into Secrets, reboot, and then press "
                            "Connect."
                        )
                    else:
                        st.error(status["detail"])
            if st.session_state.get("onedrive_store"):
                st.caption(
                    f"Connected ✓ · {st.session_state['onedrive_store'].auth_method()}"
                )

            with st.expander("Sign in to get a refresh token",
                             expanded=_has_id and not _has_key):
                st.caption(
                    "Do this once. It needs client_id and tenant in secrets; "
                    "everything else happens in your browser."
                )
                if st.button("Start sign-in", **STRETCH):
                    try:
                        config = dict(st.secrets.get("onedrive", {}))
                        flow = onedrive_store.begin_device_login(
                            config.get("client_id"),
                            account=st.session_state["onedrive_account"],
                        )
                        st.session_state["_device_login"] = flow
                    except Exception as exc:
                        st.error(str(exc))
                flow = st.session_state.get("_device_login")
                if flow:
                    st.markdown(
                        f"1. Open **{flow.get('verification_uri', '')}**\n\n"
                        f"2. Enter the code **`{flow.get('user_code', '')}`**\n\n"
                        f"3. Sign in, then press the button below."
                    )
                    if st.button("I have signed in", type="primary", **STRETCH):
                        try:
                            config = dict(st.secrets.get("onedrive", {}))
                            payload = onedrive_store.poll_device_login(
                                config.get("client_id"),
                                flow.get("device_code"),
                                account=st.session_state["onedrive_account"],
                            )
                        except Exception as exc:
                            payload = None
                            st.error(str(exc))
                        if payload is None:
                            st.info("Not signed in yet. Finish in the browser "
                                    "and press again.")
                        else:
                            st.success("Signed in. Copy this into your secrets:")
                            st.code(
                                "[onedrive]\n"
                                f'account = "{st.session_state["onedrive_account"]}"\n'
                                f'client_id = "{dict(st.secrets.get("onedrive", {})).get("client_id", "")}"\n'
                                f'refresh_token = "{payload.get("refresh_token", "")}"\n'
                                f'root_folder = "{st.session_state["onedrive_root"] or "AFM cells"}"',
                                language="toml",
                            )
                            st.caption(
                                "Paste it into Settings → Secrets, reboot, "
                                "then press Connect. Treat it like a password."
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
                    with st.spinner("Opening the sheet…"):
                        manager = initialize_sheets_manager(
                            spreadsheet_id=st.session_state["sheet_id"] or None
                        )
                    st.session_state["gs_manager"] = manager
                    if manager:
                        st.success("Connected.")
                    else:
                        # It used to say nothing at all here, which is how a
                        # failed connection looked exactly like a successful
                        # one until the send button downstream stayed grey.
                        st.error(
                            "**Could not open the sheet.** Nothing is "
                            "connected, so the send buttons stay disabled. "
                            "In the order these actually go wrong:\n\n"
                            "1. The sheet is not shared with the service "
                            "account. Open it in Google Sheets, press "
                            "Share, and add the address below as an "
                            "**Editor**. This is the one that is usually "
                            "missed, and it fails with a message about "
                            "Drive storage quota, which sounds like a "
                            "different problem.\n"
                            "2. The id in the box above is not that "
                            "sheet's. Paste the whole URL if in doubt.\n"
                            "3. `[google_sheets_credentials]` is missing "
                            "from Secrets, or the Sheets and Drive APIs are "
                            "not enabled for that project."
                        )
                        email = ""
                        try:
                            email = st.secrets["google_sheets_credentials"].get(
                                "client_email", ""
                            )
                        except Exception:
                            email = ""
                        if email:
                            st.caption("Share the sheet with this address:")
                            st.code(email, language="text")
                        else:
                            st.caption(
                                "No `client_email` in "
                                "`[google_sheets_credentials]`, so the "
                                "credentials themselves are what is "
                                "missing."
                            )
                if st.session_state["gs_manager"]:
                    manager = st.session_state["gs_manager"]
                    describe = getattr(manager, "describe", None)
                    st.caption(
                        f"Connected ✓ · {describe()}" if callable(describe)
                        else "Connected ✓ · rows go to the first tab"
                    )
                    url = manager.get_spreadsheet_url()
                    if url:
                        st.caption(f"[Open the sheet]({url})")
                    if callable(getattr(manager, "check", None)) and st.button(
                        "🧪 Check it can write", **STRETCH,
                        help="Reads the sheet, reports what it finds, and "
                             "tries one write, so a Viewer-only share is "
                             "caught before a row goes missing.",
                    ):
                        with st.spinner("Looking at the sheet…"):
                            ok, said = manager.check()
                        (st.success if ok else st.error)(said)
                    if st.button("↕️ Put the columns in order", **STRETCH):
                        with st.spinner("Rewriting the header…"):
                            ok, message = st.session_state["gs_manager"].reorder_columns()
                        (st.success if ok else st.error)(message)
                    st.caption(
                        "Rewrites the sheet so each modulus is followed by its "
                        "range and the video link is last. Rows are remapped by "
                        "column name, so nothing moves under the wrong heading, "
                        "and columns of your own are kept."
                    )
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

# Every tab is still built and still filled, so none of that code can rot.
# The two that are not wanted on the bar for now are hidden by hiding their
# buttons: a tab whose button is gone cannot be opened, and the code that
# fills it goes on working untouched. Set these to True to bring them back.
SHOW_VIDEO_TAB = False
SHOW_DATABASE_TAB = False

# The two tabs that can be hidden go last. They are hidden by CSS on the
# n-th tab button, and a tab bar inside a page (the fit's own tabs) has
# fewer than seven buttons, so its buttons are never the ones hidden.
(
    tab_analysis, tab_cells, tab_explore, tab_igor, tab_results,
    tab_export, tab_video, tab_db,
) = st.tabs(
    [
        "📊 Force curve analysis",
        "📚 All cells",
        "📈 Log curve and boundaries",
        "🔧 Create curve (Igor)",
        "📈 Results",
        "💾 Export",
        "🎥 Compression video",
        "📋 Database",
    ]
)

_hidden = [
    index for index, wanted in ((7, SHOW_VIDEO_TAB), (8, SHOW_DATABASE_TAB))
    if not wanted
]
if _hidden:
    st.markdown(
        "<style>"
        + "".join(
            f'button[data-baseweb="tab"]:nth-of-type({n}) {{display: none;}}'
            for n in _hidden
        )
        + "</style>",
        unsafe_allow_html=True,
    )


# ================================================ TAB 1: analysis workflow ==

with tab_analysis:
    section("1 · Cell information")

    # Working through a plate means several cells per session, often
    # alternating between types. Without this the previous cell's name, fit,
    # video and search results carry over silently into the next one, which
    # is how a modulus ends up filed under the wrong cell.
    new1, new2 = st.columns([1, 3])
    with new1:
        if st.button("🆕 Start a new cell", **STRETCH):
            st.session_state["_start_new_cell"] = True
            # The cell's own fields are cleared by the block at the top of
            # the script; everything else has to be carried through the
            # rerun by hand, because half of it lives in widgets this run
            # will not reach.
            rerun_keeping_settings(forget=NEW_CELL_CLEARS)
    with new2:
        st.caption(
            "Clears the curve, the fit, the video and the name, and leaves "
            "the geometry, the display settings and the database connections "
            "alone. Use it between cells, and whenever you switch cell type."
        )
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
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.number_input(
            "Cell height h₀ (μm)",
            min_value=0.1, max_value=100.0, step=0.01, format="%.2f",
            key="cell_height_um",
            help="Initial (undeformed) height. It belongs with the cell, not "
            "with the display settings: it sets both the geometry "
            "prefactors and the conversion from relative deformation to "
            "indentation, so it is measured per cell.",
        )
    with c6:
        st.number_input(
            "Probe sphere diameter (µm)",
            min_value=0.0, max_value=200.0, step=1.0, format="%.1f",
            key="probe_diameter_um",
            help="The microsphere glued to the cantilever. The four-regime "
            "C2C12 fit uses it to correct its Hertzian (ε^1.5) moduli for "
            "the curvature of the sphere; the other models treat the probe "
            "as a flat plate and only record it. 0 means not recorded, "
            "which is treated as flat.",
        )
    with c7:
        st.number_input(
            "Approach speed (µm/s)",
            min_value=0.0, max_value=500.0, step=0.5, format="%.2f",
            key="approach_speed_um_s",
            help="Recorded with the cell. A cell is viscoelastic, so the "
            "moduli belong to the speed they were measured at and only "
            "compare with cells squashed at the same one.",
        )
    with c8:
        st.selectbox(
            "Compression type", COMPRESSION_TYPES, key="compression_type",
            help="How the probe met the cell. Recorded with the cell and "
            "written into the spreadsheet row: a head-on squash and one off "
            "to the side are not the same measurement.",
        )
    c9, _c10 = st.columns([1, 3])
    with c9:
        st.text_input("Operator", placeholder="initials", key="operator")
    st.text_input("Notes", placeholder="passage, treatment, anything worth keeping",
                  key="cell_notes")
    if (
        st.session_state["probe_diameter_um"]
        and st.session_state["probe_diameter_um"] < 2 * st.session_state["cell_height_um"]
    ):
        st.caption(
            f"⚠️ A {st.session_state['probe_diameter_um']:.0f} µm sphere on a "
            f"{st.session_state['cell_height_um']:.0f} µm cell is not a flat "
            f"plate. This model assumes the cell is squashed between two "
            f"flat surfaces; with a sphere this size the contact is curved "
            f"and the moduli will be overestimates. A sphere of 40 µm or "
            f"more on a cell this tall is close enough to flat."
        )
    hint(
        f"Cell height h₀ = {st.session_state['cell_height_um']:.2f} μm scales "
        "both moduli directly, so it is measured per cell and kept with the "
        "cell. The radius it implies, and everything else geometric, is in "
        "the sidebar under **Cell geometry**."
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

        st.caption(
            f"**{epsilon.size:,} points** · ε {epsilon.min():.3f} to "
            f"{epsilon.max():.3f} · peak {float(peak_display):.4g} "
            f"{unit_label}"
            + (f" · {data['n_dropped']} blank or non-numeric rows skipped"
               if data["n_dropped"] else "")
        )

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

        # One layout, so this is settled here and never asked again.
        guided_now = True

        # ----------------------------------------------------------- model ---

        model = build_model(epsilon, force_N)
        auto_range = model.auto_detect_elastic_range()
        rupture = model.results.get("rupture", {})

        eps_hi_data = float(epsilon.max())
        # ---------------------------------------------------- pick it all
        # As soon as a curve is loaded, choose the range, the elements and
        # the picture of the cell, and say which was chosen. Waiting for a
        # button meant every curve started from settings that belonged to
        # the last one, and the first thing on screen was a fit nobody had
        # asked for. Keyed on the data and the cell type, so it happens once
        # per curve and never fights a choice made by hand afterwards.
        auto_key = (
            data["source"], int(epsilon.size), round(float(force_N[-1]), 15),
            st.session_state["cell_type"],
        )
        if st.session_state.get("_auto_picked") != auto_key:
            st.session_state["_auto_picked"] = auto_key
            # A new curve starts the fit colour over at black. The range it
            # is given here was chosen by the app, not by you, so it is not
            # a change of range and must not count as one.
            pending = {"_fit_colour_range": None, "_fit_colour_step": 0}
            notes = []
            # A new curve arrives with its boundaries drawn on it. They are
            # the thing a person checks first against the shape of the
            # curve, and a figure that has to be asked for them is one that
            # gets looked at without them.
            st.session_state["plot_layers"] = [
                {"kind": "boundaries", "label": boundaries_label(),
                 "payload": boundaries_payload()},
            ]

            # A new curve arrives with its components ticked. They are a
            # property of the cell type, not of the last curve: a search
            # that dropped the nucleus on the cell before this one was an
            # answer about that cell, and carrying it forward silently is
            # how a curve gets fitted with a model nobody chose for it.
            wanted = DEFAULT_TERMS_BY_TYPE.get(
                st.session_state["cell_type"], {}
            )
            here_now = terms_for(st.session_state["cell_type"])
            if not fixed_cell_on():
                for term in ALL_TERMS:
                    pending[f"use_{term}"] = bool(
                        wanted.get(term, False) and term in here_now
                    )
                # The same, for the four the carried-forward fit is made
                # of. The curve is fitted with all four the moment it
                # loads, so all four arrive ticked: a box that says
                # otherwise is the page disagreeing with its own fit.
                for name in PW_SWITCHABLE:
                    pending[f"pw_use_{name}"] = bool(
                        DEFAULTS.get(f"pw_use_{name}", True))
                pending["component_order"] = list(COMPONENT_ORDER_DEFAULT)
                # And with the component met first acting throughout, which
                # is the model this page is set up with. The fit gives it up
                # only if nothing else reaches the target, and says so.
                pending["pw_membrane_throughout"] = True
                st.session_state["component_search"] = None

            window = (
                model.suggest_window() if hasattr(model, "suggest_window") else {}
            )
            if window.get("success"):
                start = (round(window["epsilon_min"], 4)
                         if window.get("bad_contact") else 0.0)
                end = round(window["epsilon_max"], 4)
                pending["window_end"] = end
                pending["window_start"] = start
                pending["window_combined"] = (start, end)
                notes.append(window["why_start"])
                if window.get("bad_contact"):
                    notes.append("**the approach misbehaved near contact**")
                # Kept so the page can say the range on screen is the
                # suggested one, and put it back after it has been moved.
                # A default nobody is told about is indistinguishable from
                # an arbitrary number.
                st.session_state["_suggested_window"] = {
                    "start": start, "end": end,
                    "why_start": window.get("why_start", ""),
                    "why_end": window.get("why_end", ""),
                    "bad_contact": bool(window.get("bad_contact")),
                    "n_points": int(window.get("n_points", 0)),
                }
            else:
                st.session_state["_suggested_window"] = None

            # The boundaries a new curve starts from are this cell type's,
            # not a search result. For a C2C12 that is the membrane holding
            # at ε₁ and the nucleus met at ε₂, which is what the literature
            # says and what somebody opening the app expects to see. The
            # search still exists; it is a button now, so the numbers on
            # screen are always either a stated default or something that
            # was asked for.
            pending.update(default_boundaries(st.session_state["cell_type"]))
            # The four-regime fit starts every curve from the specification
            # too, and forgets the last curve's boundary search.
            pending.update({key: DEFAULTS[key] for key in PW_BOUNDARY_KEYS})
            pending["pw_until"] = {}
            st.session_state["pw_boundary_search"] = None
            st.session_state["pw_placements"] = None
            st.session_state["pw_selected"] = None
            # And its boundaries are then found inside the C2C12
            # constraints the first time the four-regime fit draws it.
            st.session_state["_pw_auto_found"] = None
            # The spring-network page fits a new curve once, on arrival;
            # after that only ▶ Fit & plot does.
            pending["_fit_from_curve"] = True

            # No search here. Loading a curve used to run the whole
            # comparison and apply its winner, so the boundaries on screen
            # were a result nobody had asked for and the defaults were never
            # seen. The search is two buttons now: "Find boundaries from the
            # data" moves ε₁ and ε₂, and "Fit this cell" compares the
            # arrangements. Loading is also several seconds quicker for it.

            if pending:
                st.session_state["_auto_notes"] = notes
                st.session_state["_pending_settings"] = pending
                st.rerun()


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

        section("3 · Nonlinear fitting")
        piecewise_mode = fit_mode_control()
        if piecewise_mode:
            # The four-regime fit is the whole of this step. Nothing below
            # it in the spring-network flow applies, so none of it is drawn.
            segmented = False
            fit, fitted, stage_plan, pw_result = piecewise_section(
                model, epsilon, force_N, rupture,
            )
            # Not written to _last_fit: that is the spring-network flow's
            # memory of its own last fit, and it redraws from it in shapes
            # this result does not have.
            if fit is not None:
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
                    "membrane_N": None,
                    "interior_N": None,
                    "nucleus_N": None,
                    "fit": fit,
                    "fit_windows": [],
                    # The four-regime result itself, so the Results tab can
                    # draw the same component curves as the analysis tab.
                    "piecewise_result": pw_result,
                    "piecewise_off": list(fit["piecewise"].get("components_off") or ()),
                    "source": data["source"],
                    "timestamp": datetime.now(),
                }
        else:
            # One page, not two. There used to be a plain-language mode and an
            # every-setting mode, which meant every change had to be made twice
            # and half the app was only reachable by knowing a radio existed.
            # This is that page: the choices, the button, the curve, the numbers
            # and the maths, with everything the fit assumed one click away in
            # the sidebar. `guided` stays as the flag the layout is written
            # against rather than being spelled out of two hundred branches.
            guided = True

            # The graph and the results on top, drawn from the fit last
            # applied; under them the control board with every parameter of
            # the fit and the one button, ▶ Fit & plot. Changes wait on the
            # board until it is pressed.
            fit_block = results_area = verdict_slot = boundaries_slot = None
            board_status = None
            fit_pressed = refresh_pressed = False
            if guided:
                # The same strip of component moduli the regime-by-regime
                # page opens with, above the curve and full width.
                strip_slot = st.container()
                plot_col, res_col = st.columns([1.75, 1], gap="large")
                with plot_col:
                    verdict_slot = st.container()
                    curve_slot = st.container()
                    video_slot = st.container()
                with res_col:
                    results_area = st.container()
                fit_block = st.container(border=True)
                explainer_box = st.expander(
                    "📘 How the fit is done — the maths, step by step",
                    expanded=False,
                )
                diagnostics_box = st.expander(
                    "🔍 The working, in detail", expanded=False
                )
                fit_block.__enter__()
                # The same board, in the same three steps, as the
                # regime-by-regime fit: what the cell is made of, where its
                # parts take over, then the one button.
                st.markdown("#### 🎛️ Fitting")
                board_status = st.empty()
                st.markdown("**Step 1 · What the cell is made of**")
                share_col, model_col = st.columns([1.2, 1], gap="medium")
                with share_col:
                    load_sharing_control()
                with model_col:
                    st.latex(r"F(\varepsilon) = \sum_k a_k E_k\,"
                             r"g_k(\varepsilon)")
                st.markdown("**The order they are met in**")
                component_order_control()
                # The components are the rest of step 1: they are what the
                # plot draws, one tick per curve and one bar per range.
                # Filled after the tiles, because the range the bars live
                # inside is set there, but drawn here.
                components_area = st.container()
                st.markdown("**Step 2 · Where its parts take over**")
                tile2, tile3 = st.columns([1.25, 1], gap="medium")

            if guided:
                names = components_for(st.session_state["cell_type"])
                here = terms_for(st.session_state["cell_type"])

                # The range first: it decides which points exist at all, and
                # every component below is placed inside it.
                with tile2:
                    st.latex(r"\varepsilon \in [\varepsilon_{min},\,"
                             r"\varepsilon_{max}]")
                    st.caption("which points are fitted at all")
                    guided_lo, guided_hi = epsilon_range_control(
                        "window_start", "window_end", 0.0, eps_hi_data, step,
                        label="Fitted range, ε",
                        help_text="Drag either end, or type it below. The near "
                                  "end is normally 0, because the model "
                                  "describes a cell from first contact.",
                    )
                    inside = int(
                        ((epsilon >= guided_lo) & (epsilon <= guided_hi)).sum()
                    )
                    st.caption(
                        f"ε {guided_lo:.3f} to {guided_hi:.3f} · {inside} of "
                        f"{epsilon.size} points"
                        + (f" · rupture near ε = {rupture['epsilon']:.3f}"
                           if rupture.get("method") == "force-drop"
                           and rupture.get("epsilon") is not None else "")
                    )
                    suggested_range_note(guided_lo, guided_hi)
                    layer_checkbox("range", "Shade the fitted range on the plot",
                                   key="layer_range")
                    if guided_lo > 0:
                        st.caption(
                            "⚠️ Not starting from zero. The membrane term is "
                            "measured from first contact, so a start above 0 "
                            "only makes sense when the approach itself "
                            "misbehaved."
                        )

                chosen = active_terms()
                with tile3:
                    st.latex(r"0 \le \varepsilon_1 < \varepsilon_2 \le "
                             r"\varepsilon_{max}")
                    st.caption("where the components take over")
                    st.radio(
                        "ε₁, ε₂ on ▶ Fit & plot", list(LEGACY_EPS_MODES),
                        format_func=LEGACY_EPS_MODES.get, key="legacy_eps_mode",
                        help="As set: fit at the boundaries on the board. "
                        "Estimate: read them off the log curve inside the "
                        "cell type's bands, keep the placement that predicts "
                        "best, then fit.",
                    )
                    boundaries_slot = optimisation_controls(
                        model, guided_lo, guided_hi, chosen)

                components_area.__enter__()
                st.markdown("Components and the ranges they act over "
                            "— what is ticked is what is fitted and drawn")
                chosen = active_terms()
                if not chosen:
                    st.warning("Tick at least one material before fitting.")

                # A range for each material, and the equation that follows
                # from them. The bars show what the **board** holds, not what
                # was last fitted: a bar that snapped back to the fitted
                # placement while a boundary was being changed was the board
                # arguing with the person setting it. What is on the plot is
                # said under the bars, and by the ⚠️ line at the head of the
                # board when the two differ.
                _e1 = float(st.session_state["segment_break_1"])
                _e2 = float(st.session_state["segment_break_2"])
                _mem = MEMBRANE_CHOICES[st.session_state["membrane_after_break"]]
                _cyto = CYTO_CHOICES[st.session_state["cyto_starts_at"]]
                # One row per component: whether it is in the model, and the
                # stretch of the squash it acts on. The two belong together --
                # ticking a component and then finding its range three headings
                # away was two decisions about one thing.
                component_controls(
                    here, names, guided_lo, guided_hi, step, _e1, _e2, _mem, _cyto,
                )
                components_slot = st.container()
                components_area.__exit__(None, None, None)
                chosen = active_terms()

                st.markdown("**Step 3 · Fit**")
                go1, go2 = st.columns([1, 1])
                with go1:
                    fit_pressed = st.button(
                        "▶ Fit & plot", type="primary", key="guided_fit",
                        disabled=not chosen,
                        help="Fits the components ticked in step 1, at the "
                        "boundaries step 2 gives, and redraws the graph and "
                        "the results.",
                        **STRETCH,
                    )
                with go2:
                    refresh_pressed = st.button(
                        "🔄 Refresh graph", key="guided_refresh",
                        disabled=not chosen,
                        help="Redraws the graph and the results at exactly "
                        "what the board holds now. Moves no boundary and "
                        "chooses nothing.",
                        **STRETCH,
                    )


            # 🔄 Refresh graph: fit at what the board holds, moving nothing.
            if guided and refresh_pressed and chosen:
                rerun_keeping_settings({"_fit_from_curve": True})

            # ▶ Fit & plot, now that the range, the components and the
            # boundaries on the board are known.
            if guided and fit_pressed and chosen:
                if st.session_state.get("legacy_eps_mode") == "estimate":
                    estimate_boundaries_now(model, guided_lo, guided_hi, chosen,
                                            then_fit=True)
                else:
                    fit_at_the_current_settings(
                        model, guided_lo, guided_hi, chosen,
                    )

            # Staked out above, right under the component ranges, and filled
            # once the fit exists: Streamlit draws things where they were
            # created, not where the code that fills them runs.
            if not guided:
                curve_slot = st.container()
                video_slot = st.container()

            # Everything the fit decided for you, under the curve rather than
            # between the button and it. In guided mode all of it lives behind
            # this one collapsed line: three panels of settings in the middle of
            # a page whose whole promise is "press the button" was three panels
            # to scroll past. The controls still exist, and every one of them
            # still holds its value, because a setting that is not drawn is a
            # setting Streamlit forgets.
            settings_box = None
            if guided:
                # In the sidebar, where every other setting in this app already
                # lives. It cannot be dropped altogether: a control Streamlit
                # does not draw is a control whose value it forgets, and these
                # hold the weighting, the boundaries and the arrangement. Out of
                # the page's way, still one click from anywhere on it.
                # Both belong to the Fit step, so both sit on the page right
                # under it, collapsed: first what the fit assumed (every
                # option of the model, the windows, the weighting and the
                # plot preview of the arrangement), then how it came out.
                # Staked out now, in that order; the working is filled once
                # the fit exists.
                settings_box = st.expander(
                    "⚙️ Advanced: fit assumptions (composition at ε₁, windows, "
                    "weighting) and searches", expanded=False
                )
                settings_box.__enter__()
                st.caption(
                    "Every assumption of the fit θ̂ = argmin ‖F − X(ε₁, ε₂)θ‖². "
                    "Changes here wait, like the rest of the board, for "
                    "▶ Fit & plot."
                )
                # The unconstrained placement search lives here rather than on
                # the page: it rewrites every range at once and ignores what is
                # known about the cell type, which is the right tool rarely and
                # a surprise often.
                free_placement_control(model, guided_lo, guided_hi, active_terms())

            if not guided:
                st.divider()

            # Flat inside the box: Streamlit cannot nest an expander in an
            # expander, so in guided mode these are headings rather than panels
            # of their own.
            model_panel = open_panel(
                "⚙️ Change how the materials share the load", guided,
                flat=guided,
            )

            if guided:
                # The choice itself is the first control of the parameter
                # block, "How the components share the load"; here is the
                # algebra of each way.
                share_of_load_maths()
                st.caption(
                    "Chosen at the top of this block: **"
                    + current_sharing() + "**."
                )
            else:
                element_col, model_col = st.columns([1, 1.5])
                with model_col:
                    st.markdown("**2 · How they share the load**")
                    st.radio(
                        "How the cell is modelled",
                        list(MODELS.keys()),
                        key="model_kind",
                        label_visibility="collapsed",
                        help="Segmented treats the compression as stretches with "
                        "different materials bearing the load. The others assume "
                        "every material acts across the whole curve, which is "
                        "what makes them fail on a curve that changes character "
                        "partway along.",
                    )
                    st.caption(MODELS[st.session_state["model_kind"]])
                    if MODEL_KEYS[st.session_state["model_kind"]] == "segmented":
                        st.caption(
                            "For this cell type, in order: "
                            + " → ".join(
                                plain_name(t).lower()
                                for t in terms_for(st.session_state["cell_type"])
                                if t != "tension"
                            )
                            + "."
                        )
                with element_col:
                    st.markdown("**1 · Which materials**")
                    names = components_for(st.session_state["cell_type"])
                    fixed_cell_control()
                    for term in terms_for(st.session_state["cell_type"]):
                        st.checkbox(
                            f"{names[term][0]} · {TERM_SYMBOLS.get(term, term)}",
                            key=f"use_{term}",
                        )

            active = active_terms()
            kind = MODEL_KEYS[st.session_state["model_kind"]]
            segmented = kind == "segmented"
            coupling = kind
            cardiomyocyte_model = st.session_state["model_kind"].startswith(
                "Cardiomyocyte"
            )

            if cardiomyocyte_model:
                st.warning(
                    "**This model is provisional.** It is built from your "
                    "description of the Morales Maldonado picture, a strong "
                    "shell around fluid that does not compress, not from the "
                    "paper's equations, which I could not reach. Send me the "
                    "force-versus-deformation equation and I will replace it. "
                    "**Segmented** is the one to use meanwhile: it carries the "
                    "materials, the order and the boundaries.",
                    icon="⚠️",
                )

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
                if "cortex" in terms_for(st.session_state["cell_type"]):
                    st.caption(
                        "With a cortex in the model it is the cortex that "
                        "carries the load from first contact, so the general "
                        "scaffolding is set to start at ε₁. Starting both at "
                        "zero would make them one Hertzian term with two names, "
                        "and the fit would split them arbitrarily."
                    )
                st.caption(
                    f"→ {composition_label(membrane_mode, cyto_mode)}."
                    + (f" The {term_name('nucleus').lower()} always joins at ε₂."
                       if has_deep_term() else
                       " There is no deep element in this model, so ε₂ is not "
                       "used.")
                )
            else:
                membrane_mode, cyto_mode = "freeze", "break"

            # The picture, drawn from the choices immediately above it and
            # redrawn the moment one of them changes. It is the point of putting
            # the controls and the diagram together: a radio labelled "the
            # cytoskeleton starts at ε₁" means very little until you watch the
            # gap appear under its spring.
            if guided:
                live = current_style(force_N)
                # Drawn just past the first boundary, where every choice above is
                # visible at once: which springs are loading, which is locked and
                # which has not been reached. A slider for this was one more
                # widget on a page that has too many.
                preview_at = float(np.clip(
                    float(st.session_state["segment_break_1"]) * 1.35,
                    0.05, max(0.06, float(st.session_state["window_end"])),
                ))
                st.plotly_chart(
                    cell_schematic(
                        live,
                        **figure_kwargs(
                            cell_schematic,
                            coupling=("series" if coupling == "series"
                                      else "hybrid" if coupling.startswith("hybrid")
                                      else "parallel"),
                            epsilon=float(preview_at),
                            cell_height_um=st.session_state["cell_height_um"],
                            break_1=float(st.session_state["segment_break_1"]),
                            break_2=float(st.session_state["segment_break_2"]),
                            membrane_mode=membrane_mode,
                            cyto_start=cyto_mode,
                            labels=components_for(st.session_state["cell_type"]),
                            show_nucleus="nucleus" in active,
                            show_nucleus_shell="nucleus_shell" in active,
                            show_tension="tension" in active,
                            height=330,
                        ),
                    ),
                    key="sharing_preview",
                    **STRETCH,
                )
                st.caption(
                    "Drawn from the choices above, and redrawn as you change "
                    "them. A gap over a spring means that element is not "
                    "carrying load yet; a locked block means it has stopped "
                    "taking more. Nothing here is fitted — it is what you are "
                    "about to ask the fit to assume."
                )

            close_panel(model_panel)

            # Kept so the plot code below has something to write the diagram
            # into when it is not drawn here.
            sharing_slot = st.container()

            # No second opinion on which materials to use down here. The mixture
            # is decided in one place now, by the search button in step 1, which
            # applies what it finds and shows the combinations it compared. Two
            # places offering different answers to the same question is how the
            # ticks and the fit came apart in the first place.

            # ---------------------------------------------------- exploration ---
            # Only in full control. In guided mode the boundaries are found and
            # applied automatically, so a second, manual way of finding them was
            # a section to scroll past rather than a step to take.
            if segmented and not guided:
                explore_panel = open_panel("🔬 Explore the curve", guided)
                if not guided:
                    section("4 · Explore the curve")
                e1_col, e2_col = st.columns([1, 2])
                with e1_col:
                    if st.button("🔬 Detect segments (changepoints)", type="secondary", **STRETCH):
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
                        if st.button("✓ Apply these ε₁, ε₂", type="primary", **STRETCH):
                            st.session_state["_pending_settings"] = {
                                "segment_break_1": round(float(exploration["break_1"]), 3),
                                "segment_break_2": round(float(exploration["break_2"]), 3),
                            }
                            st.rerun()
                    elif exploration:
                        st.error(exploration.get("error", "Exploration failed."))

                if exploration and exploration.get("success"):
                    flat_table(
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
                        align_right=[
                            "ε from", "ε to", "points", "exponent measured",
                            "expected", "power-law R²",
                        ],
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

            if segmented and not guided:
                close_panel(explore_panel)

            # --------------------------------------------------------- ranges ---
            range_panel = open_panel(
                "📏 Where each material takes over", guided, flat=guided,
            )
            if not guided:
                section("5 · Deformation ranges" if segmented else "4 · Deformation ranges")
            st.caption(
                f"Data spans ε = {eps_lo_data:.3f} to {eps_hi_data:.3f}. "
                f"Auto-detected usable region: {auto_window[0]:.3f} to "
                f"{auto_window[1]:.3f} ({auto_range['n_points']} points) · "
                f"rupture: {auto_range['rupture_method']}"
            )

            # One range, one pair of numbers, wherever it is set from. In guided
            # mode it was set at the top of the page and repeating the control
            # here would give the page two of them, so this only reports it.
            #
            # The near end is normally zero: the membrane term is ε³ measured
            # from first contact, so a range starting anywhere else is fitting a
            # curve the model does not describe. It moves when the approach
            # itself misbehaved, which is a probe catching and slipping rather
            # than a soft cell, and those points drive the membrane modulus to
            # zero if they are kept.
            if guided:
                fit_lo, fit_hi = range_bounds(
                    "window_start", "window_end", 0.0, eps_hi_data, step,
                )
                st.caption(
                    f"Fitting ε = {fit_lo:.3f} to {fit_hi:.3f}, set at the top of "
                    "the page under **What to fit**."
                )
            else:
                fit_lo, fit_hi = epsilon_range_control(
                    "window_start", "window_end",
                    0.0 if segmented else eps_lo_data, eps_hi_data, step,
                    label="Fitted range, ε",
                    help_text="The stretch of the curve the fit is measured on. "
                              "Drag either end, or type it in the boxes.",
                )
            st.session_state["window_combined"] = (fit_lo, fit_hi)
            st.caption(f"{int(((epsilon >= fit_lo) & (epsilon <= fit_hi)).sum())} points")

            # The sidebar button only asks for the scan; it runs here, where the
            # model, the chosen range and the composition all exist. A sidebar
            # widget is built before any of them.
            if st.session_state.pop("_want_confinement_scan", False):
                if hasattr(model, "scan_confinement"):
                    with st.spinner("Refitting across a range of q…"):
                        st.session_state["confinement_scan"] = model.scan_confinement(
                            fit_lo, fit_hi,
                            e1=float(st.session_state["segment_break_1"]),
                            e2=float(st.session_state["segment_break_2"]),
                            membrane=MEMBRANE_CHOICES.get(
                                st.session_state["membrane_after_break"], "freeze"
                            ),
                            cyto_start=CYTO_CHOICES.get(
                                st.session_state["cyto_starts_at"], "break"
                            ),
                            use_nucleus="nucleus" in active,
                            use_tension="tension" in active,
                            weighting=st.session_state["weighting"],
                        )
                else:
                    st.session_state["confinement_scan"] = {
                        "success": False,
                        "error": "This needs an up to date lulevich_model.py.",
                    }

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
                # No table of segments. It said what each component does either
                # side of each boundary, which is what the component ranges up
                # in "3 · Nonlinear fitting" now show directly, and it was a
                # second, editable copy of two numbers that already have their
                # own sliders below.

                # Each boundary on its own, because most of the time only one
                # of them is wrong. The table above moves them together and is
                # the wrong tool for nudging ε₂ while ε₁ stays put.
                bc1, bc2 = st.columns(2)
                with bc1:
                    boundary_control(
                        "segment_break_1", "ε₁ · where the first hand-over is",
                        lower=float(fit_lo), upper=float(break_2) - 0.005,
                        step=float(step),
                        help_text="Where the first material hands over. Nothing "
                                  "else moves with it.",
                    )
                with bc2:
                    boundary_control(
                        "segment_break_2", "ε₂ · where the deeper one is met",
                        lower=float(break_1) + 0.005, upper=float(fit_hi),
                        step=float(step),
                        help_text="Where the deeper material is met. Nothing "
                                  "else moves with it.",
                        disabled=not has_deep_term(),
                    )

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
                    if st.button("🔎 Estimate ε₁, ε₂: grid argmin S(ε₁, ε₂)", **STRETCH):
                        with st.spinner("Scanning boundaries…"):
                            scan_breaks = model.scan_segment_breaks(
                                fit_lo, fit_hi, terms=active or ("membrane", "interior"),
                                weighting=st.session_state["weighting"],
                            )
                            # And where the curve's own power law changes, which
                            # is a second opinion arrived at a different way:
                            # the scan above asks which boundaries fit best, this
                            # asks where the log-log slope stops being one thing
                            # and starts being another. When they agree the
                            # boundary is real; when they do not, that is worth
                            # knowing before quoting a modulus either side of it.
                            try:
                                grid_e, grid_slope = model.local_exponent(
                                    window_frac=0.18
                                )
                                st.session_state["_power_law_check"] = {
                                    "epsilon": list(map(float, grid_e)),
                                    "exponent": list(map(float, grid_slope)),
                                }
                            except Exception:
                                st.session_state["_power_law_check"] = None
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
                        "picked above, and checks them against the curve's own "
                        "power law."
                    )
                    profile = st.session_state.get("_power_law_check")
                    if profile and len(profile.get("epsilon", [])) > 4:
                        grid = np.asarray(profile["epsilon"], dtype=float)
                        slope = np.asarray(profile["exponent"], dtype=float)
                        good = np.isfinite(slope)
                        if good.sum() > 4:
                            def slope_at(where):
                                index = int(np.argmin(np.abs(grid[good] - where)))
                                return float(slope[good][index])
                            e1_now = float(st.session_state["segment_break_1"])
                            e2_now = float(st.session_state["segment_break_2"])
                            st.caption(
                                f"Measured log-log slope: "
                                f"{slope_at(grid[good][0]):.2f} near contact, "
                                f"{slope_at(e1_now):.2f} at ε₁ = {e1_now:.3f}, "
                                f"{slope_at(e2_now):.2f} at ε₂ = {e2_now:.3f}, "
                                f"{slope_at(grid[good][-1]):.2f} at the far end. "
                                "3 is a membrane on its own, 3/2 a Hertzian "
                                "network on its own, and anything above 3 is the "
                                "cell running out of room. A boundary should sit "
                                "where that number is changing, not where it is "
                                "flat."
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
                        "use_nucleus": bool(row["use_nucleus"]) and has_deep_term(),
                    }

                with b2:
                    can_search = hasattr(model, "search_compositions")
                    if st.button(
                        "🧩 Select composition by cross-validation, then fit", type="primary",
                        disabled=not can_search, **STRETCH,
                    ) and can_search:
                        with st.spinner(
                            "Fitting all four combinations at their own best "
                            "boundaries and cross-validating each…"
                        ):
                            found = model.search_compositions(
                                fit_lo, fit_hi,
                                weighting=st.session_state["weighting"],
                                # A cell type with no deep element must not be
                                # offered one here either. Two searches on one
                                # page that disagree about what the cell is made
                                # of is worse than having only one of them.
                                nucleus_mode=(
                                    "search" if has_deep_term() else "off"
                                ),
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
                    st.info(retell(search["verdict"]))
                    shows_tension = any(
                        row.get("use_tension") for row in search["candidates"]
                    )
                    flat_table(
                        pd.DataFrame(
                            [
                                {
                                    "combination": retell(row["label"]),
                                    "ε₁": f"{row['break_1']:.3f}",
                                    "ε₂": f"{row['break_2']:.3f}",
                                    **(
                                        {"T₀ (mN/m)": f"{row.get('T0_mN_m', 0.0):.4g}"}
                                        if shows_tension else {}
                                    ),
                                    "Eₘ (MPa)": f"{row['Em_MPa']:.4g}",
                                    "E_c (kPa)": f"{row['Ec_kPa']:.4g}",
                                    "Eₙ (kPa)": f"{row['En_kPa']:.4g}",
                                    "R²": f"{row['r_squared']:.5f}",
                                    "CV RMSE": f"{row['cv_rmse']:.4g}",
                                    "ΔAICc": f"{row['delta_aicc']:.1f}",
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
                        align_right=[
                            "ε₁", "ε₂", "T₀ (mN/m)", "Eₘ (MPa)", "E_c (kPa)",
                            "Eₙ (kPa)", "R²", "CV RMSE", "ΔAICc",
                        ],
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
                    # Shown with this cell type's names; matched back on the
                    # model's own label, which is what the candidates carry.
                    shown_to_row = {
                        retell(row["label"]): row for row in search["candidates"]
                    }
                    choice_names = list(shown_to_row)
                    best_shown = retell(best["label"])
                    a1, a2 = st.columns([2, 1])
                    with a1:
                        picked = st.selectbox(
                            "Override the pick", choice_names,
                            index=choice_names.index(best_shown)
                            if best_shown in choice_names else 0,
                            key="composition_pick",
                            help="The winner is already applied. Use this only to try "
                            "one of the others.",
                        )
                    with a2:
                        st.markdown("<div style='height:1.7rem'></div>",
                                    unsafe_allow_html=True)
                        if st.button("✓ Apply this (ε₁, ε₂)", **STRETCH):
                            row = shown_to_row[picked]
                            st.session_state["_pending_settings"] = stage_combination(row)
                            st.rerun()
                    st.caption(
                        f"Applied: ε₁ = {best['break_1']:.3f}, ε₂ = "
                        f"{best['break_2']:.3f}, "
                        f"{'with' if best['use_nucleus'] else 'without'} the "
                        f"{term_name('nucleus').lower()}."
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
                                term_label(term), min_value=eps_lo_data,
                                max_value=eps_hi_data, step=step, key=key,
                            )
                            term_windows[term] = (lo, hi)
                            st.selectbox(f"Stage for {term_label(term)}", [1, 2, 3],
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
                if st.button("↺ Reset element windows", **STRETCH):
                    st.session_state["_pending_clear_windows"] = True
                    st.rerun()
            with r2:
                range_label = "Fitted range"
                targets = ["(off)", range_label] + [
                    term_label(t) for t in (active if staged else [])
                ]
                st.session_state["_drag_keys"] = {
                    range_label: "window_range"
                }
                st.session_state["_drag_keys"].update(
                    {term_label(t): f"window_term_{t}" for t in (active if staged else [])}
                )
                st.selectbox("Drag on the plot to set", targets, key="drag_target")

            # Flat when guided, because this already sits inside the settings
            # box and Streamlit refuses an expander inside an expander.
            with sub_panel("⚙️ Advanced fitting options", flat=guided):
                a1, a2 = st.columns(2)
                with a1:
                    st.selectbox(
                        "Weighting", ["uniform", "relative", "noise"], key="weighting",
                        help="How much each point counts. A whole-cell curve "
                        "spans four decades of force, so **uniform** is decided "
                        "almost entirely by its last tenth and can miss the "
                        "first half by tens of per cent without the residual sum "
                        "noticing. **relative** weights by 1/|F|, so every decade "
                        "counts the same and the fit holds everywhere; it is the "
                        "default for a cardiomyocyte. **noise** weights by 1/σ "
                        "measured from the curve, the maximum-likelihood choice "
                        "and the one χ²/dof assumes.",
                    )
                    st.checkbox("Fit a constant force offset", key="fit_offset")
                with a2:
                    st.slider("Refinement passes (staged fits)", 1, 8,
                              key="refine_iterations")
                    st.checkbox("Seed staged fits from all-at-once", key="seed_parallel")
                if not guided:
                    st.checkbox("Refit live as settings change", key="live_fit")

            # ------------------------------------------------- saved presets ---
            with sub_panel("💾 Saved windows", flat=guided):
                p1, p2 = st.columns([2, 1])
                with p1:
                    preset_name = st.text_input(
                        "Preset name",
                        placeholder="e.g. C2C12 standard",
                        key="preset_name",
                        label_visibility="collapsed",
                    )
                with p2:
                    if st.button("💾 Save windows as preset", **STRETCH):
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
                        if st.button("Apply preset", key="apply_preset",
                                     **STRETCH):
                            apply_preset(presets[chosen], eps_lo_data, eps_hi_data)
                            st.rerun()
                    with a3:
                        if st.button("Delete preset", **STRETCH):
                            presets.pop(chosen, None)
                            st.rerun()

                    flat_table(
                        pd.DataFrame(
                            [
                                {
                                    "preset": name,
                                    "model": pre.get("coupling", "?"),
                                    "ε₁": pre.get("segment_break_1"),
                                    "ε₂": pre.get("segment_break_2"),
                                    "cell type": pre.get("cell_type", "?"),
                                    "windows": " | ".join(
                                        " + ".join(term_label(t) for t in st_["terms"])
                                        + f" {st_['range'][0]:.3f} to {st_['range'][1]:.3f}"
                                        for st_ in pre.get("stages", [])
                                    ),
                                    "saved": pre.get("saved_at", ""),
                                }
                                for name, pre in presets.items()
                            ]
                        ),
                        align_right=["ε₁", "ε₂"],
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
            close_panel(range_panel)

            fit_panel = open_panel("🔧 Solver options", guided, flat=guided)
            if not guided:
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

            # Two ways to press it, one button. A flag set beside the curve is
            # read here, where the fit actually happens: a second st.button with
            # the same job would be a duplicate widget key, and Streamlit
            # refuses those outright.
            # One button: ▶ Fit & plot on the control board, which leaves
            # this flag for the next pass. A new curve leaves it too, so it
            # is fitted once on arrival. Nothing else refits.
            # (live_fit is off and has no switch on the page; it is kept for
            # scripts and tests that want every change refitted.)
            run = (
                st.session_state.pop("_fit_from_curve", False)
                or st.session_state["live_fit"]
                or (not guided
                    and st.button("🚀 Fit θ̂ (current settings)", type="primary"))
            )

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
                            use_tension="tension" in active,
                            use_nucleus_shell="nucleus_shell" in active,
                            use_cortex="cortex" in active,
                            weighting=st.session_state["weighting"],
                            fit_offset=st.session_state["fit_offset"],
                            **figure_kwargs(
                                model.fit_composition,
                                term_windows=element_windows(
                                    active, fit_lo, fit_hi
                                ),
                            ),
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

            close_panel(fit_panel)
            if settings_box is not None:
                settings_box.__exit__(None, None, None)
            if fit_block is not None:
                # The parameters end here; everything from now on is what
                # the fit came to, in the right-hand column.
                fit_block.__exit__(None, None, None)
                results_area.__enter__()

            # The video and database section below runs whether or not there is a
            # fit, so the names it reads have to exist either way.
            fitted = membrane = interior = nucleus = None

            stale_fit = False
            if fit is not None and fit.get("success"):
                # How much each modulus depends on where the boundaries were put.
                # Done once, here, rather than in the panel that displays it: this
                # runs when a fit happens, that runs on every rerun.
                if fit.get("coupling") == "segmented" and hasattr(
                    model, "breakpoint_spread"
                ):
                    try:
                        fit["breakpoint_spread"] = model.breakpoint_spread(
                            fit["epsilon_range"][0], fit["epsilon_range"][1],
                            fit["break_1"], fit["break_2"],
                            fit.get("membrane", "freeze"),
                            fit.get("cyto_start", "break"),
                            use_nucleus="nucleus" in (fit.get("terms") or ()),
                            weighting=fit.get("weighting", "uniform"),
                            use_tension="tension" in (fit.get("terms") or ()),
                        )
                    except Exception:
                        fit["breakpoint_spread"] = None
                fit["fit_id"] = fit_id(fit)
                st.session_state["_last_fit"] = fit
                st.session_state["_last_fit_signature"] = fit_signature
                # No redraw of the bars here. They show what the control
                # board holds, which is what the next fit will use; what
                # this fit used is written under them by components_note.
            elif fit is None and st.session_state.get("_last_fit") is not None:
                # Nothing asked for a fit on this run, so show the last good one
                # rather than an empty page.
                fit = st.session_state["_last_fit"]
                stale_fit = (
                    st.session_state.get("_last_fit_signature") != fit_signature
                )
                if stale_fit and not guided:
                    st.info(
                        "Showing the previous fit. Something has changed since it "
                        "was made, so press **Fit θ̂ (current settings)** to bring "
                        "it up to date."
                    )

            if verdict_slot is not None:
                # Written now, from the fit the rest of this column shows.
                if fit is not None and fit.get("success") and not fit.get("fit_id"):
                    fit["fit_id"] = fit_id(fit)
                with strip_slot:
                    component_results_strip(fit)
                with verdict_slot:
                    fit_verdict(fit)
                if boundaries_slot is not None:
                    with boundaries_slot:
                        boundaries_note(fit)
                with components_slot:
                    components_note(fit)
                with explainer_box:
                    fit_explainer(fit)
                if board_status is not None:
                    waiting = (fit is None or stale_fit)
                    board_status.markdown(
                        "⚠️ **The board has changes the plot does not show "
                        "yet.** Press ▶ Fit & plot to apply them." if waiting
                        else "✓ The graph and the results show the settings "
                        "on this board.")

            if comparison and comparison.get("success"):
                st.info(retell(comparison["verdict"]))
                flat_table(
                    pd.DataFrame(
                        [
                            {
                                "model": row["label"],
                                "R²": f"{row['r_squared']:.5f}",
                                "ΔAICc": f"{row['delta_aicc']:.1f}",
                                "weight": f"{row['weight']:.3f}",
                                "CV RMSE": f"{row['cv_rmse']:.3g}",
                                "params": row["n_params"],
                            }
                            for row in comparison["candidates"]
                        ]
                    ),
                    align_right=["R²", "ΔAICc", "weight", "CV RMSE", "params"],
                )
                st.caption(
                    "ΔAICc under 2 means the curve cannot tell those models apart. A "
                    "wrong model can still reach R² > 0.99 with badly wrong moduli, "
                    "which is why this table exists."
                )

            if fit is None:
                st.info("Press **Fit θ̂ (current settings)**, or turn on live refitting.")
            elif not fit.get("success"):
                st.error(fit.get("error", "Fit failed."))
            else:
                En_value = fit.get("En", 0.0)
                fitted_coupling = fit.get("coupling", "parallel")
                params = (fit.get("Em", 0.0), fit.get("Ei", 0.0), En_value)
                params = tuple(0.0 if not np.isfinite(v) else v for v in params)
                T0_value = float(fit.get("T0", 0.0) or 0.0)
                if not np.isfinite(T0_value):
                    T0_value = 0.0
                Ene_value = float(fit.get("Ene", 0.0) or 0.0)
                if not np.isfinite(Ene_value):
                    Ene_value = 0.0
                Ecx_value = float(fit.get("Ecx", 0.0) or 0.0)
                if not np.isfinite(Ecx_value):
                    Ecx_value = 0.0
                envelope = cortex = None

                if fitted_coupling == "segmented":
                    # Draw the components with the same basis the fit used, or the
                    # curves would not add up to the line through the data.
                    tension_basis = None
                    if hasattr(model, "composition_basis"):
                        basis = figure_kwargs(
                            model.composition_basis,
                            term_windows=fit.get("term_windows"),
                        )
                        basis = model.composition_basis(
                            epsilon, fit["break_1"], fit["break_2"],
                            fit.get("membrane", "freeze"),
                            fit.get("cyto_start", "break"),
                            **basis,
                        )
                        membrane_basis = basis["membrane"]
                        cyto_basis = basis["interior"]
                        nucleus_basis = basis["nucleus"]
                        tension_basis = basis["tension"]
                        envelope_basis = basis["nucleus_shell"]
                        cortex_basis = basis["cortex"]
                    elif hasattr(model, "composition_terms"):
                        membrane_basis, cyto_basis, nucleus_basis = model.composition_terms(
                            epsilon, fit["break_1"], fit["break_2"],
                            fit.get("membrane", "freeze"), fit.get("cyto_start", "break"),
                        )
                    else:
                        membrane_basis, cyto_basis, nucleus_basis = model.segment_terms(
                            epsilon, fit["break_1"], fit["break_2"]
                        )
                    tension = (
                        tension_basis * T0_value
                        if (tension_basis is not None and T0_value) else None
                    )
                    envelope = (
                        envelope_basis * Ene_value
                        if (envelope_basis is not None and Ene_value) else None
                    )
                    cortex = (
                        cortex_basis * Ecx_value
                        if (cortex_basis is not None and Ecx_value) else None
                    )
                    fitted = (
                        membrane_basis * params[0]
                        + cyto_basis * params[1]
                        + nucleus_basis * params[2]
                        + (tension if tension is not None else 0.0)
                        + (envelope if envelope is not None else 0.0)
                        + (cortex if cortex is not None else 0.0)
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
                    tension = None
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
                    membrane = interior = nucleus = tension = None
                    envelope = cortex = None

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

                def clip_to_support(values, term):
                    """An element's own curve starts where the element does."""
                    out = clip_to_window(values)
                    if out is None:
                        return None
                    window = (fit.get("term_windows") or {}).get(term)
                    if window:
                        out[epsilon < float(window[0])] = np.nan
                    return out

                fitted = clip_to_window(fitted)
                tension = clip_to_support(tension, "tension")
                membrane = clip_to_support(membrane, "membrane")
                interior = clip_to_support(interior, "interior")
                nucleus = clip_to_support(nucleus, "nucleus")
                envelope = clip_to_support(envelope, "nucleus_shell")
                cortex = clip_to_support(cortex, "cortex")

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
                    # Straight from the element ranges, so a band behind the
                    # curve says the same thing as the algebra under the log
                    # plot. Hard-coded as membrane / cytoskeleton / cytoskeleton
                    # plus nucleus, it stopped being true the moment an element
                    # was given a range of its own -- and on a C2C12, whose
                    # sarcolemma carries throughout, it was never true at all.
                    windows_for_plot = stage_bands(fit)
                else:
                    windows_for_plot = [
                            {
                                "range": tuple(s["range"]),
                                "label": " + ".join(term_label(t) for t in s["terms"]),
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

                if guided:
                    # No retelling of the fit in words. It restated the
                    # boundaries already drawn on the curve, printed the moduli
                    # a second time under the ones below, and ended with a
                    # sentence about chi-squared that a person could read
                    # straight off the two numbers beside R². Three of those
                    # were duplicates and the fourth was noise.
                    section("Fitting results")
                    fitting_results_rows(fit, style, send=False, compact=True)

                rmse_disp, rmse_unit = from_newtons(fit["rmse"], style.force_unit)
                sigma_disp, sigma_unit = from_newtons(
                    fit.get("noise_sigma", float("nan")), style.force_unit
                )
                st.caption(
                    f"RMSE {float(rmse_disp):.3g} {rmse_unit} · measured noise "
                    f"±{float(sigma_disp):.3g} {sigma_unit} per point. "
                    "χ²/dof compares the residuals against that noise: about 1 "
                    "means the model is as close to the points as the scatter "
                    "allows, and much above 1 means it is missing something real. "
                    "R² can still look excellent when χ²/dof is in the hundreds, "
                    "which is exactly when the model is wrong. The noise is "
                    "estimated from the curve itself, so read χ²/dof as an "
                    "order of magnitude, not to two decimal places."
                )

                # No second table of the same moduli. Each is quoted with its
                # own uncertainty in the tile above, which is where a modulus
                # belongs; everything else about it is in the equation and the
                # copy block below.

                # The model must never soften. Every basis function has a
                # non-decreasing slope and every modulus is bounded at zero, so
                # this holds by construction; it is checked rather than asserted
                # because a future term could break it silently, and a cell that
                # appeared to soften under load would be a physical claim nobody
                # meant to make.
                if fitted is not None:
                    drawn = np.isfinite(fitted)
                    if drawn.sum() > 5:
                        e_drawn, f_drawn = epsilon[drawn], fitted[drawn]
                        slope = np.diff(f_drawn) / np.maximum(np.diff(e_drawn), 1e-12)
                        softening = float(np.min(np.diff(slope))) if slope.size > 2 else 0.0
                        falls = float(np.min(np.diff(f_drawn))) if f_drawn.size > 1 else 0.0
                        if falls < -1e-15 or softening < -abs(np.max(slope)) * 1e-6:
                            st.warning(
                                "The fitted curve is not stiffening monotonically, "
                                "which no combination of these elements should be "
                                "able to do. Treat this fit as suspect and tell "
                                "whoever maintains the app.",
                                icon="⚠️",
                            )

                # How well it fits, band by band, as a percentage. R² and the
                # residual sum are both dominated by the last tenth of a curve
                # that spans four decades, so both can look perfect while the
                # first half is out by a third. This is the number that says
                # whether the model follows the curve where you are looking.
                if fitted is not None:
                    inside = (epsilon >= fit["epsilon_range"][0]) & (
                        epsilon <= fit["epsilon_range"][1]
                    )
                    bands, worst_band = [], 0.0
                    edges = np.array([0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.01])
                    for lo_b, hi_b in zip(edges[:-1], edges[1:]):
                        here = inside & (epsilon >= lo_b) & (epsilon < hi_b)
                        here &= np.isfinite(fitted)
                        if here.sum() < 5:
                            continue
                        measured = force_N[here]
                        scale = np.mean(np.abs(measured))
                        if scale <= 0:
                            continue
                        miss = float(np.mean(fitted[here] - measured) / scale)
                        worst_band = max(worst_band, abs(miss))
                        bands.append({
                            "ε from": f"{lo_b:.2f}",
                            "to": f"{min(hi_b, 1.0):.2f}",
                            "points": int(here.sum()),
                            "typical force there": (
                                f"{float(from_newtons(scale, style.force_unit)[0]):.3g} "
                                f"{style.force_unit}"
                            ),
                            "model is out by": f"{100 * miss:+.1f} %",
                        })
                    # Working, not answer. In guided mode the page is meant to
                    # read curve, numbers, maths, database, and every extra
                    # panel between them is one more thing to scroll past.
                    if bands:
                        with diagnostics_box:
                            st.markdown(
                                f"**📊 How well it fits, stretch by stretch** "
                                f"(worst {100 * worst_band:.1f} %)"
                            )
                            flat_table(
                                pd.DataFrame(bands),
                                align_right=["points", "typical force there",
                                             "model is out by"],
                            )
                            st.caption(
                                "The percentage is the average signed miss in "
                                "that stretch, as a fraction of the force there. "
                                "R² and χ² are both dominated by the largest "
                                "forces, so a curve spanning four decades can "
                                "read R² = 0.99999 while the first half is out "
                                "by a third; this table is where that shows. "
                                "Weighting, in Advanced fitting options, is what "
                                "trades one end against the other: **uniform** "
                                "buys the top of the curve, **relative** spreads "
                                "the error evenly."
                            )
                            if worst_band > 0.15 and st.session_state["weighting"] == "uniform":
                                st.warning(
                                    f"The model is out by {100 * worst_band:.0f} % "
                                    f"somewhere, and the fit is weighted "
                                    f"uniformly, which means the low-force end "
                                    f"was barely counted. Switching the weighting "
                                    f"to **relative** usually brings this under "
                                    f"5 % everywhere without hurting the top of "
                                    f"the curve."
                                )

                # What the squash did to the sarcomeres. Geometry rather than a
                # fitted result: the sarcomeres are not a spring in this model,
                # they are part of the one incompressible interior. This is the
                # check that says whether the cell was squashed through a range
                # where muscle still behaves like muscle.
                if (
                    st.session_state["cell_type"] in HAS_SARCOMERES
                    and hasattr(model, "sarcomere_report")
                ):
                    # An onset belongs to a deep spring. Without one there is
                    # nothing switching on, so the read-out covers the whole
                    # fitted range rather than quoting a boundary that is not in
                    # the model.
                    deep_onset = (
                        fit.get("break_2") if "nucleus" in (fit.get("terms") or ())
                        else None
                    )
                    free = model.sarcomere_report(
                        fit["epsilon_range"][1], onset=deep_onset,
                        spread=float(st.session_state["sarcomere_spread"]),
                    )
                    held = model.sarcomere_report(
                        fit["epsilon_range"][1], onset=deep_onset, spread=0.0,
                    )
                    with diagnostics_box:
                        st.markdown("**🧬 What the squash did to the sarcomeres**")
                        s1, s2, s3 = st.columns(3)
                        s1.metric(
                            "Relaxed", f"{free['relaxed_nm']:.0f} nm",
                            delta=f"about {free['n_along_cell']:.0f} along the cell",
                            delta_color="off",
                        )
                        s2.metric(
                            f"At ε = {free['epsilon_max']:.2f}",
                            f"{free['at_epsilon_max_nm']:.0f} nm",
                            delta=f"{100 * (free['stretch'] - 1):+.0f} %",
                            delta_color="off",
                        )
                        if "at_onset_nm" in free:
                            s3.metric(
                                f"Where they engage, ε₂ = {free['onset']:.2f}",
                                f"{free['at_onset_nm']:.0f} nm",
                            )
                        st.caption(
                            "Squashing a cardiomyocyte does not shorten its "
                            "sarcomeres, it lengthens them. The myofibrils run "
                            "along the cell, across the direction of the squash, "
                            "and a cell that keeps its volume has to spread "
                            "sideways by as much as it loses in height, which "
                            "pulls them out: L = L₀ (1 − ε)^(−s∕2), where s is "
                            "how much of that spreading runs along the "
                            "myofibrils, set in the sidebar. This is geometry, "
                            "not a fitted number, and nothing above depends on it."
                        )
                        if float(st.session_state["sarcomere_spread"]) > 0:
                            st.caption(
                                f"Held at its ends instead, the same cell would "
                                f"keep its sarcomeres at "
                                f"{held['at_epsilon_max_nm']:.0f} nm throughout. "
                                f"A real attached cell lies between, so read "
                                f"{held['at_epsilon_max_nm']:.0f} to "
                                f"{free['at_epsilon_max_nm']:.0f} nm as the "
                                f"bounds at ε = {free['epsilon_max']:.2f}."
                            )
                        if free["beyond_working_range"]:
                            st.warning(
                                f"Past ε = {free['epsilon_at_limit']:.2f} the "
                                f"sarcomeres are longer than "
                                f"{free['working_limit_nm']:.0f} nm, where actin "
                                f"and myosin overlap stops improving and force "
                                f"falls away with further stretch. Your fitted "
                                f"range reaches ε = {free['epsilon_max']:.2f}, "
                                f"which is {free['at_epsilon_max_nm']:.0f} nm. "
                                f"Over that stretch the interior modulus "
                                f"measures passive structure pulled beyond its "
                                f"working length, not contractile machinery at a "
                                f"length it ever works at. That may be exactly "
                                f"what you mean to measure; it is worth saying "
                                f"which."
                            )

                # How much each number depends on where the boundaries landed.
                # A standard error is computed with the boundaries held fixed, so
                # on a curve where the boundaries are not well determined it can
                # be small next to a modulus that is not determined at all. This
                # is the part that catches that.
                spread = fit.get("breakpoint_spread")
                fitted_terms = set(fit.get("terms") or active)
                if spread and spread.get("success") and spread["n_accepted"] > 1:
                    loose_rows, spread_rows = [], []
                    for key, unit, term in (
                        ("T0_mN_m", "mN/m", "tension"),
                        ("Em_MPa", "MPa", "membrane"),
                        ("Ei_kPa", "kPa", "interior"),
                        ("En_kPa", "kPa", "nucleus"),
                    ):
                        if term not in fitted_terms:
                            continue
                        band = spread["ranges"].get(key)
                        if not band or not np.isfinite(band["relative"]):
                            continue
                        spread_rows.append(
                            {
                                "modulus": components_for(
                                    st.session_state["cell_type"]
                                )[term][0],
                                "best fit": f"{band['value']:.4g} {unit}",
                                "but anywhere in": (
                                    f"{band['low']:.4g} to {band['high']:.4g} {unit}"
                                ),
                                "how loose": f"{100 * band['relative']:.0f} % of itself",
                            }
                        )
                        if band["relative"] > 0.5:
                            loose_rows.append(term)
                    if spread_rows:
                        with diagnostics_box:
                            st.markdown(
                                "**📏 How much do these numbers depend on where "
                                "the boundaries were put?**"
                            )
                            st.caption(
                                f"ε₁ and ε₂ are fitted too, and "
                                f"{spread['n_accepted']} placements of them fit "
                                f"this curve within its own noise "
                                f"(ε₁ from {spread['break_1_range'][0]:.3f} to "
                                f"{spread['break_1_range'][1]:.3f}, ε₂ from "
                                f"{spread['break_2_range'][0]:.3f} to "
                                f"{spread['break_2_range'][1]:.3f}). This is the "
                                f"range each modulus takes across all of them, "
                                f"which is a truer error bar than the ± beside "
                                f"each number: that one is worked out with the "
                                f"boundaries held fixed, as though they were known."
                            )
                            flat_table(
                                pd.DataFrame(spread_rows),
                                align_right=["best fit", "but anywhere in",
                                             "how loose"],
                            )
                            if loose_rows:
                                st.warning(
                                    "**"
                                    + " and ".join(
                                        components_for(
                                            st.session_state["cell_type"]
                                        )[t][0] for t in loose_rows
                                    )
                                    + "** moves by more than half its own value "
                                    "across boundaries this curve cannot tell "
                                    "apart. Quote it with that range, not with "
                                    "the ± above. A wider fitted range, or more "
                                    "points near ε = 0, is what narrows it."
                                )

                st.caption(
                    f"Membrane areal modulus Eₘ·h = "
                    f"{fit.get('membrane_areal_modulus', 0.0) * 1e3:.4g} mN/m, which is what "
                    f"the ε³ term actually determines. Eₘ itself is that divided by the "
                    f"assumed bilayer thickness of "
                    f"{st.session_state['membrane_thickness_nm']:.1f} nm, so halving the "
                    f"thickness doubles Eₘ while the measurement is unchanged."
                )

                # No table of which material carried what in each stretch. The
                # component ranges say where each one acts, the equation says
                # what each contributes, and the moduli are printed twice
                # already; a third table of percentages per stretch was reading
                # the same fit for a third time.

                if fitted_coupling != "parallel" and st.session_state["show_components"]:
                    st.caption(
                        "Element curves are not drawn for series or hybrid coupling: "
                        "every element carries the same force there, so they would be "
                        "three copies of the total. The deformation share each one "
                        "takes is in the diagram beside the plot."
                    )

                # A modulus of exactly zero is the solver saying it did not want
                # that term. Usually that is informative; in one case it is two
                # settings cancelling each other, and saying which is the whole
                # difference between a useful message and a confusing one.
                zeroed = [
                    label for label, key, term in (
                        ("Eₘ", "Em_MPa", "membrane"),
                        ("E_c", "Ei_kPa", "interior"),
                        ("Eₙ", "En_kPa", "nucleus"),
                    )
                    if term in fitted_terms and float(fit.get(key, 0.0)) <= 0
                ]
                if zeroed:
                    message = (
                        f"{' and '.join(zeroed)} came back at exactly zero, which "
                        f"means the fit found no work for that term to do."
                    )
                    if (
                        "Eₘ" in zeroed
                        and fit.get("membrane") == "freeze"
                        and fit.get("cyto_start") == "zero"
                    ):
                        message += (
                            "  **Here the two choices above are cancelling each "
                            "other.** If the cytoskeleton is loaded from the very "
                            "start, freezing the membrane at ε₁ leaves its term a "
                            "flat constant for the rest of the curve, and a "
                            "constant cannot describe anything the cytoskeleton "
                            "is not already describing, so the solver sets it to "
                            "zero. For a cell whose membrane and cytoskeleton are "
                            "coupled from first contact, set the membrane to "
                            "**keeps stiffening**: both then act everywhere, which "
                            "is what coupled means."
                        )
                    st.warning(message)

                # The dropped-spring sentence is already on the page, next to
                # the model selector that caused it, with a button that fixes it.
                # Repeating it here reads as a second, different problem.
                said_already = set()
                if fit.get("dropped_terms"):
                    said_already = set(dropped_term_warning(fit["dropped_terms"]))
                for message in fit["warnings"]:
                    if message in said_already:
                        continue
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

                # The curve goes into the slot staked out at the top of the
                # page. The cell diagram no longer sits beside it: it belongs
                # with the question it answers, which is how the elements share
                # the load, and squeezing both into a side column is what made
                # them illegible.
                plot_weight = float(st.session_state["plot_width"])
                if guided:
                    plot_col = curve_slot
                    # The equation and the working go at the foot of the page
                    # now, after the numbers, rather than in a panel above the
                    # curve: the page reads choose, fit, look, then what the
                    # cell did, then the maths.
                    # The diagram is already drawn live in Step 2, from the same
                    # settings. Drawing it again here would be the same picture
                    # twice on one page.
                    panel_cols = []
                else:
                    side_panels = int(bool(video_ready)) + int(bool(show_schematic))
                    if side_panels:
                        widths = [plot_weight] + [1.0] * side_panels
                        columns = st.columns(widths)
                        plot_col = columns[0]
                        panel_cols = columns[1:]
                    else:
                        # Even with no side panel, keep the chart from spanning
                        # the whole page; the spare column is left empty.
                        plot_col, spare = st.columns(
                            [plot_weight, max(0.01, 4.0 - plot_weight)]
                        )
                        panel_cols = []

                with plot_col:
                    # Built once and reused for the save button. Building it twice
                    # doubled the work on every rerun for two identical figures.
                    # The corner note is written from the fit being drawn, not
                    # from the page's memory of the last one.
                    shown_style = style
                    if getattr(style, "range_note", None):
                        shown_style = replace_style(
                            style, range_note=fit_range_note(fit))
                    figure = force_curve_figure(
                        epsilon,
                        force_N,
                        shown_style,
                        **figure_kwargs(
                            force_curve_figure,
                            title=st.session_state["cell_name"]
                            or "Force vs relative deformation",
                            fit_force_N=fitted,
                            membrane_N=membrane,
                            interior_N=interior,
                            nucleus_N=nucleus,
                            nucleus_shell_N=envelope,
                            cortex_N=cortex,
                            deep_label=plain_name("nucleus").lower(),
                            cortex_label=plain_name("cortex").lower(),
                            interior_label=plain_name("interior").lower(),
                            # Bands, boundaries and the rupture are drawn by
                            # decorate_curve_figure below, from this fit.
                            fit_window=None,
                            rupture_epsilon=None,
                            highlight=highlight
                            if (video_ready or show_schematic) else None,
                            highlight_window=highlight_window,
                        ),
                    )
                    decorate_curve_figure(
                        figure, shown_style, fit, epsilon,
                        bands=(windows_for_plot
                               if getattr(shown_style, "show_fit_window", True)
                               else ()),
                        rupture_eps=(rupture.get("epsilon")
                                     if rupture.get("method") == "force-drop"
                                     else None),
                        boundaries=layer_on("boundaries"),
                        show_range=layer_on("range"),
                    )
                    figure = apply_plot_layers(figure, shown_style, fit)
                    st.plotly_chart(
                        figure,
                        key="main_fit_plot",
                        **plot_selection_kwargs(),
                        **STRETCH,
                    )
                    apply_plot_drag("main_fit_plot", eps_lo_data, eps_hi_data)
                    # What is on the graph right now, in one line, under it.
                    legacy_graph_note(fit, epsilon)

                    # One row of nesting at most: this sits in a column
                    # already, and the layer list has columns of its own.
                    with st.expander("🎛️ On the plot · plot options · save",
                                     expanded=False):
                        plot_boxes_control(fit)
                        st.markdown("**Plot options**")
                        plot_option_controls()
                        save_plot_controls(figure, fit, date_acquired)


                panel_index = 0
                if show_schematic and panel_index < len(panel_cols):
                    with panel_cols[panel_index]:
                        if st.session_state["schematic_style"].startswith("Balloon"):
                            st.plotly_chart(
                                balloon_figure(
                                    style,
                                    **figure_kwargs(
                                        balloon_figure,
                                        epsilon=selected_eps,
                                        cell_height_um=st.session_state["cell_height_um"],
                                        labels=components_for(
                                            st.session_state["cell_type"]
                                        ),
                                        show_nucleus="nucleus" in active,
                                        show_nucleus_shell="nucleus_shell" in active,
                                        show_tension="tension" in active,
                                        deep_onset=(
                                            fit.get("break_2")
                                            if "nucleus" in (fit.get("terms") or ())
                                            else None
                                        ),
                                        # A cardiomyocyte is a shell holding
                                        # fluid, and the model does not
                                        # discriminate a nucleus from the rest
                                        # of the interior, so nothing in the
                                        # picture should look like one.
                                        interior=(
                                            "fluid"
                                            if st.session_state["cell_type"]
                                            in INCOMPRESSIBLE_INTERIOR else "spring"
                                        ),
                                    ),
                                ),
                                key="balloon_plot",
                                **STRETCH,
                            )
                        else:
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
                                        labels=components_for(st.session_state["cell_type"]),
                                        Em_MPa=fit["Em_MPa"],
                                        Ei_kPa=fit["Ei_kPa"],
                                        En_kPa=fit.get("En_kPa") if "nucleus" in active else None,
                                        Ene_MPa=(
                                            fit.get("Ene_MPa")
                                            if "nucleus_shell" in active else None
                                        ),
                                        T0_mN_m=fit.get("T0_mN_m") if "tension" in active else None,
                                        show_nucleus="nucleus" in active,
                                        show_nucleus_shell="nucleus_shell" in active,
                                        show_tension="tension" in active,
                                    ),
                                ),
                                key="schematic_plot",
                                **STRETCH,
                            )
                    panel_index += 1

                # In guided mode the video frame goes under the curve rather
                # than beside it, for the same reason as the diagram.
                video_target = None
                if video_ready:
                    if guided:
                        video_target = video_slot
                    elif panel_index < len(panel_cols):
                        video_target = panel_cols[panel_index]
                if video_target is not None:
                    with video_target:
                        vinfo = st.session_state["video_info"]
                        frame_index = va.frame_for_epsilon(
                            selected_eps,
                            st.session_state["video_contact_frame"],
                            st.session_state["video_end_frame"] or (vinfo["n_frames"] - 1),
                            float(epsilon.max()),
                        )
                        # Through the shared reader, so a box drawn by hand and a
                        # scale taken from the probe on the video tab apply here
                        # too. This panel used to call the detector directly and
                        # ignore both.
                        vframe, vdet, vnuc, vprobe, vscale = detection_at(frame_index)
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
                                height_note = f"Cell height {vdet['height_px']:.0f} px"
                                if vscale:
                                    height_note += (
                                        f" · {vdet['height_px'] * vscale:.2f} µm"
                                    )
                                if vdet.get("manual"):
                                    height_note += " · outlined by hand"
                                st.caption(height_note)
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
                with diagnostics_box:
                    st.markdown("**🔍 Fit diagnostics**")
                    # One column per material actually in the fit, and never a
                    # column for one that is not: a share of the load reported
                    # for a material the model does not have reads as a
                    # measurement of it.
                    shares = [
                        (f"{plain_name('membrane')} share at ε_max",
                         fit.get("membrane_fraction_at_max", np.nan)),
                        (f"{plain_name('interior')} share",
                         fit.get("interior_fraction_at_max", np.nan)),
                    ]
                    if "nucleus" in (fit.get("terms") or ()):
                        shares.append((f"{plain_name('nucleus')} share",
                                       fit.get("nucleus_fraction_at_max", np.nan)))
                    share_cols = st.columns(len(shares) + 1)
                    for column, (label, value) in zip(share_cols, shares):
                        column.metric(
                            label,
                            f"{100 * value:.1f} %" if np.isfinite(value) else "n/a",
                        )
                    if fit.get("mode") == "staged":
                        share_cols[-1].metric("Refinement passes", fit["n_iterations"])
                    else:
                        share_cols[-1].metric(
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
                        flat_table(
                            pd.DataFrame(fit["iterations"]).round(
                                {"Em_MPa": 4, "Ei_kPa": 4, "En_kPa": 4}
                            ).astype(str),
                        )
                        flat_table(
                            pd.DataFrame(
                                [
                                    {
                                        "stage": i + 1,
                                        "terms": ", ".join(
                                            term_label(t) for t in plan["terms"]
                                        ),
                                        "ε window": f"{plan['range'][0]:.3f} to {plan['range'][1]:.3f}",
                                        "points": res["n_points"],
                                        "R² in window": f"{float(res['r_squared']):.4f}",
                                    }
                                    for i, (plan, res) in enumerate(
                                        zip(fit["stage_plan"], fit["stages"])
                                    )
                                ]
                            ),
                            align_right=["points", "R² in window"],
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
                        st.markdown(f"**{plain_name('nucleus')} onset scan**")
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
                        + (
                            f"R_deep = {fit.get('R0', 0.0) * 1e6:.3f} um   "
                            f"(the cell's own, since the deep layer runs its "
                            f"length)\n"
                            if st.session_state["cell_type"] in HAS_SARCOMERES
                            else f"R_deep = "
                                 f"{fit.get('R_nucleus', float('nan')) * 1e6:.3f} um\n"
                        ) +
                        f"h_membrane = {st.session_state['membrane_thickness_nm']:.2f} nm\n"
                        f"Am = {fit['Am']:.4e} N/Pa   (F_membrane = Am*Em*eps^3)\n"
                        f"Ai = {fit['Ai']:.4e} N/Pa   (F_cyto = Ai*Ec*eps^1.5)\n"
                        f"An = {fit.get('An', float('nan')):.4e} N/Pa   "
                        f"(F_deep = An*En*<eps-eps2>^1.5)",
                        language="text",
                    )

                # ---------------------------------------------- the maths last
                # The page reads: choose, fit, look at the curve, read what the
                # cell did, and only then the equation and the working. Putting
                # the maths under the plot, as it was, meant scrolling past it
                # to reach the numbers it was the working for.
                # The equation, its coefficients and their uncertainties used
                # to be written out here, in the narrow results column, where
                # it was the longest thing on the page and the numbers it
                # repeats are already in the strip at the top. It lives
                # behind one line now, with the button that writes it on the
                # curve.
                with st.expander("🧮 The equation that was fitted, with every "
                                 "coefficient", expanded=False):
                    send_to_plot_button(
                        "equation", "The fitted equation",
                        {"text": equation_text(fit, style.force_unit)},
                        key="equation",
                        help_text="Writes the equation with this cell's numbers "
                                  "in a box on the curve.",
                    )
                    fitted_equation(fit, unit=style.force_unit, heading=False)
                    st.caption(
                        "The measured exponent, the boundaries drawn on it and "
                        "the algebra of each stretch are on the **📈 Log curve "
                        "and boundaries** tab."
                    )

                # No "how this fit was calculated" panel. What it said is now
                # said where it is needed: the criterion sits under the search
                # button that applies it, and the equation below carries the
                # range, the boundaries, the confinement and every fitted number
                # with its uncertainty.

                with diagnostics_box:
                    st.markdown("**∑ What the search does, in maths**")
                    show_search_maths()

                # No block of the answer repeated at the foot. Every number in
                # it is above: the moduli in the metrics row, the range and the
                # boundaries in the sentence under the results heading, and the
                # coefficients in the equation.

            if results_area is not None:
                results_area.__exit__(None, None, None)

        section(
            "4 · Video and database" if piecewise_mode
            else "7 · Video and database" if segmented
            else "6 · Video and database"
        )
        store = st.session_state.get("onedrive_store")
        ready = store is not None

        v1, v2 = st.columns([2, 1])
        with v1:
            main_video = st.file_uploader(
                "Compression video for this cell",
                type=["mp4", "avi", "mov", "wmv", "mkv"],
                key="video_file_main",
                help="The same video the **Compression video** tab works on: "
                "upload it in either place and both see it. Outlining the "
                "cell, measuring the probe and choosing the frame all happen "
                "on that tab.",
            )
            if not VIDEO_IMPORT_ERROR and adopt_video(main_video, "video_file_main"):
                st.rerun()
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
                    pinned = st.session_state.get("video_saved_frame_index")
                    st.image(
                        thumb,
                        caption=(f"frame {pinned}, chosen in the video tab"
                                 if pinned is not None
                                 else "frame stored with the cell"),
                        width=190,
                    )
                    if pinned is not None and st.button("Use a different frame"):
                        st.session_state["video_saved_frame"] = None
                        st.session_state["video_saved_frame_index"] = None
                        st.rerun()
                st.caption(
                    "To choose the frame, outline the cell by hand or set the "
                    "scale from the probe, go to the **Compression video** "
                    "tab and press **📸 Use this frame for the cell**. What "
                    "you pin there is what is sent from here."
                )
            else:
                st.caption("No video loaded for this cell.")

        have_fit = bool(fit and fit.get("success"))
        named = bool(st.session_state["cell_name"].strip())

        # Say what is blocking the send, in the order you would fix it. A
        # greyed-out button with no explanation is the same as a broken one.
        # What is missing, and it is deliberately not one flat list: every
        # destination needs the name and the fit, but each needs a different
        # connection, and filtering one list by substring is how the download
        # button ended up disabled by a OneDrive problem it does not have.
        common_blockers = []
        if not named:
            common_blockers.append("give the cell a name in section 1")
        if not have_fit:
            common_blockers.append("fit the curve above")

        st.checkbox(
            "Also upload the video file itself",
            key="upload_video_with_cell",
            help="Off sends the cell without the video; the record, curve "
            "and fit go either way. On also copies the video into the "
            "cell's folder so it can be opened from the database.",
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

        sheet_manager = st.session_state.get("gs_manager")
        sheet_ready = bool(st.session_state.get("db_enabled") and sheet_manager)

        c0, c1, c2 = st.columns(3)
        with c0:
            sheet_blockers = list(common_blockers)
            if not sheet_ready:
                sheet_blockers.append("connect the Google Sheet in the sidebar")
            if sheet_ready:
                # Which file the row is about to go into. "Connected" is not
                # the same as "connected to the sheet you are looking at",
                # and a row that lands in the wrong spreadsheet looks
                # exactly like a button that did nothing. Asked for rather
                # than assumed: an older google_sheets_manager.py has no
                # describe(), and a missing caption must not take the send
                # button down with it.
                where = getattr(sheet_manager, "describe", None)
                st.caption(
                    f"Writes into {where()}." if callable(where)
                    else "Writes one row into the connected sheet."
                )
            if st.button(
                "📗 Send to Google Sheet", type="primary",
                disabled=bool(sheet_blockers), **STRETCH,
            ):
                try:
                    ok, message = send_cell_to_sheet(sheet_manager, fit, date_acquired)
                    (st.success if ok else st.error)(message)
                    if not ok:
                        st.caption(
                            "Nothing was written. Press **🧪 Check it can "
                            "write** in the sidebar: the usual cause is that "
                            "the sheet is shared with the service account as "
                            "a Viewer rather than an Editor."
                        )
                    if ok:
                        # The database tab reads the sheet once and keeps it;
                        # a row just added has to reach it without a refresh
                        # by hand.
                        st.session_state["sheet_rows"] = None
                except Exception as exc:
                    st.error(f"Could not write the row: {exc}")
                    st.caption(
                        "That is the error Google returned. If it mentions "
                        "permission or quota, the sheet is not shared with "
                        "the service account as an Editor."
                    )
            if st.button(
                "📈 Also save the curve as a tab",
                disabled=bool(sheet_blockers), **STRETCH,
            ):
                try:
                    ok, message, url = sheet_manager.save_curve(
                        st.session_state["cell_name"].strip(),
                        epsilon, force_N, fitted,
                    )
                    if ok:
                        st.success(message)
                        if url:
                            st.caption(f"[Open the curve tab]({url})")
                    else:
                        st.error(message)
                except Exception as exc:
                    st.error(f"Could not save the curve: {exc}")
            st.caption(
                "The row is the summary; the curve tab is every measured "
                "point, force against relative deformation, in the same "
                "spreadsheet, next to the summary row."
                if not sheet_blockers
                else "To enable: " + ", then ".join(sheet_blockers) + "."
            )
        with c1:
            drive_store = st.session_state.get("onedrive_store")
            drive_blockers = list(common_blockers)
            if drive_store is None:
                drive_blockers.append("connect OneDrive in the sidebar")
            if st.button(
                "☁️ Send to OneDrive",
                disabled=bool(drive_blockers), **STRETCH,
            ):
                try:
                    with st.spinner("Uploading to OneDrive…"):
                        saved = send_cell_to_store(
                            drive_store, fit, epsilon, force_N, fitted,
                            date_acquired, model, stage_plan,
                        )
                    st.success(
                        f"Saved **{saved['cell_id']}** to OneDrive."
                        + (f" [Open the video]({saved['video_url']})"
                           if saved.get("video_url") else "")
                    )
                    st.session_state["archive_index"] = None
                except Exception as exc:
                    st.error(f"Could not save: {exc}")
            st.caption(
                "A folder per cell: the record, the curve, the frame and the "
                "video. Your own storage, no administrator needed."
                if not drive_blockers
                else "To enable: " + ", then ".join(drive_blockers) + "."
            )
        with c2:
            # OneDrive needs a one-time sign-in. This needs nothing at all
            # and writes exactly the same record, so an analysis is never
            # stuck inside the app waiting on a credential.
            # The download needs nothing beyond a named, fitted cell.
            download_blockers = list(common_blockers)
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


# ================================================= TAB: all cells together ==

with tab_cells:
    section("📚 All cells")
    if HAS_PIECEWISE:
        all_cells_tab()
    else:
        piecewise_problem_note()


# =============================================== TAB 2: balloon and spring ==
#
# The Lulevich model has two springs and one question about them: which one
# the plates meet first. Everything in this tab is that question. It is kept
# out of the analysis tab because it is a different job: the analysis tab
# fits a cell, this one asks what shape of cell the curve is describing.

with tab_explore:
    section("The log curve, and the boundaries it says the cell has")

    explore_data = st.session_state.get("data")
    if not explore_data:
        st.info(
            "Load a force curve in the **Force curve analysis** tab and the "
            "curve's own power law appears here."
        )
    else:
        eps_ex = explore_data["epsilon"]
        force_ex = explore_data["force_N"]
        style_ex = current_style(force_ex)
        model_ex = build_model(eps_ex, force_ex)

        # The range the analysis tab settled on, so the two tabs are talking
        # about the same stretch of the same curve. Repeating the range
        # controls here would let them drift apart and give two answers.
        hi_ex = float(np.clip(
            st.session_state.get("window_end", float(eps_ex.max())),
            1e-3, float(eps_ex.max()),
        ))
        lo_ex = float(np.clip(
            st.session_state.get("window_start", 0.0), 0.0,
            max(hi_ex - 1e-3, 0.0),
        ))
        terms_ex = active_terms() or ("membrane", "interior")
        fit_ex = shown_fit()
        pw_ex = bool(fit_ex and fit_ex.get("piecewise"))
        if pw_ex:
            on_ex = [label.split(" ", 1)[1].lower() for name, label, *_ in
                     PW_COMPONENTS if name not in piecewise_off()]
            st.caption(
                f"Piecewise sharing, ε₁, ε₂, ε₃ as on the analysis tab"
                + (f" (`fit {fit_ex['fit_id']}`)" if fit_ex.get("fit_id") else "")
                + ", with " + ", ".join(on_ex) + ". Change them there and "
                "this follows."
            )
        else:
            st.caption(
                f"ε = {lo_ex:.3f} to {hi_ex:.3f}, with the components ticked on "
                "the analysis tab: "
                + ", ".join(plain_name(t).lower() for t in terms_ex)
                + ". Change either there and this follows."
            )

        st.markdown(
            "Every law in this model is a straight line on log-log axes, "
            "and a different one: a stretching shell rises as ε³, a "
            "Hertzian contact as ε³ᐟ², a network already under tension as "
            "ε. So the slope of ln F against ln ε **is** the reading of "
            "which law is carrying the load, taken from the data with no "
            "model in it, and a boundary is where that reading changes."
        )
        st.latex(
            r"p(\varepsilon) \;=\; \frac{d\log F}{d\log \varepsilon} "
            r"\;=\; \sum_k w_k\,p_k, \qquad "
            r"w_k = \frac{a_k E_k g_k(\varepsilon)}"
            r"{\sum_j a_j E_j g_j(\varepsilon)}"
        )

        st.markdown("#### The curve on log-log axes")
        results_ex = st.session_state.get("results") or {}
        st.plotly_chart(
            log_log_figure(
                eps_ex, force_ex,
                results_ex.get("fitted_N") if fit_ex is not None
                and results_ex.get("fit") is fit_ex else None,
                fit_ex, style_ex,
            ),
            key="log_log_curve", **STRETCH,
        )
        st.caption(
            "log F against log ε: a law F ∝ εᵖ is a straight line of slope p, "
            "so a straight stretch is one law and a bend is a boundary. The "
            "dashed lines are the fit's boundaries; the dotted ones are the "
            "guide lines below, each tagged with the p it is set to."
        )
        log_guide_controls()

        st.markdown("#### Where the curve changes its power law")
        where_the_power_law_changes(fit_ex, model_ex, style_ex)

        if pw_ex:
            pw_result_ex = results_ex.get("piecewise_result")
            if pw_result_ex:
                st.markdown("#### Who carries the load, where")
                st.plotly_chart(piecewise_share_figure(pw_result_ex, style_ex),
                                key="pw_share_plot", **STRETCH)
                st.markdown("#### What the curve is fitting, stretch by stretch")
                for line in piecewise_equations_latex(
                        pw_result_ex["boundaries_pct"],
                        pw_result_ex.get("ranges") or {}, piecewise_off()):
                    st.latex(line)
        else:
            share_of_force_plot(fit_ex, model_ex, style_ex)
            st.markdown("#### What the curve is fitting, stretch by stretch")
            stage_algebra(fit_ex)

        with st.expander(
            "📋 The materials, the law each obeys, and why they separate"
        ):
            materials_table(
                terms_for(st.session_state["cell_type"]),
                caption=(
                    "Every material this cell type can be fitted with. The "
                    "last column is the one that decides whether a modulus "
                    "means anything."
                ),
            )
            separation_rule()

        st.markdown("#### Find the boundaries from this curve")
        if pw_ex:
            st.caption(
                "With piecewise sharing, ε₁, ε₂, ε₃ are placed by **▶ Place "
                "ε₁, ε₂, ε₃** on the analysis tab, which compares the power "
                "law read off this curve with fitting everything; the routes "
                "are in its **ε routes compared** tab."
            )
            placements_ex = st.session_state.get("pw_placements") or {}
            power_ex = placements_ex.get("power") or {}
            if power_ex.get("success"):
                st.plotly_chart(power_law_figure(power_ex, piecewise_boundaries()),
                                key="pw_power_law_explore", **STRETCH)
        else:
            refine_boundaries_control(model_ex, lo_ex, hi_ex, terms_ex)
            left_alone = st.session_state.pop("_boundaries_left_alone", None)
            if left_alone:
                st.caption(
                    "Left as you set "
                    + ", ".join(plain_name(t).lower() for t in left_alone)
                    + ": those ranges were moved by hand, so the new "
                    "boundaries did not touch them."
                )
            search_results_as_equations()
            why_this_search(model_ex)


# ==================================================== TAB 3: video analysis ==

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
                "Upload video", type=["mp4", "avi", "mov", "wmv", "mkv"],
                key="video_file",
                help="The same video as the one in section 7 of the analysis "
                "tab. Uploading in either place loads it for both.",
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
                    st.session_state["video_saved_frame"] = None
                    st.session_state["video_saved_frame_index"] = None
                    st.rerun()
                except Exception as exc:
                    st.error(f"{exc}")

        # Two things about the cell that only the video can tell you, and
        # that the spreadsheet keeps beside the moduli: how tall the cell
        # actually was, and anything you noticed while watching it. The
        # height is filled in from the measurement below when the scale is
        # set, and can be typed over.
        vh1, vh2 = st.columns([1, 2])
        with vh1:
            st.number_input(
                "Height from the video (µm)",
                min_value=0.0, max_value=200.0, step=0.1, format="%.2f",
                key="video_height_um",
                help="The cell's height measured off the frame, in "
                "micrometres. 0 means not measured. It is written to the "
                "spreadsheet as its own column, beside the height that was "
                "typed into the geometry settings, so the two can be "
                "compared rather than one quietly replacing the other.",
            )
        with vh2:
            st.text_input(
                "Video comment",
                key="video_comment",
                placeholder="what the video showed: the probe slipped, the "
                            "cell rolled, nothing unusual…",
                help="Goes into the spreadsheet with the numbers. This is "
                "the column somebody reads first when a modulus looks odd.",
            )

        if adopt_video(uploaded_video, "video_file"):
            st.rerun()

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

            with st.expander("✍️ Draw the cell yourself, and set the scale"):
                m1, m2 = st.columns(2)
                with m1:
                    st.checkbox(
                        "Select the cell by hand", key="video_manual_cell",
                        help="Use this whenever the outline below is on the "
                        "wrong object. Your box is used exactly as drawn, and "
                        "the detector is not consulted at all.",
                    )
                    st.slider("Cell: left and right", 0.0, 1.0, step=0.01,
                              key="video_cell_box_x",
                              disabled=not st.session_state["video_manual_cell"])
                    st.slider("Cell: top and bottom", 0.0, 1.0, step=0.01,
                              key="video_cell_box_y",
                              disabled=not st.session_state["video_manual_cell"])
                with m2:
                    st.checkbox(
                        "Measure the probe to set the scale",
                        key="video_use_probe_scale",
                        help="The cantilever is the one object whose size is "
                        "known exactly, so measuring it across the image "
                        "turns every pixel into micrometres.",
                    )
                    st.slider("Probe: left and right edge", 0.0, 1.0, step=0.01,
                              key="video_probe_box_x",
                              disabled=not st.session_state["video_use_probe_scale"])
                    st.number_input(
                        "Probe width (µm)", min_value=1.0, max_value=500.0,
                        step=5.0, key="video_probe_width_um",
                        disabled=not st.session_state["video_use_probe_scale"],
                        help="A tipless cantilever is usually 40 or 60 µm wide.",
                    )
                st.slider(
                    "How much room around the cell in the saved frame",
                    0.0, 2.0, step=0.05, key="video_crop_pad",
                    help="0 crops tight to the outline, which is what made the "
                    "stored frame look zoomed in. Higher shows more of the "
                    "field around it.",
                )

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
            # One reader for the whole app, so what is set here is what every
            # other part of the app sees.
            frame, det, nucleus, probe_box, scale_um_px = detection_at(preview_frame)
            scale_detail = ""
            if st.session_state["video_use_probe_scale"] and frame is not None:
                px = st.session_state["video_probe_box_x"]
                _, scale_detail = va.scale_from_probe(
                    (px[0], 0.0, px[1], 0.0), frame.shape,
                    float(st.session_state["video_probe_width_um"]),
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
                if scale_um_px:
                    st.caption(f"Scale: {scale_detail}")
                elif st.session_state["video_use_probe_scale"]:
                    st.caption(scale_detail or "Scale not set.")

                if det and det.get("found"):
                    st.success("Cell drawn by hand" if det.get("manual")
                               else "Cell detected")
                    if scale_um_px:
                        st.metric(
                            "Cell height",
                            f"{det['height_px'] * scale_um_px:.2f} µm",
                            delta=f"{det['height_px']:.0f} px", delta_color="off",
                        )
                        st.metric(
                            "Cell width",
                            f"{det['width_px'] * scale_um_px:.2f} µm",
                            delta=f"{det['width_px']:.0f} px", delta_color="off",
                        )
                        st.caption(
                            "Compare the height against the cell height in "
                            "section 1: they should agree, and the moduli are "
                            "sensitive to that number."
                        )
                        if st.button("Use this as the video height",
                                     key="adopt_video_height", **STRETCH):
                            # Staged rather than written: the number input
                            # for it was built earlier this pass, and
                            # Streamlit refuses a write to a widget's key
                            # once its widget exists.
                            st.session_state["_pending_settings"] = {
                                "video_height_um": round(
                                    float(det["height_px"] * scale_um_px), 3
                                )
                            }
                            st.rerun()
                        st.caption(
                            "It goes into the spreadsheet as **Height (um)**, "
                            "beside the height typed into the geometry "
                            "settings rather than instead of it."
                        )
                    else:
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
                    annotated = va.annotate(frame, det, label=label,
                                            nucleus=nucleus, probe=probe_box)
                    cropped = va.crop(
                        annotated, det,
                        pad_frac=float(st.session_state["video_crop_pad"]),
                    )
                    if st.button("📸 Use this frame for the cell", type="primary",
                                 **STRETCH):
                        # Pinned rather than recomputed: the frame you looked
                        # at and approved is the one that gets stored, not
                        # whatever the detector produces later from settings
                        # that have since moved.
                        st.session_state["video_saved_frame"] = png_bytes(cropped)
                        st.session_state["video_saved_frame_index"] = int(preview_frame)
                        st.success(
                            f"Frame {preview_frame} is now the picture stored "
                            f"with this cell. It appears in section 7."
                        )
                    st.download_button(
                        "📷 Save this frame to a file",
                        data=png_bytes(annotated),
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
                        st.dataframe(safe_frame(out.head(25)),
                                     hide_index=True, **STRETCH)

                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button(
                                "📥 Download CSV",
                                data=safe_frame(out).to_csv(index=False),
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
    store = st.session_state.get("onedrive_store")
    sheet = (st.session_state.get("gs_manager")
             if st.session_state.get("db_enabled") else None)

    # Two places a database can live, and the sheet wins when both are
    # connected: a sheet is the one people actually open, and a row in it is
    # the record of a cell whether or not its files were ever archived.
    # OneDrive holds the files, so it stays available beside it.
    if store is not None and sheet is not None:
        source = st.radio(
            "Read the cells from", ["Google Sheet", "OneDrive"],
            horizontal=True, key="db_source",
            help="The sheet holds one row per cell. OneDrive holds the whole "
                 "folder for each cell: the curve, the frame and the video.",
        )
    elif sheet is not None:
        source = "Google Sheet"
    else:
        source = "OneDrive"

    if source == "Google Sheet":
        st.caption("From the connected Google Sheet, one row per cell.")
        url = sheet.get_spreadsheet_url()
        if url:
            st.caption(f"[Open the sheet in Google Sheets]({url})")
        if st.button("🔄 Refresh from the sheet", key="db_sheet_refresh"):
            st.session_state["sheet_rows"] = None
        if st.session_state.get("sheet_rows") is None:
            with st.spinner("Reading the sheet…"):
                try:
                    st.session_state["sheet_rows"] = sheet.get_all_cells()
                except Exception as exc:
                    st.error(str(exc))
                    st.session_state["sheet_rows"] = pd.DataFrame()
        rows = st.session_state.get("sheet_rows")
        if rows is None or rows.empty:
            st.info(
                "The sheet has no cells in it yet. Fit a curve, then press "
                "**Send to Google Sheet** under *Video and database*."
            )
        else:
            needle = st.text_input(
                "Filter", placeholder="cell name, date, model or operator",
                key="sheet_search", label_visibility="collapsed",
            )
            shown = rows
            if needle:
                low = needle.lower()
                shown = rows[rows.apply(
                    lambda row: low in " ".join(str(v).lower() for v in row.values),
                    axis=1,
                )]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cells", len(rows))
            m2.metric("Shown", len(shown))
            for column, label, unit, target in (
                ("Young's Modulus (Em, MPa)", "Median Eₘ", "MPa", m3),
                ("Young's Modulus (Ei, kPa)", "Median E_c", "kPa", m4),
            ):
                values = numeric_column(shown, column)
                values = values[values > 0]
                target.metric(
                    label, f"{values.median():.3g} {unit}" if len(values) else "n/a"
                )
            st.dataframe(safe_frame(shown), hide_index=True, **STRETCH)
            st.download_button(
                "⬇️ These rows as CSV",
                safe_frame(shown).to_csv(index=False).encode("utf-8"),
                file_name="afm_cells.csv", mime="text/csv",
                key="sheet_rows_csv",
            )
            st.caption(
                "Rows come from the sheet as they are stored. Edit them in "
                "Google Sheets and press refresh; nothing here writes back "
                "except **Send to Google Sheet** after a fit."
            )
    elif ONEDRIVE_IMPORT_ERROR:
        st.error(onedrive_load_problem())
        st.caption(f"Python said: {ONEDRIVE_IMPORT_ERROR}")
    elif store is None:
        st.info(
            "No database connected yet. Either connect a Google Sheet, which "
            "keeps one row per cell, or connect OneDrive, which keeps the "
            "whole folder for each cell. Both are in the sidebar."
        )
        with st.expander("📖 Setting up OneDrive, step by step", expanded=True):
            onedrive_instructions()
    else:
        head1, head2, head3 = st.columns([1, 1, 2])
        with head1:
            if st.button("🔄 Refresh", **STRETCH):
                st.session_state["archive_index"] = None
        with head2:
            if st.button("🛠️ Rebuild index", **STRETCH,
                         help="Reads every cell's record.json and writes a fresh "
                              "index. Slower, but it recovers a stale or damaged one."):
                progress = st.progress(0.0, text="Reading cells…")
                try:
                    st.session_state["archive_index"] = store.rebuild_index(
                        progress=lambda n, total, name: progress.progress(
                            n / max(total, 1), text=f"{n}/{total}  {name}"
                        )
                    )
                    progress.empty()
                    st.success("Index rebuilt.")
                except Exception as exc:
                    progress.empty()
                    st.error(str(exc))

        if st.session_state.get("archive_index") is None:
            try:
                with st.spinner("Loading the index…"):
                    st.session_state["archive_index"] = store.load_index()
            except Exception as exc:
                st.error(str(exc))
                st.session_state["archive_index"] = pd.DataFrame()

        index = st.session_state.get("archive_index")
        if index is None or index.empty:
            st.info("No cells stored yet. Fit a curve and press **Send to OneDrive**.")
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
                values = numeric_column(shown, column)
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
                            # OneDrive addresses a stored file by its path
                            # where Box used an opaque id.
                            thumb_id = record.get("thumbnail_path") or record.get(
                                "thumbnail_file_id"
                            )
                            image = None
                            if thumb_id and str(thumb_id) not in ("", "nan"):
                                image = cached_thumbnail(
                                    st.session_state["onedrive_root"], str(thumb_id)
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
                    safe_frame(table.round(
                        {c: 4 for c in numeric if c in table.columns}
                    )),
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
                    st.session_state["archive_index"] = None
                    if done:
                        st.success(f"Refitted {len(done)} cell(s).")
                        flat_table(
                            pd.DataFrame(
                                [
                                    {
                                        "cell": r["cell_id"],
                                        "Eₘ (MPa)": f"{r['Em_MPa']:.4f}",
                                        "Ec (kPa)": f"{r['Ec_kPa']:.4f}",
                                        "Eₙ (kPa)": f"{r['En_kPa']:.4f}",
                                        "R²": f"{r['r_squared']:.4f}",
                                    }
                                    for r in done
                                ]
                            ),
                            align_right=[
                                "Eₘ (MPa)", "Ec (kPa)", "Eₙ (kPa)", "R²",
                            ],
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
                                _pair = tuple(settings["combined_window"])
                                pending["window_combined"] = _pair
                                pending["window_start"] = round(float(_pair[0]), 4)
                                pending["window_end"] = round(float(_pair[1]), 4)
                            for term in ALL_TERMS:
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
                    data=safe_frame(index).to_csv(index=False),
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
            if fit.get("coupling") == "piecewise":
                # The four-regime fit has six moduli, not two, and no
                # bending constant.
                # The same rows, the same strings, as the analysis tab:
                # each value with its ± beside it.
                results_tab_metrics(fit)
                if fit.get("fit_id"):
                    st.caption(f"fit {fit['fit_id']}, the same fit as on the "
                               "📊 Force curve analysis tab.")
                bounds = fit["piecewise"]["boundaries_pct"]
                st.caption(
                    "4-regime piecewise fit · "
                    + ", ".join(f"{n} = {v:.2f} %"
                                for n, v in zip(EPS_NAMES, bounds[1:4]))
                    + f", end {bounds[4]:.1f} % "
                    f"({fit['piecewise'].get('boundary_source', '')}) · "
                    f"{fit['n_points']} points"
                    + (" · membrane acting throughout"
                       if fit["piecewise"].get("membrane_throughout") else "")
                )
            else:
                results_tab_metrics(fit)
                st.caption(
                    f"Window ε ∈ [{fit['epsilon_range'][0]:.3f}, {fit['epsilon_range'][1]:.3f}] · "
                    f"{fit['n_points']} points · {fit['weighting']} weighting · "
                    f"bending constant Kₘ = {fit.get('Km_kT', float('nan')):.3g} k_BT"
                )

        # A clean figure: the measured points and the fitted curve, nothing
        # else. Everything that explains the fit is on the analysis tab.
        clean_style = replace_style(
            style, range_note=None, show_component_heights=False,
            show_components=False, show_fit_window=False,
        )
        results_figure = force_curve_figure(
            results["epsilon"],
            results["force_N"],
            clean_style,
            title=results["cell_name"] or "Force vs relative deformation",
            fit_force_N=results["fitted_N"],
        )
        finish_legend_below(results_figure, clean_style, 0)
        st.plotly_chart(results_figure, **STRETCH, key="results_tab_plot")


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
        if fit.get("piecewise"):
            # Every coefficient, modulus and anchor of the four-regime fit,
            # flat so the CSV gets one column each.
            summary["piecewise_boundaries_pct"] = " / ".join(
                f"{b:g}" for b in fit["piecewise"]["boundaries_pct"]
            )
            summary.update(fit["piecewise"]["flat"])

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


# ------------------------------------------------ settings kept offstage --
# Last thing in the run, on purpose: see keep_unrendered_settings. Every
# run, whichever way the load is shared, so switching the sharing back finds
# the other way's settings (its ε, its ranges) as they were left.
keep_unrendered_settings()
