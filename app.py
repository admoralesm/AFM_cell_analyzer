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

from lulevich_model import LulevichModel
from plot_utils import (
    FORCE_UNITS,
    INPUT_FORCE_UNITS,
    PlotStyle,
    autoscale_unit,
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


def section(text: str):
    st.markdown(f'<div class="section">{text}</div>', unsafe_allow_html=True)


def hint(text: str):
    st.markdown(f'<p class="hint">{text}</p>', unsafe_allow_html=True)


# ========================================================== session state ==

DEFAULTS = {
    # display
    "force_unit": "auto",
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
    # geometry / model
    "radius_mode": "From height",
    "radius_aspect": 0.55,
    "cell_radius_um": 4.45,
    "membrane_thickness_nm": 4.0,
    "poisson_membrane": 0.50,
    "poisson_interior": 0.50,
    # fitting
    "fit_mode": "Simultaneous (one window)",
    "sequential_order": "interior-first",
    "refine_iterations": 3,
    "lock_range": False,
    "range_presets": {},
    "range_mode": "Auto (detected)",
    "fit_terms": "Membrane + interior",
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

if st.session_state.pop("_reset_requested", False):
    # Widget-backed keys can only be reassigned before their widget is built,
    # so the reset button sets a flag and the actual reset happens here.
    for key, value in DEFAULTS.items():
        if key not in ("data", "results", "gs_manager"):
            st.session_state[key] = value
    st.session_state.pop("manual_range", None)

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
    return LulevichModel(
        force_N,
        epsilon,
        cell_height=height_m,
        cell_radius=radius_m,
        membrane_thickness=float(st.session_state["membrane_thickness_nm"]) * 1e-9,
        poisson_membrane=float(st.session_state["poisson_membrane"]),
        poisson_interior=float(st.session_state["poisson_interior"]),
    )


def selected_terms():
    return {
        "Membrane + interior": ("membrane", "interior"),
        "Membrane only (ε³)": ("membrane",),
        "Interior only (ε³ᐟ²)": ("interior",),
    }[st.session_state["fit_terms"]]


# ================================================================ sidebar ==

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    hint("These apply everywhere in the app.")

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
            "Membrane thickness hₘ (nm)",
            min_value=0.5,
            max_value=100.0,
            step=0.5,
            key="membrane_thickness_nm",
        )
        st.slider("Poisson ratio, membrane νₘ", 0.0, 0.5, step=0.01, key="poisson_membrane")
        st.slider("Poisson ratio, interior νᵢ", 0.0, 0.5, step=0.01, key="poisson_interior")
        st.caption("0.5 = incompressible, the usual choice for living cells.")

    with st.expander("📈 Fitting", expanded=True):
        st.radio(
            "Strategy",
            ["Simultaneous (one window)", "Sequential (separate windows)"],
            key="fit_mode",
            help="Simultaneous solves for Eₘ and Eᵢ together on one window. "
            "Sequential measures Eᵢ on a low-ε window where the ε³ᐟ² term "
            "dominates, then Eₘ on a separate high-ε window where ε³ takes over.",
        )
        if st.session_state["fit_mode"] == "Sequential (separate windows)":
            st.selectbox(
                "Measure first",
                ["interior-first", "membrane-first"],
                key="sequential_order",
                help="Which modulus is fitted on its own window before the other "
                "is held fixed. interior-first matches the physics.",
            )
            st.slider(
                "Refinement passes",
                1,
                8,
                key="refine_iterations",
                help="One pass is biased: the first window has no knowledge of the "
                "other term and absorbs it. Repeating the pair of fits removes that. "
                "3 is usually converged; 1 gives the plain one-pass result.",
            )
        else:
            st.selectbox(
                "Model terms",
                ["Membrane + interior", "Membrane only (ε³)", "Interior only (ε³ᐟ²)"],
                key="fit_terms",
            )
            st.radio(
                "Fit window",
                ["Auto (detected)", "Manual"],
                key="range_mode",
                horizontal=True,
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
        st.slider("Axis title size", 12, 44, key="axis_title_size")
        st.slider("Tick label size", 10, 36, key="tick_size")
        st.slider("Axis line thickness", 1, 8, key="axis_width")
        st.checkbox("Bold axis titles and ticks", key="bold_axes")
        st.checkbox("Show grid", key="show_grid")
        st.checkbox("Log-log axes", key="log_scale", help="A power law is a straight line here.")
        st.checkbox("Show model components", key="show_components")
        st.checkbox("Shade the fit window", key="show_fit_window")
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

        # ---------------------------------------------------- fit window ---
        section("3 · Fit window")
        model = build_model(epsilon, force_N)
        auto_range = model.auto_detect_elastic_range()
        rupture = model.results.get("rupture", {})

        eps_lo_data = float(max(epsilon.min(), 0.0))
        eps_hi_data = float(epsilon.max())
        step = max((eps_hi_data - eps_lo_data) / 200.0, 1e-4)
        sequential = st.session_state["fit_mode"] == "Sequential (separate windows)"

        def clamp_range(pair, fallback):
            """Keep a stored range inside the current curve's bounds."""
            try:
                lo, hi = float(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError):
                return fallback
            lo = float(np.clip(lo, eps_lo_data, eps_hi_data))
            hi = float(np.clip(hi, eps_lo_data, eps_hi_data))
            return (lo, hi) if lo < hi else fallback

        interior_range = membrane_range = None

        if sequential:
            suggestion = model.suggest_sequential_windows()
            st.caption(suggestion["note"])
            default_int = clamp_range(
                suggestion["interior_range"], (eps_lo_data, (eps_lo_data + eps_hi_data) / 2)
            )
            default_mem = clamp_range(
                suggestion["membrane_range"], ((eps_lo_data + eps_hi_data) / 2, eps_hi_data)
            )
            st.session_state["interior_range"] = clamp_range(
                st.session_state.get("interior_range"), default_int
            )
            st.session_state["membrane_range"] = clamp_range(
                st.session_state.get("membrane_range"), default_mem
            )

            c1, c2 = st.columns(2)
            with c1:
                interior_range = st.slider(
                    "Cytoskeleton window (low ε, fits Eᵢ)",
                    min_value=eps_lo_data,
                    max_value=eps_hi_data,
                    step=step,
                    key="interior_range",
                    help="Where the Hertzian ε³ᐟ² term dominates.",
                )
                st.caption(
                    f"{int(((epsilon >= interior_range[0]) & (epsilon <= interior_range[1])).sum())} points"
                )
            with c2:
                membrane_range = st.slider(
                    "Membrane window (high ε, fits Eₘ)",
                    min_value=eps_lo_data,
                    max_value=eps_hi_data,
                    step=step,
                    key="membrane_range",
                    help="Where the balloon ε³ term dominates.",
                )
                st.caption(
                    f"{int(((epsilon >= membrane_range[0]) & (epsilon <= membrane_range[1])).sum())} points"
                )
            if st.button("↺ Use suggested split", **STRETCH):
                st.session_state["interior_range"] = default_int
                st.session_state["membrane_range"] = default_mem
                st.rerun()
            fit_lo = min(interior_range[0], membrane_range[0])
            fit_hi = max(interior_range[1], membrane_range[1])

        elif st.session_state["range_mode"] == "Auto (detected)" and not st.session_state["lock_range"]:
            fit_lo = float(np.clip(auto_range["elastic_epsilon_min"], eps_lo_data, eps_hi_data))
            fit_hi = float(np.clip(auto_range["elastic_epsilon_max"], eps_lo_data, eps_hi_data))
            st.success(
                f"Auto window: ε ∈ [{fit_lo:.3f}, {fit_hi:.3f}] "
                f"({auto_range['n_points']} points) · "
                f"rupture detection: {auto_range['rupture_method']}"
            )
            hint("Switch to **Manual** in the sidebar to drag the window yourself.")
        else:
            default = clamp_range(
                (auto_range["elastic_epsilon_min"], auto_range["elastic_epsilon_max"]),
                (eps_lo_data, eps_hi_data),
            )
            # A locked range survives new uploads; an unlocked one is only
            # reset when it no longer fits inside the current curve.
            stored = st.session_state.get("manual_range")
            if st.session_state["lock_range"] and stored:
                st.session_state["manual_range"] = clamp_range(stored, default)
            elif not stored or not (eps_lo_data <= stored[0] < stored[1] <= eps_hi_data):
                st.session_state["manual_range"] = default
            fit_lo, fit_hi = st.slider(
                "Fitting range in ε",
                min_value=eps_lo_data,
                max_value=eps_hi_data,
                step=step,
                key="manual_range",
                help="Bounded by your actual data, not a fixed 0–0.5.",
            )
            n_in = int(((epsilon >= fit_lo) & (epsilon <= fit_hi)).sum())
            cols = st.columns(3)
            cols[0].metric("Points in window", n_in)
            cols[1].metric("Suggested ε min", f"{auto_range['elastic_epsilon_min']:.3f}")
            cols[2].metric("Suggested ε max", f"{auto_range['elastic_epsilon_max']:.3f}")

        # ------------------------------------------------- saved presets ---
        with st.expander("💾 Saved ranges", expanded=False):
            st.checkbox(
                "Lock this range (keep it when a new curve is loaded)",
                key="lock_range",
            )
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
                            "mode": "sequential" if sequential else "simultaneous",
                            "range": [float(fit_lo), float(fit_hi)],
                            "interior_range": [float(x) for x in interior_range]
                            if interior_range
                            else None,
                            "membrane_range": [float(x) for x in membrane_range]
                            if membrane_range
                            else None,
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
                        preset = presets[chosen]
                        if preset["mode"] == "sequential" and preset["interior_range"]:
                            st.session_state["fit_mode"] = "Sequential (separate windows)"
                            st.session_state["interior_range"] = clamp_range(
                                preset["interior_range"], (eps_lo_data, eps_hi_data)
                            )
                            st.session_state["membrane_range"] = clamp_range(
                                preset["membrane_range"], (eps_lo_data, eps_hi_data)
                            )
                        else:
                            st.session_state["fit_mode"] = "Simultaneous (one window)"
                            st.session_state["range_mode"] = "Manual"
                            st.session_state["manual_range"] = clamp_range(
                                preset["range"], (eps_lo_data, eps_hi_data)
                            )
                        st.session_state["lock_range"] = True
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
                                "mode": p["mode"],
                                "ε window": f"{p['range'][0]:.3f} to {p['range'][1]:.3f}",
                                "cytoskeleton": f"{p['interior_range'][0]:.3f} to {p['interior_range'][1]:.3f}"
                                if p.get("interior_range")
                                else "—",
                                "membrane": f"{p['membrane_range'][0]:.3f} to {p['membrane_range'][1]:.3f}"
                                if p.get("membrane_range")
                                else "—",
                                "saved": p["saved_at"],
                            }
                            for name, p in presets.items()
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
                    file_name="afm_range_presets.json",
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
        run = st.session_state["live_fit"] or st.button(
            "🚀 Fit curve", type="primary"
        )

        fit = None
        if run:
            if sequential:
                fit = model.fit_sequential(
                    interior_range=interior_range,
                    membrane_range=membrane_range,
                    weighting=st.session_state["weighting"],
                    fit_offset=st.session_state["fit_offset"],
                    order=st.session_state["sequential_order"],
                    refine_iterations=st.session_state["refine_iterations"],
                )
            else:
                fit = model.fit(
                    epsilon_min=fit_lo,
                    epsilon_max=fit_hi,
                    terms=selected_terms(),
                    fit_offset=st.session_state["fit_offset"],
                    weighting=st.session_state["weighting"],
                )

        if fit is None:
            st.info("Press **Fit curve**, or turn on live refitting in the sidebar.")
        elif not fit.get("success"):
            st.error(fit.get("error", "Fit failed."))
        else:
            fitted = model.combined_model(
                epsilon, fit["Em"], fit["Ei"], fit.get("force_offset", 0.0)
            )
            membrane = model.balloon_model_cubic(epsilon, fit["Em"])
            interior = model.hertzian_contact_model(epsilon, fit["Ei"])

            st.session_state["results"] = {
                "cell_name": st.session_state["cell_name"],
                "date_acquired": str(date_acquired),
                "cell_height_um": st.session_state["cell_height_um"],
                "spring_constant": st.session_state["spring_constant"],
                "video_link": st.session_state["video_link"],
                "epsilon": epsilon,
                "force_N": force_N,
                "fitted_N": fitted,
                "membrane_N": membrane,
                "interior_N": interior,
                "fit": fit,
                "fit_windows": (
                    [
                        {"range": tuple(interior_range), "label": "cytoskeleton (Eᵢ)",
                         "color": "#9467bd"},
                        {"range": tuple(membrane_range), "label": "membrane (Eₘ)",
                         "color": "#2ca02c"},
                    ]
                    if sequential
                    else [{"range": (fit_lo, fit_hi), "label": "fit window"}]
                ),
                "source": data["source"],
                "timestamp": datetime.now(),
            }

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Eₘ membrane",
                f"{fit['Em_MPa']:.3g} MPa",
                delta=f"± {fit['Em_MPa_std']:.2g}" if np.isfinite(fit["Em_MPa_std"]) else None,
                delta_color="off",
            )
            m2.metric(
                "Eᵢ interior",
                f"{fit['Ei_kPa']:.3g} kPa",
                delta=f"± {fit['Ei_kPa_std']:.2g}" if np.isfinite(fit["Ei_kPa_std"]) else None,
                delta_color="off",
            )
            m3.metric("R²", f"{fit['r_squared']:.4f}")
            rmse_disp, rmse_unit = from_newtons(fit["rmse"], style.force_unit)
            m4.metric("RMSE", f"{float(rmse_disp):.3g} {rmse_unit}")

            for message in fit["warnings"]:
                st.warning(message)

            # A loaded video gets a panel beside the plot showing the cell at
            # whichever point on the curve is selected.
            video_ready = (
                not VIDEO_IMPORT_ERROR
                and st.session_state.get("video_path")
                and st.session_state.get("video_info")
                and os.path.exists(st.session_state["video_path"])
                and st.session_state["video_show_panel"]
            )

            highlight = None
            selected_eps = None
            if video_ready:
                eps_lo_sel = float(max(epsilon.min(), 0.0))
                eps_hi_sel = float(epsilon.max())
                selected_eps = st.slider(
                    "Show the cell at ε =",
                    min_value=eps_lo_sel,
                    max_value=eps_hi_sel,
                    value=float(np.clip(fit_hi, eps_lo_sel, eps_hi_sel)),
                    step=max((eps_hi_sel - eps_lo_sel) / 200.0, 1e-4),
                    key="video_sync_eps",
                )
                nearest = int(np.argmin(np.abs(epsilon - selected_eps)))
                highlight = (float(epsilon[nearest]), float(force_N[nearest]))

            plot_col, video_col = st.columns([3, 1.15]) if video_ready else (st.container(), None)

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
                        fit_window=st.session_state["results"]["fit_windows"],
                        rupture_epsilon=rupture.get("epsilon")
                        if rupture.get("method") == "force-drop"
                        else None,
                        highlight=highlight,
                    ),
                    **STRETCH,
                    key="main_fit_plot",
                )

            if video_ready and video_col is not None:
                with video_col:
                    vpath = st.session_state["video_path"]
                    vinfo = st.session_state["video_info"]
                    frame_index = va.frame_for_epsilon(
                        selected_eps,
                        st.session_state["video_contact_frame"],
                        st.session_state["video_end_frame"] or (vinfo["n_frames"] - 1),
                        float(epsilon.max()),
                    )
                    vframe, vdet = cached_detection(
                        vpath,
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
                        shot = va.annotate(
                            vframe, vdet, label=f"ε = {highlight[0]:.3f}"
                        )
                        st.image(
                            va.crop(shot, vdet) if vdet and vdet.get("found") else shot,
                            caption=f"Frame {frame_index} · ε = {highlight[0]:.3f} · "
                            f"F = {float(force_here):.3g} {unit_here}",
                            **STRETCH,
                        )
                        if vdet and vdet.get("found"):
                            st.caption(f"Cell height {vdet['height_px']:.0f} px")
                        else:
                            st.caption("No cell detected in this frame.")
                        st.download_button(
                            "📷 Save screenshot",
                            data=png_bytes(shot),
                            file_name=(
                                f"{st.session_state['cell_name'] or 'cell'}"
                                f"_eps{highlight[0]:.3f}.png"
                            ),
                            mime="image/png",
                            **STRETCH,
                        )

            with st.expander("🔍 Fit diagnostics"):
                d1, d2, d3 = st.columns(3)
                d1.metric(
                    "Membrane share at ε_max",
                    f"{100 * fit['membrane_fraction_at_max']:.1f} %"
                    if np.isfinite(fit["membrane_fraction_at_max"])
                    else "n/a",
                )
                if fit.get("mode") == "sequential":
                    d2.metric(
                        "Refinement passes run",
                        fit["n_iterations"],
                        help="Stops early once both moduli move by less than 0.1 %.",
                    )
                    d3.metric("Order", fit["order"])
                else:
                    d2.metric(
                        "Condition number",
                        f"{fit['condition_number']:.1f}"
                        if np.isfinite(fit["condition_number"])
                        else "n/a",
                        help="How separable ε³ and ε³ᐟ² are over this window. "
                        "Above ~30 the split between Eₘ and Eᵢ is unreliable.",
                    )
                    d3.metric(
                        "Eₘ/Eᵢ correlation",
                        f"{fit['corr_Em_Ei']:+.2f}" if np.isfinite(fit["corr_Em_Ei"]) else "n/a",
                        help="Near −1 means the two terms are trading off against each other.",
                    )

                mask = fit["mask"]
                st.plotly_chart(
                    residual_figure(epsilon[mask], (force_N - fitted)[mask], style),
                    **STRETCH,
                    key="residual_plot",
                )

                if fit.get("mode") == "sequential":
                    st.markdown("**Convergence of the two stages**")
                    st.caption(
                        "Pass 1 is the plain one-pass sequential fit. Each later pass "
                        "subtracts the other term's current estimate before refitting, "
                        "which removes the bias the first pass carries."
                    )
                    st.dataframe(
                        pd.DataFrame(fit["iterations"]).round(
                            {"Em_MPa": 4, "Ei_kPa": 4}
                        ),
                        hide_index=True,
                        **STRETCH,
                    )
                    stage_rows = []
                    for label, stage in (
                        ("stage 1", fit["stages"]["first"]),
                        ("stage 2", fit["stages"]["second"]),
                    ):
                        stage_rows.append(
                            {
                                "stage": label,
                                "term": ", ".join(stage["terms"]),
                                "ε window": f"{stage['epsilon_range'][0]:.3f} to {stage['epsilon_range'][1]:.3f}",
                                "points": stage["n_points"],
                                "R² (window)": round(float(stage["r_squared"]), 4),
                            }
                        )
                    st.dataframe(pd.DataFrame(stage_rows), hide_index=True, **STRETCH)
                    sens = None
                else:
                    st.markdown("**How much does the answer depend on the window?**")
                    sens = model.range_sensitivity(
                        fit_lo,
                        fit_hi,
                        terms=selected_terms(),
                        fit_offset=st.session_state["fit_offset"],
                        weighting=st.session_state["weighting"],
                    )

                if sens is not None:
                    s1, s2 = st.columns(2)
                    s1.metric(
                        "Eₘ spread",
                        f"{100 * sens['Em_relative_spread']:.1f} %"
                        if np.isfinite(sens["Em_relative_spread"])
                        else "n/a",
                    )
                    s2.metric(
                        "Eᵢ spread",
                        f"{100 * sens['Ei_relative_spread']:.1f} %"
                        if np.isfinite(sens["Ei_relative_spread"])
                        else "n/a",
                    )
                    st.caption(
                        "Range of each modulus across shrinking upper bounds, as a "
                        "fraction of its mean. Under ~10 % the fit is robust; much more "
                        "means the curve does not constrain the two terms separately."
                    )
                    figure = sensitivity_figure(sens["trials"], style)
                    if figure is not None:
                        st.plotly_chart(figure, key="sensitivity_plot", **STRETCH)
                    if sens["trials"]:
                        st.dataframe(
                            pd.DataFrame(sens["trials"]).round(
                                {"epsilon_max": 3, "Em_MPa": 4, "Ei_kPa": 4, "r_squared": 4}
                            ),
                            hide_index=True,
                            **STRETCH,
                        )

                st.markdown("**Geometry prefactors actually used**")
                st.code(
                    f"h0 = {fit['cell_height'] * 1e6:.3f} μm\n"
                    f"R0 = {fit['R0'] * 1e6:.3f} μm\n"
                    f"Am = {fit['Am']:.4e} N/Pa   (F_membrane = Am·Em·ε³)\n"
                    f"Ai = {fit['Ai']:.4e} N/Pa   (F_interior = Ai·Ei·ε³ᐟ²)",
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
            "force_offset_N": fit["force_offset"],
            "r_squared": fit["r_squared"],
            "adj_r_squared": fit["adj_r_squared"],
            "rmse_N": fit["rmse"],
            "epsilon_min": fit["epsilon_range"][0],
            "epsilon_max": fit["epsilon_range"][1],
            "n_points": fit["n_points"],
            "weighting": fit["weighting"],
            "fit_offset_enabled": fit["fit_offset"],
            "condition_number": fit["condition_number"],
            "membrane_fraction_at_max": fit["membrane_fraction_at_max"],
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
