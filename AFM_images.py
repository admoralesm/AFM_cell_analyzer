"""
AFM image analysis: Igor Binary Wave (.ibw) images, their channels, and the
signal-versus-time trace a line scan makes.

This is the engine of the desktop "AFM IBW Viewer (2D + 3D + Time Profile,
dual-axis)" with the Tk window taken off: everything here takes arrays and
settings and gives back arrays, numbers and text, so the Streamlit "AFM
images" tab, a script or a batch job can drive it.

  * try_read_ibw reads a file with igor2 (channel names from the wave's
    dimension labels, the instrument note as text);
  * extract_2d / orient_image / image_stats / contrast_limits give one
    channel as an image;
  * build_continuous_time turns the scan lines of one or two channels
    (trace and retrace) into one signal against time, with the turnaround
    and the move between lines as gaps;
  * force_convert, fit_baseline / apply_baseline, smooth_xy and apply_ymath
    are the processing applied to that trace, in the desktop app's order:
    force conversion -> baseline -> multiply -> add -> formula;
  * write_excel_bytes writes sheets to an .xlsx that Excel will open even
    for traces longer than its row limit.

The module-level helpers below the settings are the desktop code unchanged.
"""

import ast
import io
import math
import os
import re
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


# ========================= SETTINGS =========================
APP_TITLE = "AFM IBW Viewer — 2D + 3D + Time Profile (Dual-axis)"
DPI_EXPORT = 600
MAX_PLOT_POINTS = 250_000

DEFAULT_UNITS = "nm"
DEFAULT_LINE_RATE_HZ = 10.0

DEFAULT_TFRAC = 1 / 3  # Segment A fraction
DEFAULT_RFRAC = 1 / 3  # Segment B fraction
DEFAULT_MFRAC = 1 / 3  # Move fraction

DEFAULT_INCLUDE_MOVE_GAP = True
DEFAULT_REVERSE_SEG_B_X = True

DEFAULT_INSERT_TURNAROUND_GAP = False
DEFAULT_TURNAROUND_GAP_S = 0.0

DEFAULT_CMAP = "viridis"
DEFAULT_CONTRAST_MODE = "percentile"
DEFAULT_PCTL_LO = 1.0
DEFAULT_PCTL_HI = 99.0

DEFAULT_3D_DOWNSAMPLE = 2
DEFAULT_TIME_TRANSPOSE = True  # transpose ON by default (user request)

# ---- Time axis units. One of: "s", "ms", "min", "hr"
#      Used only the very first time the app runs; after that the last unit
#      picked in the GUI is remembered in CONFIG_PATH and restored on start.
DEFAULT_TIME_UNIT = "s"

# Map: code -> (factor multiplied onto seconds, axis label, suffix for column names)
TIME_UNIT_MAP = {
    "s":   (1.0,        "Time (s)",   "Time_s"),
    "ms":  (1000.0,     "Time (ms)",  "Time_ms"),
    "min": (1.0 / 60.0, "Time (min)", "Time_min"),
    "hr":  (1.0 / 3600.0, "Time (hr)", "Time_hr"),
}
TIME_UNIT_CHOICES = ["s", "ms", "min", "hr"]

# ---- FIX: image orientation default (rotates AFM image 90°)
DEFAULT_IMAGE_ORIENTATION = "Rotate 90° CW"

# Time style (primary / left axis)
DEFAULT_TIME_MODE = "line"          # line | scatter | both
DEFAULT_TIME_LINESTYLE = "-"
DEFAULT_TIME_MARKER = ""
DEFAULT_TIME_LINEWIDTH = 1.8        # thinner, publication-like default
DEFAULT_TIME_ALPHA = 1.0

# Time plot behavior
DEFAULT_CONNECT_NAN_GAPS = False
DEFAULT_ZOOM_AUTOSCALE_Y = True

# Channel naming presets for exports
DEFAULT_USE_CHANNEL_NAME_PRESETS = True

# Baseline / smoothing / ticks defaults
DEFAULT_BASELINE_MODE = "None"      # None | Offset | Linear
DEFAULT_SMOOTH_OVERLAY = False
DEFAULT_SMOOTH_WINDOW = 101
# ---- Smoothing method for the time trace(s)
#      "Moving average"  : NaN-aware boxcar (original behavior)
#      "Savitzky-Golay"  : preserves peak shapes (recommended for AFM traces)
#      "Spline (visual)" : cubic-spline densified curve for very smooth rendering
DEFAULT_SMOOTH_METHOD = "Savitzky-Golay"
SMOOTH_METHODS = ["Moving average", "Savitzky-Golay", "Spline (visual)"]
DEFAULT_SMOOTH_POLYORDER = 3        # Savitzky-Golay polynomial order
DEFAULT_SMOOTH_REPLACE = False      # True: smoothed curve REPLACES the raw line
DEFAULT_X_TICK_STEP = ""
DEFAULT_Y_TICK_STEP = ""
DEFAULT_Y2_TICK_STEP = ""
DEFAULT_INT_TICKS = False

LINESTYLES = ["-", "--", "-.", ":", "None"]
MARKERS = ["", ".", "o", "x", "+", "s", "d", "^", "v", "*"]

TIME_STYLE_PRESETS = [
    "Custom",
    "Publication (thin, smooth)",
    "Clean line (continuous)",
    "Clean line (break at gaps)",
    "Line + dots (break at gaps)",
    "Scatter only",
]

# Fonts / spacing
TITLE_FS = 10
LABEL_FS = 10
TICK_FS = 9
CBAR_TICK_FS = 8
CBAR_LABEL_FS = 9

TIME_TITLE_FS = 14
TIME_LABEL_FS = 12

# ----- Publication defaults (time plot) -----
DEFAULT_PUB_TITLE = "Signal vs Time"
DEFAULT_PUB_XLABEL = ""        # empty -> use auto from time unit
DEFAULT_PUB_YLABEL = ""        # empty -> use auto from units (LEFT axis)
DEFAULT_PUB_Y2LABEL = ""       # empty -> use auto from units2 (RIGHT axis)
DEFAULT_PUB_TITLE_FS = TIME_TITLE_FS
DEFAULT_PUB_LABEL_FS = TIME_LABEL_FS
DEFAULT_PUB_TICK_FS = TICK_FS
DEFAULT_PUB_SHOW_LEGEND = False
DEFAULT_PUB_LEGEND_LABEL = ""
DEFAULT_PUB_LEGEND_LABEL_2 = ""
DEFAULT_PUB_LEGEND_LOC = "best"
DEFAULT_PUB_LEGEND_FS = 10
DEFAULT_PUB_SHOW_GRID = False      # Origin/Nature: no grid by default
DEFAULT_PUB_GRID_ALPHA = 0.20
DEFAULT_PUB_AXIS_LW = 1.2          # slightly heavier boxed frame (Origin-like)
DEFAULT_PUB_TICK_DIRECTION = "in"  # Origin/Nature: ticks point inward
DEFAULT_PUB_HIDE_TOP_RIGHT_SPINES = False
DEFAULT_PUB_SAVE_DPI = DPI_EXPORT
DEFAULT_PUB_SAVE_TRANSPARENT = False
DEFAULT_PUB_SAVE_TIGHT = True
DEFAULT_PUB_SAVE_WIDTH_IN = 7.0
DEFAULT_PUB_SAVE_HEIGHT_IN = 4.0
DEFAULT_PUB_SAVE_ONLY_TIME = True

# ----- Cantilever / force conversion -----
# F = k * deflection.  Deflection in volts is first converted to nm via the
# Defl InvOLS (inverse optical lever sensitivity, nm/V):
#   nm  = V  * InvOLS
#   nN  = nm * k          (nm × N/m = nN)
DEFAULT_SPRING_K = 0.1          # N/m
DEFAULT_DEFL_INVOLS = 50.0      # nm/V
DEFAULT_SETPOINT = 0.0          # raw deflection units (V or nm)
DEFAULT_SUBTRACT_SETPOINT = False
DEFAULT_FORCE_MODE = "None"
FORCE_MODES = [
    "None",
    "Defl (V) → nm",
    "Defl (V) → force (nN)",
    "Defl (nm) → force (nN)",
]
# Output unit for force conversions. Native result of nm × k(N/m) is nN;
# these factors rescale it: 1 nN = 1000 pN = 0.001 µN.
FORCE_UNITS = ["pN", "nN", "µN"]
DEFAULT_FORCE_UNIT = "nN"
FORCE_UNIT_SCALE = {"pN": 1000.0, "nN": 1.0, "µN": 1e-3}
# k, InvOLS and conversion choices are remembered between sessions here:
CONFIG_PATH = os.path.join(os.path.expanduser("~"),
                           ".afm_ibw_viewer_config.json")

# ----- Y axis: calculations on the trace + axis scale -----
# Order applied to each trace:  baseline  ->  x multiply  ->  + add  ->  formula
# These are NOT remembered between sessions, so a forgotten factor can never
# silently change the next data set.
Y_SCALES = ["linear", "log", "symlog"]
DEFAULT_Y_SCALE = "linear"
DEFAULT_Y_MULT = "1"
DEFAULT_Y_ADD = "0"
DEFAULT_Y_FORMULA = ""
DEFAULT_Y_UNIT_LABEL = ""       # blank = keep the normal unit label
DEFAULT_Y_SYMLOG_ZONE = ""      # blank = auto

# ----- Bold axis text (title, labels, tick numbers) -----
DEFAULT_PUB_BOLD = False

# ----- Style theme for the time plot -----
#   "None"          : plain matplotlib styling (previous behavior)
#   "Origin/Nature" : boxed axes, inward ticks on all four sides, minor ticks,
#                     no grid, Arial/Helvetica fonts — publication ready
DEFAULT_PUB_THEME = "Origin/Nature"
PUB_THEMES = ["None", "Origin/Nature"]

LEGEND_LOCS = [
    "best", "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center", "center"
]
TICK_DIRECTIONS = ["in", "out", "inout"]

# Default colors (used if user hasn't picked any)
DEFAULT_TIME_COLOR_1 = None         # matplotlib default (C0-ish)
# red-ish so the two traces are easily distinguishable
DEFAULT_TIME_COLOR_2 = "#d62728"
# ============================================================


def _ensure_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    return str(x)


def infer_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _safe_filename(s: str) -> str:
    s = _ensure_text(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


# =====================================================================
#                    SAFE EXCEL (.xlsx) EXPORT
# =====================================================================
# Why this exists:
#   A dual-channel time series is ny x (nx_A + nx_B + gaps) samples long,
#   which for a normal AFM scan is easily 1-3 MILLION rows.  Excel's hard
#   limit is 1,048,576 rows.  When that limit is crossed, pandas aborts in
#   the middle of writing, but the (half-built) workbook is still flushed
#   to the destination path when the ExcelWriter closes -- with openpyxl
#   the result does not even contain '[Content_Types].xml', so Excel
#   reports "the file is corrupt".  That is the bug this section fixes.
#
# The rules below make an export either fully correct or not written at
# all; the destination file is never left in a broken state.
EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_DATA_ROWS = EXCEL_MAX_ROWS - 1     # one row is used by the header
EXCEL_MAX_COLS = 16_384
EXCEL_MAX_CELL_CHARS = 32_767
EXCEL_MAX_SHEET_NAME = 31

# Control characters that cannot legally be stored in the XML that makes up
# an .xlsx file (openpyxl raises IllegalCharacterError, older xlsxwriter
# writes them straight through and Excel then refuses to open the file).
# IBW notes are raw instrument text and regularly contain them.
_ILLEGAL_XLSX_CHARS = re.compile(r"[\000-\010]|[\013-\014]|[\016-\037]")
_SURROGATES = re.compile(r"[\ud800-\udfff]")


def _clean_excel_text(s, limit=EXCEL_MAX_CELL_CHARS) -> str:
    """Make any string safe to store in a spreadsheet cell."""
    s = _ensure_text(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _ILLEGAL_XLSX_CHARS.sub("", s)
    s = _SURROGATES.sub("", s)
    if len(s) > limit:
        s = s[: limit - 3] + "..."
    return s


def _safe_sheet_name(name, used=None) -> str:
    """Excel sheet names: <= 31 chars, no []:*?/\\ , must be unique."""
    s = _clean_excel_text(name, EXCEL_MAX_SHEET_NAME).strip()
    s = re.sub(r"[\[\]:*?/\\]", "_", s)
    s = s.strip("'").strip()
    if not s:
        s = "Sheet"
    s = s[:EXCEL_MAX_SHEET_NAME]
    if used is None:
        return s
    base = s
    i = 2
    while s.lower() in used:
        suffix = f"_{i}"
        s = base[: EXCEL_MAX_SHEET_NAME - len(suffix)] + suffix
        i += 1
    used.add(s.lower())
    return s


def _clean_df_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df that Excel can definitely store.

    * column names sanitized and made unique
    * +/-inf turned into blanks (xlsxwriter cannot write them)
    * text columns stripped of illegal control characters and truncated
    """
    out = df.copy()

    cols, seen = [], {}
    for c in out.columns:
        name = _clean_excel_text(c, 255) or "col"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    out.columns = cols

    for c in out.columns:
        kind = getattr(out[c].dtype, "kind", "O")
        if kind == "f":
            v = np.asarray(out[c].to_numpy(dtype=float, copy=True))
            v[~np.isfinite(v)] = np.nan       # +/-inf -> empty cell
            out[c] = v
        elif kind in ("O", "S", "U"):
            out[c] = out[c].map(
                lambda v: v if v is None or isinstance(v, (int, float, np.number))
                else _clean_excel_text(v))

    if out.shape[1] > EXCEL_MAX_COLS:
        out = out.iloc[:, :EXCEL_MAX_COLS]
    return out


def _split_sheet_for_excel(name, df, max_rows=EXCEL_MAX_DATA_ROWS):
    """Split a too-tall DataFrame into several sheets Excel can hold."""
    n = len(df)
    if n <= max_rows:
        return [(name, df)]
    parts = []
    n_parts = int(math.ceil(n / float(max_rows)))
    for i in range(n_parts):
        chunk = df.iloc[i * max_rows: (i + 1) * max_rows]
        parts.append((f"{name}_{i + 1}", chunk))
    return parts


def pick_excel_engine():
    """Return the name of an available pandas Excel engine, or None."""
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except Exception:
        pass
    try:
        import openpyxl  # noqa: F401
        return "openpyxl"
    except Exception:
        return None


# ---------------------------------------------------------------------
# Blank-cell repair.
#
# Depending on the installed pandas/openpyxl versions, a NaN (every Move /
# turnaround gap in a time trace is a NaN) can be written as
#       <c r="B514" t="inlineStr"/>
# i.e. a cell declared as an inline string that contains no string.  That
# is well-formed XML but invalid per the spreadsheet schema, so Excel
# refuses the workbook with "We found a problem with some content" --
# which is exactly the 'corrupted file' symptom.  Dropping the bogus type
# attribute turns it into a normal empty cell and Excel is happy.
_XLSX_BAD_CELL_FINDER = re.compile(
    rb'<c\b[^>]*?\st="(?:inlineStr|s|str)"\s*(?:/>|>\s*</c>)')
_XLSX_BAD_CELL_FIXES = (
    (re.compile(rb'(<c\b[^>]*?)\s+t="(?:inlineStr|s|str)"\s*/>'), rb'\1/>'),
    (re.compile(
        rb'(<c\b[^>]*?)\s+t="(?:inlineStr|s|str)"\s*>\s*</c>'), rb'\1/>'),
)


def _is_sheet_part(name):
    return name.startswith("xl/worksheets/") and name.endswith(".xml")


def count_bad_xlsx_cells(path):
    """Number of malformed empty cells in a workbook (0 = clean)."""
    total = 0
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if _is_sheet_part(n):
                total += len(_XLSX_BAD_CELL_FINDER.findall(z.read(n)))
    return total


def repair_xlsx_blank_cells(path):
    """Rewrite 'path' in place with malformed empty cells fixed.

    Returns the number of cells repaired (0 if the file was already fine,
    in which case it is left untouched).
    """
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        needs_fix = any(_is_sheet_part(m.filename)
                        and _XLSX_BAD_CELL_FINDER.search(z.read(m.filename))
                        for m in members)
        if not needs_fix:
            return 0
        tmp = path + ".repair.tmp"
        fixed = 0
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
            for info in members:
                data = z.read(info.filename)
                if _is_sheet_part(info.filename):
                    for pat, rep in _XLSX_BAD_CELL_FIXES:
                        data, n = pat.subn(rep, data)
                        fixed += n
                out.writestr(info, data)
    os.replace(tmp, path)
    return fixed


def verify_xlsx(path):
    """Raise if 'path' is not a valid .xlsx workbook Excel will open.

    An .xlsx is a zip; every workbook Excel accepts contains
    [Content_Types].xml plus a parseable xl/workbook.xml, and no cell that
    declares a text type without holding any text.  Files produced by an
    aborted write, or by the NaN bug above, fail this immediately.
    """
    if not os.path.exists(path) or os.path.getsize(path) < 512:
        raise ValueError("the written file is empty or missing")
    with zipfile.ZipFile(path) as z:
        broken = z.testzip()
        if broken:
            raise ValueError(f"damaged data inside the workbook ({broken})")
        names = set(z.namelist())
        for required in ("[Content_Types].xml", "xl/workbook.xml"):
            if required not in names:
                raise ValueError(
                    f"workbook is incomplete (missing {required})")
        for part in ("xl/workbook.xml", "xl/sharedStrings.xml"):
            if part in names:
                ET.fromstring(z.read(part))
        for part in names:
            if _is_sheet_part(part):
                if _XLSX_BAD_CELL_FINDER.search(z.read(part)):
                    raise ValueError(
                        "the Excel library produced malformed blank cells "
                        "that Excel would reject")


def write_excel_safely(out_path, sheets, image_bytes=None,
                       image_sheet="Figure"):
    """Write [(sheet_name, DataFrame), ...] to out_path as a valid .xlsx.

    Guarantees:
      * data is cleaned (illegal characters, +/-inf) before writing
      * sheets longer than Excel's row limit are split automatically
      * everything is written to a temporary file next to the destination,
        verified, and only then moved into place, so a failed export can
        never leave a corrupt file behind
      * if the destination is locked (open in Excel / syncing in OneDrive)
        the workbook is saved next to it under a free name instead

    Returns (actual_path, notes).  Raises on unrecoverable failure.
    """
    engine = pick_excel_engine()
    if engine is None:
        raise RuntimeError(
            "No Excel engine installed.\n"
            "Install one with:   pip install xlsxwriter")

    notes = []
    prepared, used_names = [], set()
    for raw_name, df in sheets:
        if df is None:
            continue
        clean = _clean_df_for_excel(df)
        chunks = _split_sheet_for_excel(raw_name, clean)
        if len(chunks) > 1:
            notes.append(
                f"'{raw_name}' has {len(clean):,} rows, more than Excel's "
                f"limit of {EXCEL_MAX_DATA_ROWS:,}. It was split across "
                f"{len(chunks)} sheets. Export as .csv to keep it in one "
                f"piece.")
        for chunk_name, chunk in chunks:
            prepared.append((_safe_sheet_name(chunk_name, used_names), chunk))

    if not prepared:
        raise ValueError("There is nothing to export.")

    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(
        out_dir, f".~export_{os.getpid()}_{int(time.time())}.tmp.xlsx")

    try:
        with pd.ExcelWriter(tmp_path, engine=engine) as writer:
            for sheet_name, df in prepared:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
            if image_bytes is not None and engine == "xlsxwriter":
                try:
                    buf = io.BytesIO(image_bytes)
                    ws = writer.book.add_worksheet(
                        _safe_sheet_name(image_sheet, used_names))
                    ws.insert_image("A1", "figure.png", {"image_data": buf})
                except Exception as e:
                    notes.append(f"Figure image skipped ({e}).")
        # Some pandas/openpyxl combinations write NaN gaps as empty inline
        # strings, which Excel rejects as corrupt. Fix them before the file
        # is allowed anywhere near the destination path.
        repair_xlsx_blank_cells(tmp_path)
        verify_xlsx(tmp_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

    actual = out_path
    try:
        os.replace(tmp_path, out_path)
    except PermissionError:
        time.sleep(0.6)
        try:
            os.replace(tmp_path, out_path)
        except PermissionError:
            base, ext = os.path.splitext(out_path)
            i = 1
            while os.path.exists(f"{base} ({i}){ext}"):
                i += 1
            actual = f"{base} ({i}){ext}"
            os.replace(tmp_path, actual)
            notes.append(
                f"'{os.path.basename(out_path)}' was locked (it is probably "
                f"open in Excel or syncing), so the workbook was saved as "
                f"'{os.path.basename(actual)}' instead.")
    return actual, notes


def write_csv_fallback(base_path, sheets):
    """Write the same data as CSV files. Returns the list of paths written."""
    base = os.path.splitext(base_path)[0]
    written = []
    single = len([s for s in sheets if s[1] is not None]) == 1
    for name, df in sheets:
        if df is None:
            continue
        path = f"{base}.csv" if single else f"{base}_{_safe_filename(name)}.csv"
        df.to_csv(path, index=False)
        written.append(path)
    return written


def try_read_ibw(path):
    """Read an Igor Binary Wave (.ibw) using igor2."""
    try:
        from igor2 import binarywave
    except Exception as e:
        raise ImportError(
            "Missing dependency 'igor2'. Install:  pip install igor2") from e

    bw = binarywave.load(path)
    wave = bw.get("wave", None)
    if wave is None:
        raise ValueError("IBW missing 'wave' key.")
    data = wave.get("wData", None)
    if data is None:
        raise ValueError("IBW wave has no 'wData'.")

    arr = np.array(data)

    note = _ensure_text(wave.get("note", "") or "")
    name = _ensure_text(wave.get("bname", "") or "")

    meta = {
        "wave_name": name,
        "channel_labels": _ibw_dimension_labels(wave),
        "note": note,
        "shape": arr.shape,
        "ndim": arr.ndim,
        "dtype": str(arr.dtype),
        "path": path,
    }
    return arr, meta


def _ibw_dimension_labels(wave):
    """Per-dimension index labels stored in an IBW (Asylum puts the channel
    names such as 'HeightRetrace' here). Returns {axis: [name, ...]}; an
    empty dict when the file has none."""
    out = {}
    try:
        labels = wave.get("labels", None)
        if not labels:
            return out
        for axis, dim in enumerate(labels):
            names = [_ensure_text(x).strip("\x00").strip() for x in (dim or [])]
            # entry 0 is the label of the whole dimension, the rest name
            # each index along it
            if len(names) > 1 and any(names[1:]):
                out[axis] = names[1:]
    except Exception:
        return {}
    return out


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


def list_ibw_files(folder, recursive=False):
    """All .ibw files in a folder (natural sort, so scan2 comes before
    scan10)."""
    found = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs.sort(key=_natural_key)
            for f in files:
                if f.lower().endswith(".ibw"):
                    found.append(os.path.join(root, f))
    else:
        try:
            for f in os.listdir(folder):
                full = os.path.join(folder, f)
                if f.lower().endswith(".ibw") and os.path.isfile(full):
                    found.append(full)
        except Exception:
            return []
    return sorted(found, key=lambda p: _natural_key(
        os.path.relpath(p, folder)))


def parse_note_for_params(note_text):
    note_text = _ensure_text(note_text)
    if not note_text.strip():
        return {}
    txt = note_text.replace("\r", "\n")
    out = {}

    patterns = [
        (r"(scan\s*rate|line\s*rate|scanrate|linerate)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(hz|s-1)?", "scan_hz"),
        (r"(line\s*time)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*s", "line_time_s"),
        (r"(pixels\s*per\s*line|nx|samples\s*per\s*line|points\s*per\s*line|scanpoints|pointslines)\s*[:=]\s*(\d+)", "nx"),
        (r"(lines|ny|scanlines)\s*[:=]\s*(\d+)", "ny"),
        (r"(units|z\s*units)\s*[:=]\s*([a-zA-Zµu]+)", "units"),
    ]

    for pat, key in patterns:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if not m:
            continue
        if key == "scan_hz":
            try:
                out["scan_hz"] = float(m.group(2))
            except Exception:
                pass
        elif key == "line_time_s":
            try:
                lt = float(m.group(2))
                if lt > 0:
                    out["scan_hz"] = 1.0 / lt
            except Exception:
                pass
        elif key == "nx":
            try:
                out["nx"] = int(m.group(2))
            except Exception:
                pass
        elif key == "ny":
            try:
                out["ny"] = int(m.group(2))
            except Exception:
                pass
        elif key == "units":
            out["units"] = m.group(2)

    return out


def parse_note_key_values(note_text):
    """Parse the IBW note into an ordered {key: value} dict.
    Asylum Research notes are typically 'Key: value' lines separated by \\r."""
    note_text = _ensure_text(note_text)
    out = {}
    for line in note_text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        for sep in (":", "="):
            if sep in line:
                k, v = line.split(sep, 1)
                k = k.strip()
                v = v.strip()
                if k and k not in out:
                    out[k] = v
                break
    return out


# Candidate note keys -> friendly label, shown as the summary in "Image info"
NOTE_SUMMARY_FIELDS = [
    (("ImagingMode", "Imaging Mode", "Mode"), "Imaging mode"),
    (("ScanSize", "Scan Size"), "Scan size"),
    (("FastScanSize", "Fast Scan Size"), "Fast scan size"),
    (("SlowScanSize", "Slow Scan Size"), "Slow scan size"),
    (("ScanRate", "Scan Rate"), "Scan rate"),
    (("ScanPoints", "PointsLines", "Points"), "Points/line"),
    (("ScanLines", "Lines"), "Scan lines"),
    (("ScanAngle", "Scan Angle"), "Scan angle"),
    (("DeflectionSetpointVolts", "Setpoint", "SetpointVolts"), "Setpoint"),
    (("DeflectionSetpointNewtons",), "Setpoint (N)"),
    (("DeflectionSetpointMeters",), "Setpoint (m)"),
    (("AmplitudeSetpointVolts",), "Amplitude setpoint (V)"),
    (("DeflectionOffsetVolts", "DeflectionOffset"), "Deflection offset (V)"),
    (("FreeAirDeflection", "FreeAirDeflectionVolts"), "Free-air deflection"),
    (("TriggerPoint",), "Trigger point"),
    (("IntegralGain", "IGain"), "Integral gain"),
    (("ProportionalGain", "PGain"), "Proportional gain"),
    (("DriveAmplitude", "Drive Amplitude"), "Drive amplitude"),
    (("DriveFrequency", "Drive Frequency", "ResFreq1"), "Drive frequency"),
    (("SpringConstant", "Spring Constant", "SpringC"), "Spring constant k"),
    (("InvOLS", "Invols", "DeflInvOLS", "Defl InvOLS", "AmpInvOLS"), "InvOLS"),
    (("Date",), "Date"),
    (("Time",), "Time"),
]


def normalize_fracs(tf, rf, mf):
    fr = np.array([tf, rf, mf], dtype=float)
    fr[fr < 0] = 0.0
    s = fr.sum()
    if s <= 0:
        return DEFAULT_TFRAC, DEFAULT_RFRAC, DEFAULT_MFRAC
    fr = fr / s
    return float(fr[0]), float(fr[1]), float(fr[2])


def extract_2d(arr, channel_axis, channel_index):
    if arr.ndim == 2:
        return np.array(arr, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported ndim={arr.ndim}; need 2D or 3D.")

    ca = int(channel_axis)
    ci = int(channel_index)

    if ca == 0:
        if not (0 <= ci < arr.shape[0]):
            raise ValueError("Channel index out of range.")
        return np.array(arr[ci, :, :], dtype=float)
    if ca == 1:
        if not (0 <= ci < arr.shape[1]):
            raise ValueError("Channel index out of range.")
        return np.array(arr[:, ci, :], dtype=float)
    if ca == 2:
        if not (0 <= ci < arr.shape[2]):
            raise ValueError("Channel index out of range.")
        return np.array(arr[:, :, ci], dtype=float)

    raise ValueError("channel_axis must be 0/1/2.")


def _nan_percentile(img, p):
    a = img[np.isfinite(img)]
    if a.size == 0:
        return np.nan
    return float(np.percentile(a, p))


def downsample_for_plot(t, y, max_points=MAX_PLOT_POINTS):
    n = len(t)
    if n <= max_points:
        return t, y
    step = int(math.ceil(n / max_points))
    idx = np.arange(0, n, step, dtype=int)
    return t[idx], y[idx]


def build_continuous_time(
    segA2d,
    segB2d,
    f_line_hz,
    tfrac,
    rfrac,
    mfrac,
    useA=True,
    useB=True,
    include_move_gap=True,
    reverse_B_x=True,
    turnaround_on=False,
    turnaround_gap_s=0.0
):
    """
    Builds: for each line j:
      Segment A (Nx pts) -> optional turnaround gap (NaNs) -> Segment B (Nx pts) -> optional move gap (NaNs)
    """
    if not (useA or useB):
        raise ValueError(
            "Select at least one channel for Segment A or Segment B.")

    A = np.asarray(segA2d, dtype=float) if segA2d is not None else None
    B = np.asarray(segB2d, dtype=float) if segB2d is not None else None

    base = A if A is not None else B
    if base is None or base.ndim != 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    ny, nx = base.shape
    if A is not None and A.shape != (ny, nx):
        raise ValueError("Segment A shape mismatch.")
    if B is not None and B.shape != (ny, nx):
        raise ValueError("Segment B shape mismatch.")

    f = float(f_line_hz)
    if f <= 0:
        raise ValueError("Line rate must be >0 Hz.")
    T_line = 1.0 / f

    tfrac, rfrac, mfrac = normalize_fracs(tfrac, rfrac, mfrac)
    tA = tfrac * T_line
    tB = rfrac * T_line
    tM = mfrac * T_line

    dtA = max(tA / float(nx), 1e-9)
    dtB = max(tB / float(nx), 1e-9)
    dtG = min(dtA, dtB)

    B_use = B
    if B_use is not None and reverse_B_x:
        B_use = B_use[:, ::-1]

    turn_s = float(max(0.0, turnaround_gap_s)) if turnaround_on else 0.0
    n_turn = int(round(turn_s / dtG)) if turn_s > 0 else 0
    n_turn = max(0, n_turn)

    n_move = int(round(tM / dtG)) if include_move_gap else 0
    n_move = max(0, n_move)

    nan_row = np.full(nx, np.nan, dtype=float)

    t_list, y_list = [], []
    tcur = 0.0

    def add_seg(yseg, dt):
        nonlocal tcur
        yseg = np.asarray(yseg, dtype=float)
        n = len(yseg)
        if n <= 0:
            return
        tseg = tcur + np.arange(n, dtype=float) * dt
        tcur = tseg[-1] + dt
        t_list.append(tseg)
        y_list.append(yseg)

    def add_gap(npts):
        nonlocal tcur
        if npts <= 0:
            return
        tseg = tcur + np.arange(npts, dtype=float) * dtG
        tcur = tseg[-1] + dtG
        t_list.append(tseg)
        y_list.append(np.full(npts, np.nan, dtype=float))

    for j in range(ny):
        rowA = nan_row if not useA else (A[j, :] if A is not None else nan_row)
        rowB = nan_row if not useB else (
            B_use[j, :] if B_use is not None else nan_row)

        add_seg(rowA, dtA)
        add_gap(n_turn)
        add_seg(rowB, dtB)
        add_gap(n_move)

    t = np.concatenate(t_list) if t_list else np.array([], dtype=float)
    y = np.concatenate(y_list) if y_list else np.array([], dtype=float)
    return t, y


# =====================================================================
# Y MATH — safe evaluation of the formulas typed in the "Y axis" tab
# =====================================================================
class YMathError(ValueError):
    """A problem with a Y-math entry, worded for the user."""


# Names a formula may use as data
YMATH_VARIABLES = ("y", "t", "y1", "y2", "p1", "p2")
YMATH_MAX_CHARS = 300

_YMATH_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant,
    ast.Compare, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
    ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq,
)


def _ymath_moving_average(a, n):
    """NaN-aware centred moving average over n points (n is made odd).
    Gaps (NaN) are skipped inside a window and stay gaps in the result.
    Runs in the same time whatever the window size."""
    a = np.asarray(a, dtype=float)
    try:
        n = float(n)
        if not math.isfinite(n):
            raise ValueError
        n = int(round(n))
    except Exception:
        raise YMathError("smooth(y, n) needs a whole number of points, "
                         "e.g. smooth(y, 21).")
    if a.ndim != 1:
        raise YMathError("smooth() needs a trace, e.g. smooth(y, 21).")
    if n < 3 or a.size < 2:
        return a.copy()
    if n % 2 == 0:
        n += 1
    # a half-window longer than the trace already covers every point
    n = min(n, 2 * a.size + 1)
    out = pd.Series(a).rolling(n, center=True, min_periods=1).mean()
    out = np.asarray(out, dtype=float).copy()
    out[~np.isfinite(a)] = np.nan
    return out


def ymath_namespace(t=None):
    """Functions and constants available in a Y formula.
    't' is the time axis of the trace (needed by deriv / integ)."""

    def _arr(a):
        return np.asarray(a, dtype=float)

    def _stat(fn):
        def f(a):
            a = _arr(a)
            a = a[np.isfinite(a)]
            return float(fn(a)) if a.size else float("nan")
        return f

    _mean, _median, _std = _stat(np.mean), _stat(np.median), _stat(np.std)
    _lo, _hi, _sum = _stat(np.min), _stat(np.max), _stat(np.sum)

    def _min(a, *b):
        return np.minimum(a, b[0]) if b else _lo(a)

    def _max(a, *b):
        return np.maximum(a, b[0]) if b else _hi(a)

    def _round(a, digits=0):
        return np.round(a, int(round(float(digits))))

    def _log(*_a):
        raise YMathError("log() is ambiguous. Use log10(y) for base 10 "
                         "or ln(y) for the natural log.")

    def _fixed(name, fn, nmin, nmax=None, example=None):
        """Accept exactly the stated number of values. This gives a clear
        message for a slip such as a decimal comma, and it keeps numpy's
        optional extra arguments (which can overwrite an input) out of
        reach."""
        nmax = nmin if nmax is None else nmax
        want = str(nmin) if nmin == nmax else f"{nmin} or {nmax}"
        ex = example or f"{name}(y)"

        def call(*args):
            if not (nmin <= len(args) <= nmax):
                raise YMathError(
                    f"{name}() takes {want} value{'s' if want != '1' else ''}"
                    f", e.g. {ex}. Decimals use a point, not a comma.")
            return fn(*args)
        return call

    def _trace(a, name):
        a = _arr(a)
        if t is None or a.ndim != 1 or a.shape != np.shape(t):
            raise YMathError(f"{name}() needs a trace, e.g. {name}(y).")
        return a, np.asarray(t, dtype=float)

    def deriv(a):
        """dy/dt, using the time axis in the unit shown on the plot.
        Central differences inside each run of data and one-sided differences
        at the ends of a run, so the points next to a gap keep a value (the
        same formulas as numpy.gradient, applied run by run)."""
        a, tt = _trace(a, "deriv")
        n = a.size
        out = np.full(a.shape, np.nan)
        if n < 2:
            return out
        g = np.isfinite(a) & np.isfinite(tt)
        left = np.zeros(n, dtype=bool)
        right = np.zeros(n, dtype=bool)
        left[1:] = g[1:] & g[:-1]       # has a usable point just before
        right[:-1] = g[:-1] & g[1:]     # has a usable point just after
        i = np.flatnonzero(left & right)
        hd = tt[i + 1] - tt[i]
        hs = tt[i] - tt[i - 1]
        out[i] = (hs ** 2 * a[i + 1] + (hd ** 2 - hs ** 2) * a[i]
                  - hd ** 2 * a[i - 1]) / (hs * hd * (hd + hs))
        i = np.flatnonzero(right & ~left)
        out[i] = (a[i + 1] - a[i]) / (tt[i + 1] - tt[i])
        i = np.flatnonzero(left & ~right)
        out[i] = (a[i] - a[i - 1]) / (tt[i] - tt[i - 1])
        return out

    def integ(a):
        """Running integral of y over t (trapezoid rule, gaps are skipped)."""
        a, tt = _trace(a, "integ")
        if a.size < 2:
            return np.zeros(a.shape)
        seg = 0.5 * (a[1:] + a[:-1]) * np.diff(tt)
        seg = np.where(np.isfinite(seg), seg, 0.0)
        out = np.concatenate([[0.0], np.cumsum(seg)])
        out[~np.isfinite(a)] = np.nan
        return out

    def cumsum(a):
        a = _arr(a)
        out = np.cumsum(np.where(np.isfinite(a), a, 0.0))
        out = np.asarray(out, dtype=float)
        out[~np.isfinite(a)] = np.nan
        return out

    def norm(a):
        """Rescale to 0..1 (minimum -> 0, maximum -> 1)."""
        a = _arr(a)
        lo, hi = _lo(a), _hi(a)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi == lo:
            raise YMathError("norm() needs a trace that is not constant.")
        return (a - lo) / (hi - lo)

    ns = {"pi": math.pi, "e": math.e, "nan": float("nan"), "log": _log}
    one_value = {
        "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp,
        "ln": np.log, "log10": np.log10, "log2": np.log2,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "sign": np.sign, "floor": np.floor, "ceil": np.ceil,
        "mean": _mean, "median": _median, "std": _std, "sum": _sum,
        "deriv": deriv, "integ": integ, "cumsum": cumsum, "norm": norm,
    }
    for name, fn in one_value.items():
        ns[name] = _fixed(name, fn, 1)
    ns["round"] = _fixed("round", _round, 1, 2, "round(y, 2)")
    ns["min"] = _fixed("min", _min, 1, 2, "min(y) or min(y, 5)")
    ns["max"] = _fixed("max", _max, 1, 2, "max(y) or max(y, 0)")
    ns["minimum"] = _fixed("minimum", np.minimum, 2, None, "minimum(y, 5)")
    ns["maximum"] = _fixed("maximum", np.maximum, 2, None, "maximum(y, 0)")
    ns["clip"] = _fixed("clip", np.clip, 3, None, "clip(y, 0, 10)")
    ns["where"] = _fixed("where", np.where, 3, None, "where(y > 0, y, 0)")
    ns["smooth"] = _fixed("smooth", _ymath_moving_average, 2, None,
                          "smooth(y, 21)")
    return ns


YMATH_FUNCTION_NAMES = tuple(sorted(ymath_namespace().keys()))


class _IntToFloat(ast.NodeTransformer):
    """Whole numbers become floats so a typo such as 9**9**9 cannot hang the
    program building a gigantic integer."""

    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(float(node.value)), node)
        return node


def ymath_normalize(expr):
    """Tidy what the user typed: '=y^2' -> 'y**2', 'Y*1000' -> 'y*1000'."""
    s = _ensure_text(expr).strip()
    if s.startswith("="):
        s = s[1:].strip()
    for a, b in (("^", "**"), ("×", "*"), ("·", "*"), ("÷", "/"),
                 ("−", "-"), ("–", "-"), ("π", "pi")):
        s = s.replace(a, b)
    return s.lower()


def ymath_compile(expr, variables=YMATH_VARIABLES):
    """Check a formula and return (code, names_used). Only numbers, the
    listed variables, + - * / ** % comparisons and the listed functions are
    accepted, so a formula can never touch files or the rest of the program."""
    s = ymath_normalize(expr)
    if not s:
        raise YMathError("The formula is empty.")
    if len(s) > YMATH_MAX_CHARS:
        raise YMathError(
            f"The formula is too long (limit {YMATH_MAX_CHARS} characters).")
    try:
        tree = ast.parse(s, mode="eval")
    except Exception:
        raise YMathError(
            f"Cannot read \"{expr}\". Check the brackets and operators "
            "(use * to multiply, e.g. 2*y).")
    allowed = set(variables) | set(YMATH_FUNCTION_NAMES)
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, _YMATH_NODES):
            raise YMathError(
                f"\"{expr}\" uses something that is not allowed in a formula. "
                "Use numbers, y, t, y1, y2, + - * / ^ and functions such as "
                "abs, sqrt, log10.")
        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            raise YMathError(
                f"\"{expr}\": compare two things at a time. For a range "
                "use where((y > 1) * (y < 5), y, nan).")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise YMathError(
                    f"\"{expr}\": write functions as name(value), "
                    "e.g. sqrt(y).")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                    node.value, (int, float)):
                raise YMathError(
                    f"\"{expr}\": only numbers can be typed in a formula.")
        if isinstance(node, ast.Name):
            if node.id not in allowed:
                known = ", ".join(variables) if variables else "none"
                raise YMathError(
                    f"\"{node.id}\" is not a known name. Variables: {known}. "
                    "Functions: " + ", ".join(
                        n for n in YMATH_FUNCTION_NAMES
                        if n not in ("pi", "e", "nan", "log")) + ".")
            names.add(node.id)
    try:
        tree = ast.fix_missing_locations(_IntToFloat().visit(tree))
        return compile(tree, "<y formula>", "eval"), names
    except Exception:
        # e.g. a number with hundreds of digits
        raise YMathError(f"\"{expr}\" contains a number that is too large.")


def ymath_scalar(text, default):
    """Value of a 'Multiply by' / 'Add' box: a number or simple arithmetic
    such as 1/1000, 1e-3 or 2*pi. Blank returns the default."""
    s = _ensure_text(text).strip()
    if not s:
        return float(default)
    not_number = (f"\"{text}\" is not a number. Type a number or simple "
                  "arithmetic such as 1000, 0.001 or 1/1000.")
    try:
        v = float(s)
    except Exception:
        try:
            code, _names = ymath_compile(s, variables=())
            with np.errstate(all="ignore"):
                v = float(eval(code, {"__builtins__": {}}, ymath_namespace()))
        except Exception:
            raise YMathError(not_number)
    if not math.isfinite(v):
        raise YMathError(not_number)
    return v



# =====================================================================
#   The desktop app's processing steps, as plain functions (no window)
# =====================================================================

IMAGE_ORIENTATIONS = ["None", "Rotate 90° CW", "Rotate 90° CCW", "Transpose",
                      "Transpose + Flip UD", "Transpose + Flip LR"]


def load_ibw_bytes(data, name="upload.ibw"):
    """try_read_ibw for an uploaded file's bytes. Returns a dict with
    arr, meta, note_dict and guess (scan rate, points, lines, units)."""
    suffix = os.path.splitext(name)[1] or ".ibw"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        arr, meta = try_read_ibw(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    meta = dict(meta)
    meta["path"] = name
    note = meta.get("note", "")
    return {"arr": arr, "meta": meta,
            "note_dict": parse_note_key_values(note),
            "guess": parse_note_for_params(note)}


def guess_channel_axis(arr):
    """The axis that holds the channels of a 3-D image (desktop rule: the
    last one when it is small, else the shortest)."""
    arr = np.asarray(arr)
    if arr.ndim != 3:
        return 2
    if arr.shape[2] <= 64:
        return 2
    return int(np.argmin(arr.shape))


def n_channels(arr, axis):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return 1
    if arr.ndim != 3:
        return 0
    return int(arr.shape[int(axis)])


def channel_label(idx, use_presets=True):
    """Short name used in file names: height / deflection / lateral."""
    if idx is None:
        return "None"
    i = int(idx)
    if use_presets:
        if i == 0:
            return "height"
        if i == 1:
            return "deflection"
        if i == 2:
            return "lateral"
    return f"ch{i}"


def channel_display_name(meta, axis, idx, use_presets=True):
    """The channel's own name stored in the IBW (e.g. HeightRetrace) when
    there is one, else the preset name."""
    labels = (meta or {}).get("channel_labels") or {}
    names = labels.get(int(axis)) or []
    if 0 <= int(idx) < len(names) and names[int(idx)]:
        return names[int(idx)]
    return channel_label(idx, use_presets)


def channel_image(arr, axis, ch):
    """One channel as a float 2-D array (no orientation applied)."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return np.array(arr, dtype=float)
    return extract_2d(arr, int(axis), int(ch))


def orient_image(img, choice):
    """The display orientation of the desktop app."""
    if img is None:
        return img
    img = np.asarray(img, dtype=float)
    choice = (choice or "None").strip()
    if choice == "Rotate 90° CW":
        return np.rot90(img, k=-1)
    if choice == "Rotate 90° CCW":
        return np.rot90(img, k=1)
    if choice == "Transpose":
        return img.T
    if choice == "Transpose + Flip UD":
        return np.flipud(img.T)
    if choice == "Transpose + Flip LR":
        return np.fliplr(img.T)
    return img


def image_stats(img):
    """Basic statistics of the displayed image (finite pixels only)."""
    a = np.asarray(img, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    mean = float(a.mean())
    return {
        "min": float(a.min()), "max": float(a.max()), "mean": mean,
        "median": float(np.median(a)),
        "rms_rq": float(np.sqrt(np.mean((a - mean) ** 2))),
        "ra": float(np.mean(np.abs(a - mean))),
        "pv": float(a.max() - a.min()), "n_pixels": int(a.size),
    }


def contrast_limits(img, mode="percentile", lo=DEFAULT_PCTL_LO,
                    hi=DEFAULT_PCTL_HI):
    """Colour limits: percentile (default 1-99) or min/max."""
    if img is None or not np.isfinite(img).any():
        return None, None
    if (mode or "").lower() == "minmax":
        return float(np.nanmin(img)), float(np.nanmax(img))
    plo = max(0.0, min(float(lo), 100.0))
    phi = max(0.0, min(float(hi), 100.0))
    if phi <= plo:
        phi = min(100.0, plo + 1.0)
    return _nan_percentile(img, plo), _nan_percentile(img, phi)


def note_lookup(nd, candidates):
    """Case-insensitive lookup of the first matching key in a note dict."""
    low = {str(k).lower(): v for k, v in (nd or {}).items()}
    for c in candidates:
        v = low.get(str(c).lower())
        if v is not None:
            return v
    return None


def first_float(s):
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    try:
        return float(m.group(0)) if m else None
    except Exception:
        return None


def note_calibration(nd):
    """(k N/m, InvOLS nm/V, setpoint in raw units) from the IBW note; None
    where the note does not say. InvOLS stored in m/V is turned into nm/V."""
    k = first_float(note_lookup(
        nd, ("SpringConstant", "Spring Constant", "SpringC")))
    iv = first_float(note_lookup(
        nd, ("InvOLS", "Invols", "DeflInvOLS", "Defl InvOLS")))
    if iv is not None and 0 < iv < 1e-3:
        iv *= 1e9
    sp = first_float(note_lookup(
        nd, ("DeflectionSetpointVolts", "Setpoint", "SetpointVolts",
             "DeflectionSetpoint")))
    if k is not None and not k > 0:
        k = None
    if iv is not None and not iv > 0:
        iv = None
    return k, iv, sp


def note_summary(nd):
    """[(label, value)] of the key scan parameters found in the note."""
    out = []
    for cands, label in NOTE_SUMMARY_FIELDS:
        v = note_lookup(nd, cands)
        if v is not None and str(v) != "":
            out.append((label, str(v)))
    return out


def scan_size_um(nd):
    """(fast, slow) scan size in µm from the note, None where unknown."""
    def um(cands):
        v = first_float(note_lookup(nd, cands))
        if v is None or not v > 0:
            return None
        return v * 1e6 if v < 1e-2 else v       # metres -> µm
    full = um(("ScanSize", "Scan Size"))
    fast = um(("FastScanSize", "Fast Scan Size")) or full
    slow = um(("SlowScanSize", "Slow Scan Size")) or full
    return fast, slow


def force_convert(y, mode, k=DEFAULT_SPRING_K, invols=DEFAULT_DEFL_INVOLS,
                  setpoint=0.0, subtract_setpoint=False,
                  force_unit=DEFAULT_FORCE_UNIT):
    """Deflection -> nm or force. With the setpoint subtracted first the
    result is relative to the setpoint:
    V->nm: (y-sp)*InvOLS.  V->nN: (y-sp)*InvOLS*k.  nm->nN: (y-sp)*k."""
    if y is None:
        return y
    mode = (mode or "None").strip()
    if mode == "None":
        return y
    y = np.asarray(y, dtype=float)
    sp = float(setpoint) if subtract_setpoint else 0.0
    fscale = FORCE_UNIT_SCALE.get(force_unit, 1.0)
    if mode == "Defl (V) → nm":
        return (y - sp) * float(invols)
    if mode == "Defl (V) → force (nN)":
        return (y - sp) * float(invols) * float(k) * fscale
    if mode == "Defl (nm) → force (nN)":
        return (y - sp) * float(k) * fscale
    return y


def units_for(mode, base_units, force_unit=DEFAULT_FORCE_UNIT,
              typed_unit=""):
    """The y-axis unit of a trace after its force conversion. A unit typed
    for the Y math wins (the math can change the unit)."""
    if typed_unit:
        return typed_unit
    mode = (mode or "None").strip()
    if "force" in mode:
        return force_unit if force_unit in FORCE_UNITS else DEFAULT_FORCE_UNIT
    if mode == "Defl (V) → nm":
        return "nm"
    return base_units or DEFAULT_UNITS


def fit_baseline(x, y, xmin, xmax, mode):
    """(a, b) of the baseline fitted to the region xmin..xmax:
    Offset = the median there (a = 0), Linear = least squares."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x >= xmin) & (x <= xmax)
    if not np.any(m):
        raise ValueError("No points with data in the selected region.")
    xs, ys = x[m], y[m]
    if mode == "Offset":
        return 0.0, float(np.median(ys))
    A = np.vstack([xs, np.ones_like(xs)]).T
    sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return float(sol[0]), float(sol[1])


def apply_baseline(x, y, mode, a=0.0, b=0.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if mode == "Offset":
        return y - float(b)
    if mode == "Linear":
        return y - (float(a) * x + float(b))
    return y


def nan_moving_average(y, win):
    y = np.asarray(y, dtype=float)
    if win <= 1:
        return y.copy()
    win = int(win)
    if win % 2 == 0:
        win += 1
    if win < 3:
        return y.copy()
    m = np.isfinite(y).astype(float)
    y0 = np.where(np.isfinite(y), y, 0.0)
    k = np.ones(win, dtype=float)
    num = np.convolve(y0, k, mode="same")
    den = np.convolve(m, k, mode="same")
    out = np.full_like(y, np.nan, dtype=float)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def smooth_xy(x, y, method=DEFAULT_SMOOTH_METHOD, win=DEFAULT_SMOOTH_WINDOW,
              poly=DEFAULT_SMOOTH_POLYORDER):
    """Smooth (x, y), gap-aware: each run of finite points is smoothed on
    its own and the runs are rejoined with NaN separators, so the move and
    turnaround gaps stay gaps. "Spline (visual)" returns a denser curve."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return x, y
    win = max(3, int(win))
    if win % 2 == 0:
        win += 1
    poly = max(1, min(int(poly), 7))
    method = (method or DEFAULT_SMOOTH_METHOD).strip()
    if method.startswith("Moving"):
        return x, nan_moving_average(y, win)
    finite = np.isfinite(x) & np.isfinite(y)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return x, y
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    xs_out, ys_out = [], []
    for run in runs:
        xr, yr = x[run], y[run]
        n = xr.size
        if method.startswith("Sav"):
            if n >= 5:
                w = max(3, min(win, n if n % 2 == 1 else n - 1))
                p = min(poly, w - 1)
                try:
                    from scipy.signal import savgol_filter
                    yr_s = savgol_filter(yr, w, p)
                except Exception:
                    yr_s = nan_moving_average(yr, w)
                xs_out.append(xr)
                ys_out.append(yr_s)
            else:
                xs_out.append(xr)
                ys_out.append(yr)
        else:
            xr2, uidx = np.unique(xr, return_index=True)
            yr2 = yr[uidx]
            n2 = xr2.size
            if n2 >= 4:
                w = max(3, min(win, n2 if n2 % 2 == 1 else n2 - 1))
                yr_pre = nan_moving_average(yr2, w)
                try:
                    from scipy.interpolate import make_interp_spline
                    xd = np.linspace(xr2[0], xr2[-1],
                                     int(min(max(n2 * 5, 200), 20000)))
                    spl = make_interp_spline(xr2, yr_pre, k=3)
                    xs_out.append(xd)
                    ys_out.append(spl(xd))
                except Exception:
                    xs_out.append(xr2)
                    ys_out.append(yr_pre)
            else:
                xs_out.append(xr2)
                ys_out.append(yr2)
        xs_out.append(np.array([np.nan]))
        ys_out.append(np.array([np.nan]))
    return np.concatenate(xs_out), np.concatenate(ys_out)


def apply_ymath(x, y, mult=1.0, add=0.0, formula="", which=1, others=None):
    """baseline-corrected y -> y x multiply + add -> formula.

    ``others`` maps "y1"/"y2" to the other trace (after its own baseline,
    multiply and add) on the same time grid, for formulas that combine
    them. Returns (result, error or "", warning or ""). With the defaults
    the input comes back untouched; a formula that fails leaves the
    multiply/add result and says why."""
    y = np.asarray(y, dtype=float)
    out = y
    if mult != 1.0:
        out = out * mult
    if add != 0.0:
        out = out + add
    formula = (formula or "").strip()
    if not formula:
        return out, "", ""
    x = np.asarray(x, dtype=float)
    try:
        code, names = ymath_compile(formula)
        ns = ymath_namespace(x)
        ns["y"] = out
        ns["t"] = x
        for n, w in (("y1", 1), ("p1", 1), ("y2", 2), ("p2", 2)):
            if n in names:
                if w == which:
                    ns[n] = out
                else:
                    other = (others or {}).get(f"y{w}")
                    if other is None:
                        raise YMathError(
                            f"the formula uses y{w}, but Plot {w} is not on "
                            "the plot.")
                    other = np.asarray(other, dtype=float)
                    if other.shape != out.shape:
                        raise YMathError(
                            "Plot 1 and Plot 2 do not share the same time "
                            "points, so they cannot be combined in a formula.")
                    ns[n] = other
        with np.errstate(all="ignore"):
            res = eval(code, {"__builtins__": {}}, ns)
        res = np.asarray(res)
        if np.iscomplexobj(res):
            raise YMathError(
                "the result is a complex number (for example a negative "
                "number raised to a fractional power).")
        res = res.astype(float)
        if res.ndim == 0:
            res = np.full(out.shape, float(res))
        elif res.shape != out.shape:
            raise YMathError("the formula must give one value per point.")
        res = np.where(np.isfinite(res), res, np.nan)
        res[~np.isfinite(out)] = np.nan
        warn = ""
        if np.any(np.isfinite(out)) and not np.any(np.isfinite(res)):
            warn = ("the formula has no defined value at any point (for "
                    "example the log or square root of a negative number, "
                    "or a division by zero), so nothing is drawn.")
        return res, "", warn
    except YMathError as e:
        return out, str(e), ""
    except Exception as e:
        return out, (f"could not be calculated ({type(e).__name__}: {e})."), ""


def auto_symlog_zone(yvals):
    """Linear zone for symlog: two decades below the largest |value|."""
    a = np.abs(np.asarray(yvals, dtype=float))
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return 1.0
    return float(10.0 ** (math.floor(math.log10(float(a.max()))) - 2))


def symlog_transform(y, zone):
    """sign(y)·log10(1 + |y|/zone): what a symlog axis draws, so a plot
    library without symlog can show it on a linear axis."""
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.log10(1.0 + np.abs(y) / float(zone))


def symlog_ticks(lo, hi, zone):
    """Tick positions (transformed) and labels for a symlog axis spanning
    lo..hi in data units."""
    vals = [0.0]
    top = max(abs(lo), abs(hi), zone)
    k = math.floor(math.log10(zone))
    while 10.0 ** k <= top * 1.0001:
        v = 10.0 ** k
        if v >= zone * 0.999:
            if hi > 0:
                vals.append(v)
            if lo < 0:
                vals.append(-v)
        k += 1
    vals = sorted(set(vals))
    pos = list(symlog_transform(np.array(vals), zone))
    labels = [("0" if v == 0 else f"{v:g}") for v in vals]
    return pos, labels


def p1_p2_aligned(x1, y1, x2, y2):
    """(t, y1, y2) on one time grid (y2 interpolated onto x1 when they
    differ), with the samples missing in any of the three dropped."""
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    if x1.shape == x2.shape and np.allclose(x1, x2, rtol=1e-9, atol=1e-12,
                                            equal_nan=True):
        t, a, b = x1, y1, y2
    else:
        m2 = np.isfinite(x2) & np.isfinite(y2)
        if not np.any(m2):
            return None, None, None
        order = np.argsort(x2[m2])
        b = np.interp(x1, x2[m2][order], y2[m2][order],
                      left=np.nan, right=np.nan)
        t, a = x1, y1
    mask = np.isfinite(t) & np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return None, None, None
    return t[mask], a[mask], b[mask]


def default_save_base(name, p1=(None, None), p2=(None, None),
                      use_presets=True, labels=None):
    """<file>_<channels>_vs_time, as the desktop app names its exports.
    p1 / p2 are (segment A index or None, segment B index or None).
    ``labels`` are the channel names stored in the file; when given they
    are used instead of the height / deflection / lateral presets, which
    only fit files whose channels come in that order."""
    base = _safe_filename(infer_basename(name or "afm")) or "afm"

    def lab(i):
        if labels and 0 <= int(i) < len(labels) and labels[int(i)]:
            return _safe_filename(labels[int(i)])
        return _safe_filename(channel_label(i, use_presets))
    a, b = p1
    a2, b2 = p2
    part1 = "_".join(lab(i) for i in (a, b) if i is not None)
    part2 = "_".join(lab(i) for i in (a2, b2) if i is not None)
    if part1 and part2:
        return f"{base}_{part1}_vs_{part2}_vs_time"
    if part1 and a is not None and b is not None:
        return f"{base}_A{lab(a)}_B{lab(b)}_vs_time"
    if part1 or part2:
        return f"{base}_{part1 or part2}_vs_time"
    return f"{base}_vs_time"


def write_excel_bytes(sheets, image_bytes=None):
    """write_excel_safely, into memory. Returns (bytes, notes)."""
    folder = tempfile.mkdtemp(prefix="afm_xlsx_")
    path = os.path.join(folder, "export.xlsx")
    try:
        actual, notes = write_excel_safely(path, sheets,
                                           image_bytes=image_bytes)
        with open(actual, "rb") as handle:
            return handle.read(), notes
    finally:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except OSError:
                pass
        try:
            os.rmdir(folder)
        except OSError:
            pass


def image_info_text(item, display_channel, orientation, stats=None,
                    units=DEFAULT_UNITS, k=None, invols=None, setpoint=None,
                    ymath_lines=()):
    """The desktop "Image info" text: file, key scan parameters from the
    note, the calibration chain, image statistics and the whole note."""
    meta = item.get("meta", {}) or {}
    nd = item.get("note_dict", {}) or {}
    lines = [f"File: {os.path.basename(meta.get('path', ''))}"]
    if meta.get("wave_name"):
        lines.append(f"Wave: {meta.get('wave_name')}")
    lines.append(f"Shape: {meta.get('shape', '')} | ndim: "
                 f"{meta.get('ndim', '')} | dtype: {meta.get('dtype', '')}")
    lines.append(f"Displayed channel: {display_channel}  |  Orientation: "
                 f"{orientation}")
    lines += ["", "== Key scan parameters (from IBW note) =="]
    summary = note_summary(nd)
    lines += [f"{a}: {b}" for a, b in summary] or ["(none recognized)"]
    k_n, iv_n, sp_n = note_calibration(nd)
    k_use = k_n or k
    iv_use = iv_n or invols
    sp_use = sp_n if sp_n is not None else setpoint
    lines += ["", "== Derived calibration (note values; settings as "
                  "fallback) =="]
    if k_use:
        lines.append(f"Spring constant k: {k_use:.6g} N/m")
    if iv_use:
        lines.append(f"Defl InvOLS: {iv_use:.6g} nm/V")
    if sp_use is not None and iv_use:
        sp_nm = sp_use * iv_use
        line = f"Setpoint: {sp_use:.6g} V  =  {sp_nm:.6g} nm"
        if k_use:
            sp_nN = sp_nm * k_use
            line += f"  =  {sp_nN:.6g} nN  ({sp_nN / 1e3:.6g} µN)"
        lines.append(line)
    lines.append("Unit chain:  defl(V) × InvOLS(nm/V) = defl(nm)  ;  "
                 "defl(nm) × k(N/m) = force(nN)  ;  1 µN = 1000 nN")
    if stats:
        lines += ["", f"== Image statistics (displayed channel, {units}) =="]
        lines += [f"Min: {stats['min']:.6g}", f"Max: {stats['max']:.6g}",
                  f"Mean: {stats['mean']:.6g}",
                  f"Median: {stats['median']:.6g}",
                  f"Rq (RMS roughness): {stats['rms_rq']:.6g}",
                  f"Ra (mean abs dev): {stats['ra']:.6g}",
                  f"Peak-to-valley: {stats['pv']:.6g}",
                  f"Finite pixels: {stats['n_pixels']}"]
    if ymath_lines:
        lines += ["", "== Y math / axis scale (time plot) ==",
                  "Order: baseline -> multiply -> add -> formula"]
        lines += list(ymath_lines)
    if nd:
        lines += ["", f"== Full note ({len(nd)} entries) =="]
        lines += [f"{k_}: {v}" for k_, v in nd.items()]
    return "\n".join(lines)
