"""
Stack & compare: a few data files (CSV, TSV, TXT or Excel) drawn one above
the other, each with its own x axis, or with the x axes linked, and each
with its own zero on x so that, say, the moment the probe touched or the
peak of a beat lines up across recordings.

Only data and figures here; the Streamlit panel is in app.py.
"""
from __future__ import annotations

import io
import math
import re
from typing import List, Optional

import numpy as np
import pandas as pd

ZERO_MODES = (
    "As in the file",
    "Start of the data",
    "A value I type",
    "At the highest point",
    "At the lowest point",
    "Where it first crosses a level",
)

DEFAULT_COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                   "#17becf")

MAX_PLOT_POINTS = 150_000


# ----------------------------------------------------------------- reading --

def _numeric_share(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    converted = frame.apply(pd.to_numeric, errors="coerce")
    return float(converted.notna().to_numpy().mean())


def read_table(data: bytes, name: str = "") -> pd.DataFrame:
    """
    The numeric columns of a data file, whatever its separator and however
    many lines of notes come before the header.

    Column names are kept when there is a header row; otherwise they are
    "Column 1", "Column 2", ... Raises ValueError when nothing numeric is
    found.
    """
    lower = name.lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        raw = pd.read_excel(io.BytesIO(data), header=None)
        return _tidy(raw)
    text = data.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    # Skip leading notes: start at the first line, from the top, after
    # which most lines split into the same number (>= 2) of numbers.
    sep = _guess_sep(lines)
    if sep != ",":
        # decimal commas (1,5 for 1.5), as European exports write them
        lines = [re.sub(r"(\d),(\d)", r"\1.\2", ln) for ln in lines]
    best = None
    for skip in range(0, min(60, max(1, len(lines)))):
        try:
            raw = pd.read_csv(io.StringIO("\n".join(lines[skip:])),
                              sep=sep, header=None, engine="python",
                              skip_blank_lines=True, on_bad_lines="skip",
                              comment=None)
        except Exception:
            continue
        if raw.shape[1] < 1 or raw.shape[0] < 2:
            continue
        share = _numeric_share(raw.iloc[1:min(len(raw), 200)])
        if share > 0.8:
            best = raw
            break
    if best is None:
        raise ValueError("No columns of numbers were found in this file.")
    return _tidy(best)


def _guess_sep(lines: List[str]):
    sample = [ln for ln in lines[:200] if ln.strip()][-50:]
    for sep in ("\t", ";", ","):
        counts = [ln.count(sep) for ln in sample]
        if counts and min(counts[len(counts) // 2:]) >= 1:
            return sep
    return r"\s+"


def _tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """A header row (if the first row is words) and numeric columns only."""
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    raw = raw.reset_index(drop=True)
    first = raw.iloc[0] if len(raw) else None
    names = None
    if first is not None:
        as_num = pd.to_numeric(first, errors="coerce")
        if as_num.isna().mean() > 0.5:
            names = [str(v).strip() if str(v).strip() not in ("", "nan")
                     else f"Column {k + 1}" for k, v in enumerate(first)]
            raw = raw.iloc[1:]
    if names is None:
        names = [f"Column {k + 1}" for k in range(raw.shape[1])]
    names = _unique(names)
    frame = raw.apply(pd.to_numeric, errors="coerce")
    frame.columns = names
    keep = [c for c in frame.columns if frame[c].notna().sum() >= 2]
    if not keep:
        raise ValueError("No columns of numbers were found in this file.")
    return frame[keep].reset_index(drop=True)


def _unique(names):
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n} ({seen[n]})")
        else:
            seen[n] = 0
            out.append(n)
    return out


TIME_WORDS = re.compile(r"(time|^t\b|^t[ _(]|sec|\(s\)|_s$|frame|x\b|"
                        r"distance|position|z\b|separation|ramp)", re.I)


def guess_x(columns: List[str]) -> str:
    """The column most likely to be the x axis (time, distance, ...)."""
    for c in columns:
        if re.search(r"time|sec|\(s\)|_s$", str(c), re.I):
            return c
    for c in columns:
        if TIME_WORDS.search(str(c)):
            return c
    return columns[0]


def guess_y(columns: List[str], x: str) -> str:
    rest = [c for c in columns if c != x]
    return rest[0] if rest else x


# ----------------------------------------------------------------- zeroing --

def zero_point(x, y, mode: str, typed: float = 0.0,
               level: float = 0.0) -> float:
    """
    Where on the file's own x axis the new zero goes.

    ``x`` is already multiplied by the file's x factor; ``typed`` is in
    those units too.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any() or mode == "As in the file":
        return 0.0
    xs, ys = x[ok], y[ok]
    if mode == "Start of the data":
        return float(np.min(xs))
    if mode == "A value I type":
        return float(typed)
    if mode == "At the highest point":
        return float(xs[int(np.argmax(ys))])
    if mode == "At the lowest point":
        return float(xs[int(np.argmin(ys))])
    if mode == "Where it first crosses a level":
        above = ys >= level
        flips = np.nonzero(above[1:] != above[:-1])[0]
        if flips.size == 0:
            return float(xs[0])
        k = int(flips[0])
        y0, y1 = ys[k], ys[k + 1]
        if y1 == y0:
            return float(xs[k])
        f = (level - y0) / (y1 - y0)
        return float(xs[k] + f * (xs[k + 1] - xs[k]))
    return 0.0


def prepare(frame: pd.DataFrame, xcol: str, ycol: str, x_factor=1.0,
            y_factor=1.0, zero=0.0, x_min=None, x_max=None):
    """(x - zero, y), x scaled, sorted as in the file, cropped to the
    range asked for (in the zeroed units)."""
    x = pd.to_numeric(frame[xcol], errors="coerce").to_numpy(float)
    y = pd.to_numeric(frame[ycol], errors="coerce").to_numpy(float)
    x = x * float(x_factor) - float(zero)
    y = y * float(y_factor)
    keep = np.isfinite(x) & np.isfinite(y)
    if x_min is not None:
        keep &= x >= x_min
    if x_max is not None:
        keep &= x <= x_max
    return x[keep], y[keep]


def thin_index(y, limit=MAX_PLOT_POINTS):
    """Which points to draw: at most ``limit``, keeping each block's min
    and max so spikes survive."""
    n = len(y)
    if n <= limit:
        return np.arange(n)
    blocks = max(1, limit // 2)
    edges = np.linspace(0, n, blocks + 1).astype(int)
    keep = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        lo, hi = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        keep.extend(sorted({lo, hi}))
    return np.asarray(keep, int)


def smooth(y, window: int):
    """Centred moving average over ``window`` points (1 = off)."""
    window = int(window)
    if window <= 1 or len(y) < 3:
        return y
    window = min(window, len(y))
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(y, (pad, window - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def aligned_table(traces) -> pd.DataFrame:
    """The plotted data side by side: one x and one y column per file."""
    columns = {}
    for t in traces:
        label = t["label"]
        columns[f"{label} | x (zeroed)"] = pd.Series(t["x"])
        columns[f"{label} | {t['ylabel'] or 'y'}"] = pd.Series(t["y"])
    return pd.DataFrame(columns)


# ------------------------------------------------------------------ figure --

def stacked_figure(traces, linked=True, overlay=False, panel_height=260,
                   line_width=1.8, axis_width=1.2, font_size=14,
                   x_title="x", zero_line=True, gap=0.04, x_range=None,
                   grid=False, ticks="outside", mirror=True,
                   clickable=False):
    """
    One panel per trace, stacked, or all on one plot (``overlay``).

    Each trace dict: x, y, label, ylabel, colour, neg_up, slot (its number,
    returned in customdata with the file's own x so a click can set its
    zero), raw_x (x on the file's own axis, same length as x).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = max(1, len(traces))
    rows = 1 if overlay else n
    if rows > 1:
        # room between panels for each one's own x labels when unlinked
        plot_px = panel_height * rows
        need = (24 if linked else 30 + 2.6 * font_size) / plot_px
        gap = min(0.3, max(gap, need))
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=bool(linked),
                        vertical_spacing=(gap if rows > 1 else 0.0))
    for k, t in enumerate(traces):
        row = 1 if overlay else k + 1
        x_all, y_all = np.asarray(t["x"], float), np.asarray(t["y"], float)
        raw_all = np.asarray(t.get("raw_x", x_all), float)
        if len(raw_all) != len(x_all):
            raw_all = x_all
        idx = thin_index(y_all)
        x, y, raw = x_all[idx], y_all[idx], raw_all[idx]
        kind = go.Scatter if len(x) <= 60_000 else go.Scattergl
        fig.add_trace(kind(
            x=x, y=y, mode="lines+markers" if clickable else "lines",
            name=t["label"],
            # invisible markers: only points can be clicked (selected),
            # not a bare line
            marker=dict(size=8, opacity=0, color=t["colour"]),
            selected=dict(marker=dict(opacity=0)),
            unselected=dict(marker=dict(opacity=0)),
            line=dict(color=t["colour"], width=line_width),
            customdata=np.column_stack([np.full(len(x), t.get("slot", k)),
                                        raw]),
            hovertemplate=(f"<b>{t['label']}</b><br>x = %{{x:.6g}}<br>"
                           "y = %{y:.6g}<extra></extra>")),
            row=row, col=1)
        yaxis = dict(title_text=(t["ylabel"] if not overlay
                                 else (traces[0]["ylabel"] if n == 1
                                       else "")),
                     title_font=dict(color=t["colour"] if not overlay
                                     else "black"))
        if t.get("neg_up"):
            yaxis["autorange"] = "reversed"
        fig.update_yaxes(row=row, col=1, **yaxis)
    axes = dict(showline=True, linewidth=axis_width, linecolor="black",
                mirror=mirror, ticks=ticks, tickwidth=axis_width,
                ticklen=max(4, int(3 * axis_width)), tickcolor="black",
                showgrid=bool(grid), gridcolor="#e6e6e6", zeroline=False,
                tickfont=dict(size=font_size, color="black"),
                title_font_size=font_size + 2, automargin=True,
                exponentformat="power")
    fig.update_xaxes(**axes)
    fig.update_yaxes(**axes)
    if overlay:
        if any(t.get("neg_up") for t in traces):
            fig.update_yaxes(autorange="reversed")
    # every panel gets its own x labels when the axes are free; when they
    # are linked, only the bottom one does (and they zoom together)
    for r in range(1, rows + 1):
        show = (not linked) or r == rows
        fig.update_xaxes(row=r, col=1, showticklabels=show or not linked,
                         title_text=(x_title if show else None))
        if x_range is not None:
            fig.update_xaxes(row=r, col=1, range=list(x_range))
    if zero_line:
        for r in range(1, rows + 1):
            fig.add_vline(x=0, line=dict(color="#888", width=1, dash="dot"),
                          row=r, col=1)
    fig.update_layout(
        height=int(panel_height * rows / max(0.4, 1 - gap * (rows - 1))
                   + 90) if rows > 1 else int(panel_height + 90),
        margin=dict(l=80, r=30, t=50, b=60),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", color="black",
                  size=font_size),
        showlegend=bool(overlay or n > 1),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        hovermode="closest", clickmode="event+select", dragmode="zoom")
    return fig


def nice_stem(names) -> str:
    stems = [re.sub(r"\.[^.]+$", "", str(n)) for n in names if n]
    stem = "_vs_".join(stems[:3]) or "stacked"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:120]


def finite_or_none(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None
