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
    "Departure from baseline (contact point)",
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

CONTACT_MODE = "Departure from baseline (contact point)"


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

FRAMES = ("Box (all four sides)", "Open (left and bottom only)",
          "No axis lines")

LEGEND_PLACES = ("Above", "Right, outside", "Inside, top right",
                 "Inside, top left", "Inside, bottom right",
                 "Inside, bottom left", "Below", "Hidden")

_INSIDE = dict(bgcolor="rgba(255,255,255,0.75)", bordercolor="#999",
               borderwidth=0)
LEGEND_PLACES_SPEC = {
    "Above": dict(orientation="h", x=0, xanchor="left", y=1.01,
                  yanchor="bottom"),
    "Right, outside": dict(orientation="v", x=1.02, xanchor="left", y=1,
                           yanchor="top"),
    "Inside, top right": dict(orientation="v", x=0.99, xanchor="right",
                              y=0.99, yanchor="top", **_INSIDE),
    "Inside, top left": dict(orientation="v", x=0.01, xanchor="left",
                             y=0.99, yanchor="top", **_INSIDE),
    "Inside, bottom right": dict(orientation="v", x=0.99, xanchor="right",
                                 y=0.01, yanchor="bottom", **_INSIDE),
    "Inside, bottom left": dict(orientation="v", x=0.01, xanchor="left",
                                y=0.01, yanchor="bottom", **_INSIDE),
    "Below": dict(orientation="h", x=0, xanchor="left", y=-0.2,
                  yanchor="top"),
    "Hidden": dict(),
}


def _first(items, key):
    for t in items:
        if t.get(key) is not None:
            return t.get(key)
    return None


def _axis_title(items):
    names = []
    for t in items:
        label = t.get("ylabel") or ""
        if label and label not in names:
            names.append(label)
    return " / ".join(names)


def stacked_figure(traces, linked=True, overlay=False, panel_height=260,
                   line_width=1.8, axis_width=1.2, font_size=14,
                   x_title="x", zero_line=True, gap=0.04, x_range=None,
                   grid=False, ticks="outside", mirror=True,
                   clickable=False, frame=None, legend="Above",
                   y_titles=None, align_titles=True, title_space=70,
                   title_size=None, bold_titles=False, legend_size=None,
                   legend_line=40, width=None, stats_box="Off",
                   stats_place="Inside, bottom right", baselines=True,
                   crosshair=False, crosshair_width=1.0, zero_width=1.0,
                   bold_ticks=False, legend_title="", legend_box=False,
                   legend_bold=False):
    """
    The files in panels, one above the other.

    Each trace dict: x, y, label, ylabel, colour, neg_up, slot (its number,
    returned in customdata with the file's own x so a click can set its
    zero), raw_x (x on the file's own axis), and optionally
      panel    which panel it goes in (files with the same number are
               overlaid; default: one panel each, in order),
      right    its own y axis on the right of the panel (for a second
               file overlaid in different units),
      x_range  (lo, hi) for its panel's x axis,
      y_range  (lo, hi) for its y axis.
    ``overlay`` puts every file in one panel.

    Style for the whole figure: ``frame`` is one of FRAMES (the axis
    outline), ``legend`` one of LEGEND_PLACES, ``y_titles`` maps a panel
    number to the title of its left axis (and ``(panel, "right")`` to its
    right one) instead of the files' own, ``align_titles`` puts every y
    title at the same distance from the axis (``title_space`` px), so the
    panels line up whatever their tick labels.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    traces = list(traces)
    for k, t in enumerate(traces):
        t["_panel"] = 1 if overlay else int(t.get("panel") or (k + 1))
    panels = sorted({t["_panel"] for t in traces}) or [1]
    row_of = {p: r + 1 for r, p in enumerate(panels)}
    rows = len(panels)
    # which traces sit on which axis of which row (the first file in a
    # panel always takes the left axis)
    left = {r: [] for r in range(1, rows + 1)}
    right = {r: [] for r in range(1, rows + 1)}
    for t in traces:
        r = row_of[t["_panel"]]
        (right if (t.get("right") and left[r]) else left)[r].append(t)
    if rows > 1:
        # room between panels for each one's own x labels when unlinked
        plot_px = panel_height * rows
        need = (24 if linked else 30 + 2.6 * font_size) / plot_px
        gap = min(0.3, max(gap, need))
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=bool(linked),
                        vertical_spacing=(gap if rows > 1 else 0.0),
                        specs=[[{"secondary_y": True}]] * rows)
    for k, t in enumerate(traces):
        r = row_of[t["_panel"]]
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
            row=r, col=1,
            secondary_y=any(t is u for u in right[r]))
    if frame is None:
        frame = FRAMES[0] if mirror else FRAMES[1]
    show_line = frame != FRAMES[2]
    mirror = frame == FRAMES[0]
    title_size = int(title_size or font_size + 2)
    y_titles = y_titles or {}

    def styled(text):
        return f"<b>{text}</b>" if (bold_titles and text) else text

    x_title = styled(x_title)
    side_titles = []
    axes = dict(showline=show_line, linewidth=axis_width, linecolor="black",
                ticks=ticks, tickwidth=axis_width,
                ticklen=max(4, int(3 * axis_width)), tickcolor="black",
                showgrid=bool(grid), gridcolor="#e6e6e6", zeroline=False,
                tickfont=dict(size=font_size, color="black",
                              weight="bold" if bold_ticks else "normal"),
                title_font_size=title_size, automargin=True,
                exponentformat="power")
    fig.update_xaxes(mirror=mirror, **axes)
    for r in range(1, rows + 1):
        for side, items in ((False, left[r]), (True, right[r])):
            if side and not items:
                fig.update_yaxes(row=r, col=1, secondary_y=True,
                                 visible=False)
                continue
            colour = items[0]["colour"] if len(items) == 1 else "black"
            panel = panels[r - 1]
            title = y_titles.get((panel, "right") if side else panel)
            title = styled(title if title else _axis_title(items))
            if align_titles:
                side_titles.append((r, side, title, colour))
            spec = dict(axes, title_text=("" if align_titles else title),
                        title_font=dict(color=colour, size=title_size),
                        mirror=(mirror if not right[r] else False),
                        showgrid=bool(grid) and not side)
            if right[r]:
                spec.update(linecolor=colour, tickcolor=colour,
                            tickfont=dict(size=font_size, color=colour,
                                          weight="bold" if bold_ticks
                                          else "normal"))
            y_range = _first(items, "y_range")
            neg = any(t.get("neg_up") for t in items)
            if y_range is not None:
                lo, hi = sorted(map(float, y_range))
                spec["range"] = [hi, lo] if neg else [lo, hi]
                spec["autorange"] = False
            elif neg:
                spec["autorange"] = "reversed"
            fig.update_yaxes(row=r, col=1, secondary_y=side, **spec)
    # every panel gets its own x labels when the axes are free; when they
    # are linked, only the bottom one does (and they zoom together)
    shared_range = x_range or (_first(traces, "x_range") if linked
                               else None)
    for r in range(1, rows + 1):
        show = (not linked) or r == rows
        fig.update_xaxes(row=r, col=1, showticklabels=show,
                         title_text=(x_title if show else None))
        rng = shared_range if linked else (
            x_range or _first(left[r] + right[r], "x_range"))
        if rng is not None:
            fig.update_xaxes(row=r, col=1, range=sorted(map(float, rng)),
                             autorange=False)
    if zero_line:
        for r in range(1, rows + 1):
            fig.add_vline(x=0, line=dict(color="#888",
                                         width=float(zero_width),
                                         dash="dot"),
                          row=r, col=1)
    has_right = any(right.values())
    height = (int(panel_height * rows / max(0.4, 1 - gap * (rows - 1)) + 90)
              if rows > 1 else int(panel_height + 90))

    # baseline helpers, drawn after every file's own trace so the trace
    # numbers (what a click reports) stay the files' order
    if baselines:
        for t in traces:
            line = t.get("baseline_line")
            if line is None:
                continue
            r = row_of[1 if overlay else int(t.get("panel") or 1)]
            bx, by = line
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode="lines", showlegend=False,
                hoverinfo="skip", name=f"{t['label']} baseline",
                line=dict(color=t["colour"], width=max(1.0, line_width * 0.7),
                          dash="dash")),
                row=r, col=1,
                secondary_y=any(t is u for u in right[r]))
            window = t.get("baseline_window")
            if window is not None:
                fig.add_vrect(x0=window[0], x1=window[1], row=r, col=1,
                              fillcolor=t["colour"], opacity=0.07,
                              line_width=0)
            # the threshold a peak has to clear, dotted either side
            for band in t.get("threshold_lines") or ():
                fig.add_trace(go.Scatter(
                    x=bx, y=band, mode="lines", showlegend=False,
                    hoverinfo="skip", line=dict(color=t["colour"], width=1,
                                                dash="dot")),
                    row=r, col=1,
                    secondary_y=any(t is u for u in right[r]))
    # the peaks counted in the statistics, marked on the trace
    for t in traces:
        marks = t.get("peak_marks")
        if not marks:
            continue
        r = row_of[1 if overlay else int(t.get("panel") or 1)]
        for (px, py, symbol, text) in marks:
            if not len(px):
                continue
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="markers", showlegend=False,
                name=f"{t['label']} {text}",
                marker=dict(symbol=symbol, size=10, color=t["colour"],
                            line=dict(color="black", width=1)),
                hovertemplate=f"{text}<br>x = %{{x:.6g}}<br>"
                              "y = %{y:.6g}<extra></extra>"),
                row=r, col=1, secondary_y=any(t is u for u in right[r]))

    # the summary balloons: one per panel (or one per file below the plot)
    extra_bottom = 0
    if stats_box != "Off":
        size = int(legend_size or max(9, font_size - 2))
        groups = {}
        for t in traces:
            lines = summary_lines(t, stats_box)
            if not lines:
                continue
            r = row_of[1 if overlay else int(t.get("panel") or 1)]
            head = (f"<span style='color:{t['colour']}'><b>"
                    f"{t['label']}</b></span>")
            groups.setdefault(r, []).append(
                head + "<br>" + "<br>".join(lines))
        style = dict(showarrow=False, align="left",
                     bgcolor="rgba(255,255,255,0.88)", bordercolor="#888",
                     borderwidth=1, borderpad=5,
                     font=dict(size=size, color="black"))
        if stats_place == "Below the plot":
            blocks = [b for r in sorted(groups) for b in groups[r]]
            n_lines = max([b.count("<br>") + 1 for b in blocks] or [0])
            extra_bottom = int(n_lines * (size + 5) + 30)
            for k, block in enumerate(blocks):
                fig.add_annotation(
                    text=block, xref="paper", yref="paper",
                    x=(k + 0.5) / len(blocks), y=0, xanchor="center",
                    yanchor="top", yshift=-(40 + 2 * font_size), **style)
        else:
            right_side = "right" in stats_place
            for r, blocks in groups.items():
                lo, _hi = fig.get_subplot(r, 1, secondary_y=False) \
                    .yaxis.domain
                fig.add_annotation(
                    text="<br><br>".join(blocks), xref="paper",
                    yref="paper", x=0.99 if right_side else 0.01,
                    xanchor="right" if right_side else "left",
                    y=lo + 0.01, yanchor="bottom", **style)
    # the y titles as annotations at one fixed distance from the plot, so
    # they line up from panel to panel (an axis title sits wherever its
    # own tick labels end)
    for r, side, title, colour in side_titles:
        if not title:
            continue
        # a right axis overlays the left one, whose domain is the panel's
        lo, hi = fig.get_subplot(r, 1, secondary_y=False).yaxis.domain
        fig.add_annotation(
            text=title, xref="paper", yref="paper", x=1 if side else 0,
            y=(lo + hi) / 2, xshift=(title_space if side else -title_space),
            textangle=90 if side else -90, showarrow=False,
            xanchor="center", yanchor="middle",
            font=dict(size=title_size, color=colour))
    left_m = (title_space + title_size + 6) if align_titles else 80
    right_m = ((title_space + title_size + 6) if align_titles else 80) \
        if has_right else 30
    place = LEGEND_PLACES_SPEC.get(legend, LEGEND_PLACES_SPEC["Above"])
    legend_spec = dict(place, font=dict(size=int(legend_size or font_size),
                                        weight="bold" if legend_bold
                                        else "normal"),
                       itemwidth=max(30, int(legend_line or 40)))
    if legend_title:
        legend_spec["title"] = dict(
            text=f"<b>{legend_title}</b>",
            font=dict(size=int(legend_size or font_size) + 1))
    if legend_box:
        legend_spec.update(bordercolor="black", borderwidth=1,
                           bgcolor="rgba(255,255,255,0.85)")
    if legend == "Right, outside":
        legend_spec["x"] = 1.0 + (right_m + 10) / 900.0 if has_right \
            else 1.02
        right_m += 160
    if legend == "Below":
        legend_spec["y"] = -(font_size * 4.5 + 10) / max(height - 110, 100)
    fig.update_layout(
        height=height + (40 if legend == "Below" else 0) + extra_bottom,
        width=(int(width) if width else None),
        margin=dict(l=left_m, r=right_m, t=50,
                    b=60 + (40 if legend == "Below" else 0) + extra_bottom),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", color="black",
                  size=font_size),
        showlegend=bool(len(traces) > 1) and legend != "Hidden",
        legend=legend_spec,
        hovermode="closest", clickmode="event+select", dragmode="zoom")
    if crosshair:
        # a crosshair that follows the cursor: lines across the whole
        # panel at its x and y, with the nearest point always found
        spikes = dict(showspikes=True, spikemode="across",
                      spikesnap="cursor", spikedash="solid",
                      spikecolor="#444", spikethickness=float(crosshair_width))
        fig.update_xaxes(**spikes)
        fig.update_yaxes(**spikes)
        fig.update_layout(hoverdistance=-1, spikedistance=-1)
    for t in traces:
        t.pop("_panel", None)
    return fig


STATS_BOXES = ("Off", "Analysis summary", "Descriptive statistics")
STATS_PLACES = ("Inside, bottom right", "Inside, bottom left",
                "Below the plot")


def describe(y):
    """Descriptive statistics of a trace, as (label, value) pairs."""
    v = np.asarray(y, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return []
    sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    return [("n", f"{v.size:,}"),
            ("Mean ± SD", f"{np.mean(v):.4g} ± {sd:.3g}"),
            ("Median", f"{np.median(v):.4g}"),
            ("Min / max", f"{np.min(v):.4g} / {np.max(v):.4g}"),
            ("Range", f"{np.ptp(v):.4g}")]


def summary_lines(trace, mode):
    """The lines of a file's summary balloon: the analysis summary sent
    with it (none for a file loaded by hand), or descriptive statistics
    of what is plotted."""
    if trace.get("peak_stats"):
        stats = trace["peak_stats"]
    elif mode == "Analysis summary":
        stats = trace.get("stats") or []
    else:
        factor = float(trace.get("stats_factor", 1.0) or 1.0)
        unit = trace.get("stats_unit", "")
        stats = describe(np.asarray(trace.get("y", []), float) * factor)
        if unit:
            stats = [(k, v if k == "n" else f"{v} {unit}")
                     for k, v in stats]
    return [f"{k}: {v}" for k, v in stats]


def fit_baseline(x, y, x0, x1, order=1):
    """A polynomial (order 0 or 1) through the points with x0 <= x <= x1.
    Returns the coefficients (highest power first) and the residual SD, or
    (None, None) when fewer than 3 points fall in the window."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lo, hi = sorted((float(x0), float(x1)))
    keep = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if keep.sum() < 3:
        return None, None
    coeffs = np.polyfit(x[keep], y[keep], int(order))
    resid = y[keep] - np.polyval(coeffs, x[keep])
    return coeffs, float(np.std(resid, ddof=1)) if keep.sum() > 2 else 0.0


def baseline_departure(x, y, x0, x1, k=3.0, order=1, direction="either",
                       run=5):
    """
    The contact point: where the signal first leaves its baseline.

    The baseline is fitted over the window [x0, x1] (before contact); the
    contact is the first point after the window where the residual
    exceeds k × the baseline's residual SD for ``run`` points in a row
    (upwards, downwards or either). Returns (x of contact or None,
    coefficients, sd).
    """
    coeffs, sd = fit_baseline(x, y, x0, x1, order)
    if coeffs is None:
        return None, None, None
    return departure_from(x, y, coeffs, sd, max(x0, x1), k, direction, run)


def departure_from(x, y, coeffs, sd, after_x, k=3.0, direction="either",
                   run=5):
    """baseline_departure for a baseline already known (fitted, or set by
    hand): the contact after ``after_x``. Returns (x or None, coeffs,
    sd)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sd = float(sd) if sd is not None and np.isfinite(sd) else 0.0
    resid = y - np.polyval(coeffs, x)
    limit = float(k) * max(sd, 1e-12)
    if direction == "up":
        out = resid > limit
    elif direction == "down":
        out = resid < -limit
    else:
        out = np.abs(resid) > limit
    after = x > float(after_x)
    out &= after & np.isfinite(resid)
    count = 0
    for i in np.flatnonzero(after):
        count = count + 1 if out[i] else 0
        if count >= int(run):
            j = i - int(run) + 1
            # walk back to where it was last within one SD of the
            # baseline: the k-SD test fires late on a slow rise
            sign = np.sign(resid[j]) or 1.0
            calm = smooth(np.nan_to_num(resid), 2 * int(run) + 1)
            while j > 0 and after[j - 1] and sign * calm[j - 1] > 0.5 * sd:
                j -= 1
            return float(x[j]), coeffs, sd
    return None, coeffs, sd


PEAK_DIRECTIONS = ("both", "up", "down")


def unit_of(name):
    """The unit in a column name: 'Time (s)' -> 's', else ''."""
    found = re.search(r"\(([^()]+)\)\s*$", str(name))
    return found.group(1).strip() if found else ""


def peak_analysis(x, y, base, min_height, direction="both", min_sep=0.0,
                  factor=1.0, unit="", x_unit=""):
    """
    Peaks above and dips below a baseline.

    ``base`` is the baseline at every x. A peak counts when it rises at
    least ``min_height`` above the baseline (a dip: falls that far below)
    and is at least ``min_sep`` (x units) from the next one counted.
    Amplitudes are measured from the baseline and reported × ``factor`` in
    ``unit`` (e.g. a deflection in µm × k → force in nN).

    Returns {"up": indices, "down": indices, "stats": [(label, value)]}.
    """
    from scipy.signal import find_peaks

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    base = np.asarray(base, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(base)
    out = {"up": np.array([], int), "down": np.array([], int), "stats": []}
    if ok.sum() < 3:
        return out
    idx = np.flatnonzero(ok)
    r = (y - base)[ok]
    xs = x[ok]
    dx = float(np.median(np.diff(xs))) if xs.size > 1 else 1.0
    distance = max(1, int(round(float(min_sep) / dx))) if (
        min_sep and dx > 0) else None
    height = max(float(min_height), 0.0)
    kw = dict(height=height, prominence=height or None)
    if distance:
        kw["distance"] = distance
    up = find_peaks(r, **kw)[0] if direction in ("both", "up") else []
    down = find_peaks(-r, **kw)[0] if direction in ("both", "down") else []
    out["up"], out["down"] = idx[np.asarray(up, int)], idx[np.asarray(down,
                                                                       int)]
    u = f" {unit}" if unit else ""
    f = float(factor)
    level = float(np.nanmedian(base[ok])) * f
    stats = [("Baseline", f"{level:.4g}{u}"),
             ("Threshold", f"± {height * f:.3g}{u}")]
    span = float(np.ptp(xs)) if xs.size > 1 else 0.0
    for name, picks, sign in (("up", up, 1.0), ("down", down, -1.0)):
        if direction not in ("both", name):
            continue
        picks = np.asarray(picks, int)
        amps = sign * r[picks] * f
        label = "Peaks up" if name == "up" else "Peaks down"
        if picks.size == 0:
            stats.append((label, "none"))
            continue
        stats.append((label, f"{picks.size}"))
        stats.append((f"  amplitude ({name})", _mean_sd(amps) + u))
        stats.append((f"  largest ({name})", f"{np.max(amps):.4g}{u}"))
    every = np.sort(np.concatenate([np.asarray(up, int),
                                    np.asarray(down, int)]))
    if every.size >= 2 and span > 0:
        gaps = np.diff(xs[every])
        per = f" /{x_unit}" if x_unit else " per x unit"
        stats.append(("Rate", f"{every.size / span:.4g}{per}"))
        stats.append(("Interval", _mean_sd(gaps)
                      + (f" {x_unit}" if x_unit else "")))
    out["stats"] = stats
    return out


def _mean_sd(v, digits=3):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return "–"
    if v.size == 1:
        return f"{v[0]:.{digits + 1}g}"
    mean, sd = float(np.mean(v)), float(np.std(v, ddof=1))
    scale = abs(mean) if mean else max(sd, 1e-12)
    decimals = max(0, digits - 1 - int(math.floor(math.log10(scale))))
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


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
