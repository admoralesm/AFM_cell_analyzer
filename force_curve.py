"""
Single force curve: one AFM force recording against time.

Reads an Asylum .ibw force file (its channels, the time step from the
wave's x scaling or the note's NumPtsPerSec, the spring constant and the
segment boundaries from the note) or a plain table (CSV, TXT, Excel), turns
deflection into force, and finds where the curve ends: a baseline is fitted
to the quiet stretch after the curve, and the end is the last moment the
force is away from that baseline. The start can be found the same way from
a baseline before the curve.

    python tests_force_curve.py
"""
import os
import re
import tempfile

import numpy as np
import pandas as pd

import stack_compare as sc

TABLE_TYPES = ("csv", "txt", "tsv", "dat", "xlsx", "xls")
FILE_TYPES = ("ibw",) + TABLE_TYPES

FORCE = "Force (nN)"
FORCE_LESS_BASE = "Force − baseline (nN)"
TIME = "Time (s)"

# how a deflection channel can be stored, and its factor to nanometres
DEFLECTION_UNITS = {"m": 1e9, "µm": 1e3, "nm": 1.0, "V": None}


# --------------------------------------------------------------- reading --

def _text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("latin-1", "replace").replace("\x00", "")
    if isinstance(value, np.ndarray):
        return "".join(_text(v) for v in value.ravel())
    return str(value)


def note_dict(note):
    """The 'Key: value' lines of an Asylum wave note."""
    out = {}
    for line in re.split(r"[\r\n]+", _text(note)):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key and key not in out:
            out[key] = value.strip()
    return out


def _note_float(nd, *names):
    low = {k.lower(): v for k, v in (nd or {}).items()}
    for name in names:
        value = low.get(name.lower())
        if value is None:
            continue
        found = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
        if found:
            try:
                return float(found.group(0))
            except ValueError:
                pass
    return None


def _note_ints(nd, name):
    low = {k.lower(): v for k, v in (nd or {}).items()}
    value = low.get(name.lower())
    if not value:
        return []
    return [int(float(v)) for v in re.findall(r"[-+]?\d+(?:\.\d+)?",
                                               str(value))]


def _channel_names(wave, axis, count):
    labels = wave.get("labels") or []
    names = []
    try:
        dim = labels[axis] if axis < len(labels) else []
        names = [_text(v).strip() for v in (dim or [])][1:]
    except Exception:
        names = []
    names = (names + [""] * count)[:count]
    return [n or f"Channel {k + 1}" for k, n in enumerate(names)]


def deflection_unit(values):
    """How a deflection channel is most likely stored: metres when it is
    tiny, nanometres when it is in the tens to thousands."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return "m"
    span = float(np.nanpercentile(np.abs(v), 99))
    if span < 1e-3:
        return "m"
    if span < 50:
        return "V"
    return "nm"


def read_ibw(data, name="force.ibw"):
    """
    An Igor .ibw force file, as
    {"kind": "ibw", "channels": {name: values}, "dt", "t0", "time_from",
     "k" (N/m), "invols" (nm/V), "segments" (sample indices), "note"}.
    ``dt`` is None when neither the wave nor its note gives the time step.
    """
    from igor2 import binarywave

    fd, path = tempfile.mkstemp(suffix=".ibw")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        loaded = binarywave.load(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    wave = loaded["wave"]
    header = wave.get("wave_header", {})
    arr = np.asarray(wave.get("wData"), dtype=float)
    if arr.ndim == 0 or arr.size == 0:
        raise ValueError("the file holds no numbers")
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim == 1:
        arr = arr[:, None]
    nd = note_dict(wave.get("note"))
    # points along the first axis, one column per channel (as Asylum
    # writes them); a wave stored the other way round is turned
    axis = 1
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 16:
        arr = arr.T
        axis = 0
    names = _channel_names(wave, axis, arr.shape[1])
    if arr.shape[1] == 1:
        bname = _text(header.get("bname")).strip()
        names = [names[0] if not names[0].startswith("Channel")
                 else (bname or "Signal")]
    # the time step: the wave's own x scaling when it is in seconds, else
    # the sampling rate in the note
    if "sfA" in header:
        dx, x0 = float(np.ravel(header["sfA"])[0]), float(
            np.ravel(header["sfB"])[0])
        units = _text(np.asarray(header.get("dimUnits"))[0]
                      if header.get("dimUnits") is not None else "")
        long_units = wave.get("dimension_units")
        if not units and long_units is not None:
            units = _text(np.ravel(long_units)[0] if np.size(long_units)
                          else "")
    else:
        dx, x0 = float(header.get("hsA", 1.0)), float(header.get("hsB", 0.0))
        units = _text(header.get("xUnits"))
    units = units.strip().lower()
    rate = _note_float(nd, "NumPtsPerSec", "SampleRate", "SamplingRate")
    dt, time_from = None, None
    if units in ("s", "sec", "seconds") and dx > 0:
        dt, time_from = dx, "the wave's own time scaling"
    elif rate and rate > 0:
        dt, x0, time_from = 1.0 / rate, 0.0, "NumPtsPerSec in the note"
    elif 0 < dx < 1:
        dt, time_from = dx, "the wave's x scaling (no unit stored)"
    k = _note_float(nd, "SpringConstant", "Spring Constant", "SpringC")
    invols = _note_float(nd, "InvOLS", "DeflInvOLS", "Defl InvOLS")
    if invols is not None and 0 < invols < 1e-3:
        invols *= 1e9            # stored in m/V
    segments = [i for i in _note_ints(nd, "Indexes")
                if 0 < i < arr.shape[0] - 1]
    return {"kind": "ibw", "name": name, "n": int(arr.shape[0]),
            "channels": {n: arr[:, j] for j, n in enumerate(names)},
            "dt": dt, "t0": float(x0) if dt else 0.0,
            "time_from": time_from,
            "k": k if k and k > 0 else None,
            "invols": invols if invols and invols > 0 else None,
            "segments": segments, "note": nd}


def read_force_file(data, name):
    """An .ibw force file (see read_ibw) or a table of numbers, as
    {"kind": "table", "frame": DataFrame, "name"}."""
    if str(name).lower().endswith(".ibw"):
        return read_ibw(data, name)
    frame = sc.read_table(data, name)
    return {"kind": "table", "name": name, "frame": frame,
            "n": int(len(frame))}


def is_deflection(name):
    return bool(re.search(r"defl", str(name), re.I))


def is_force(name):
    return bool(re.search(r"force", str(name), re.I))


def guess_time(columns):
    """The time column of a table, or None when it has none."""
    for c in columns:
        if re.search(r"time|\(s\)|\bsec|\bt\b|_s$", str(c), re.I):
            return c
    return None


def guess_force(columns, time_col=None):
    rest = [c for c in columns if c != time_col]
    for c in rest:
        if is_force(c):
            return c
    for c in rest:
        if is_deflection(c):
            return c
    return rest[0] if rest else None


def to_nanonewton(values, unit, k=None, invols=None):
    """
    Force in nN from a channel: ``unit`` "N" / "nN" / "pN" for a force,
    or a deflection unit ("m", "µm", "nm", "V") times the spring constant
    ``k`` (N/m; nm × N/m = nN), with ``invols`` (nm/V) for volts.
    """
    v = np.asarray(values, dtype=float)
    if unit == "N":
        return v * 1e9
    if unit == "pN":
        return v * 1e-3
    if unit == "nN":
        return v
    if k is None or not k > 0:
        raise ValueError("a spring constant is needed to turn deflection "
                         "into force")
    if unit == "V":
        if invols is None or not invols > 0:
            raise ValueError("InvOLS is needed for a deflection in volts")
        return v * float(invols) * float(k)
    return v * DEFLECTION_UNITS[unit] * float(k)


# ---------------------------------------------------------------- the curve --

def baseline_after(t, f, window, order=0, k=3.0, run=5,
                   direction="either"):
    """
    The baseline after the curve and where the curve ends.

    ``window`` (t0, t1) is the quiet stretch after the curve. The baseline
    (flat, or sloped with ``order`` 1) is fitted there; walking back in
    time from the window, the end is where the force is first more than
    ``k`` × the baseline's SD away from it for ``run`` points in a row,
    and then back to within half an SD. Returns
    {"end": t or None, "coeffs", "sd", "window", "level"} where "level"
    is the baseline at the end (or at the window's start).
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    lo, hi = sorted(float(v) for v in window)
    coeffs, sd = sc.fit_baseline(t, f, lo, hi, order)
    if coeffs is None:
        return {"end": None, "coeffs": None, "sd": None,
                "window": (lo, hi), "level": None}
    # the same search as a contact point, run backwards in time
    rev_t, rev_f = -t[::-1], f[::-1]
    rev_coeffs = np.array(coeffs, float).copy()
    if len(rev_coeffs) == 2:          # y = a t + b  ->  y = -a (-t) + b
        rev_coeffs[0] = -rev_coeffs[0]
    found, _, _ = sc.departure_from(rev_t, rev_f, rev_coeffs, sd, -lo, k,
                                    direction, run)
    end = None if found is None else float(-found)
    at = end if end is not None else lo
    return {"end": end, "coeffs": coeffs, "sd": sd, "window": (lo, hi),
            "level": float(np.polyval(coeffs, at))}


def baseline_before(t, f, window, order=0, k=3.0, run=5,
                    direction="either"):
    """The baseline before the curve and where the curve starts: the same
    as baseline_after, forwards in time ("start" in place of "end")."""
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    lo, hi = sorted(float(v) for v in window)
    start, coeffs, sd = sc.baseline_departure(t, f, lo, hi, k, order,
                                              direction, run)
    at = start if start is not None else hi
    return {"start": start, "coeffs": coeffs, "sd": sd, "window": (lo, hi),
            "level": (float(np.polyval(coeffs, at))
                      if coeffs is not None else None)}


def default_window(t, after=True, fraction=0.15):
    """The last (or first) ``fraction`` of the recording."""
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    if not t.size:
        return (0.0, 1.0)
    a, b = float(t.min()), float(t.max())
    width = max(fraction, 0.01) * (b - a)
    return (b - width, b) if after else (a, a + width)


def curve_stats(t, f, end_info, start_info=None, unit="nN"):
    """
    What the curve did, as [(label, value text)] for the summary balloon
    and {name: number}: the end, the baseline after it, the peak (from
    that baseline), and with a start also the duration, time to peak and
    the area under the curve above the baseline (nN·s).
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    lines, numbers = [], {}
    coeffs = end_info.get("coeffs") if end_info else None
    base = (np.polyval(coeffs, t) if coeffs is not None
            else np.zeros_like(f))
    end = end_info.get("end") if end_info else None
    start = (start_info or {}).get("start")
    if end is not None:
        lines.append(("Curve ends at", f"{end:.4g} s"))
        numbers["end_s"] = end
    if coeffs is not None:
        lines.append(("Baseline after the end",
                      f"{end_info['level']:.4g} ± {end_info['sd']:.2g} "
                      f"{unit}"))
        numbers["baseline"] = end_info["level"]
        numbers["baseline_sd"] = end_info["sd"]
    lo = start if start is not None else float(np.nanmin(t))
    hi = end if end is not None else float(np.nanmax(t))
    inside = (t >= lo) & (t <= hi) & np.isfinite(f)
    if inside.any():
        rel = f - base
        i = np.flatnonzero(inside)[int(np.nanargmax(np.abs(rel[inside])))]
        lines.append(("Peak force", f"{rel[i]:.4g} {unit} at {t[i]:.4g} s"))
        numbers["peak"] = float(rel[i])
        numbers["peak_t"] = float(t[i])
        if end is not None:
            lines.append(("Peak to end", f"{end - t[i]:.4g} s"))
            numbers["peak_to_end_s"] = end - float(t[i])
        if start is not None:
            lines.append(("Curve starts at", f"{start:.4g} s"))
            numbers["start_s"] = start
            lines.append(("Start to peak", f"{t[i] - start:.4g} s"))
            numbers["start_to_peak_s"] = float(t[i]) - start
            if end is not None:
                lines.append(("Duration", f"{end - start:.4g} s"))
                numbers["duration_s"] = end - start
                area = float(np.trapezoid(rel[inside], t[inside])
                             if hasattr(np, "trapezoid")
                             else np.trapz(rel[inside], t[inside]))
                lines.append(("Area above the baseline",
                              f"{area:.4g} {unit}·s"))
                numbers["area"] = area
    return lines, numbers


def incline(t, f, start_window, end_window):
    """
    The straight line from the baseline at the start to the baseline at
    the end: through the mean force in the window before the curve (at the
    window's middle) and the mean in the window after it. Subtracted, the
    curve starts and ends at zero, whatever the drift between them.
    Returns {"coeffs" (slope, intercept), "start": (t, level), "end":
    (t, level)} or None when a window holds too few points.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    ends = []
    for window in (start_window, end_window):
        lo, hi = sorted(float(v) for v in window)
        inside = (t >= lo) & (t <= hi) & np.isfinite(f)
        if inside.sum() < 3:
            return None
        ends.append((float(np.mean(t[inside])), float(np.mean(f[inside]))))
    (ta, ya), (tb, yb) = ends
    if abs(tb - ta) < 1e-12:
        coeffs = np.array([0.0, 0.5 * (ya + yb)])
    else:
        slope = (yb - ya) / (tb - ta)
        coeffs = np.array([slope, ya - slope * ta])
    return {"coeffs": coeffs, "start": ends[0], "end": ends[1]}


def corrected(t, f, end_info, start_info=None, line=None):
    """
    The force less a baseline, with both baselines put in those terms.
    The baseline is ``line`` (polynomial coefficients, e.g. the incline
    from start to end) or else the one fitted after the end, which then
    becomes 0. Returns (force, end info, start info); unchanged when there
    is no baseline.
    """
    coeffs = line if line is not None else (end_info or {}).get("coeffs")
    if coeffs is None:
        return np.asarray(f, float), end_info, start_info
    t = np.asarray(t, float)
    coeffs = np.asarray(coeffs, float)
    out = np.asarray(f, float) - np.polyval(coeffs, t)

    def less(info, key, fallback):
        if not info or info.get("coeffs") is None:
            return info
        diff = np.polysub(np.asarray(info["coeffs"], float), coeffs)
        at = info[key] if info.get(key) is not None else fallback
        return dict(info, coeffs=diff, level=float(np.polyval(diff, at)))

    end_c = (dict(end_info, coeffs=np.array([0.0]), level=0.0)
             if line is None and end_info else
             less(end_info, "end", (end_info or {}).get("window", (0, 0))[0]))
    start_c = less(start_info, "start",
                   (start_info or {}).get("window", (0, 0))[1])
    return out, end_c, start_c


# ------------------------------------------- a baseline in two lines --

def side_fit(t, f, window, sloped=True):
    """
    A straight line fitted (least squares) to the force in one window, or
    its mean level when not ``sloped``: {"slope", "intercept", "sd", "n"},
    or None when the window holds fewer than 3 points.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    lo, hi = sorted(float(v) for v in window)
    inside = (t >= lo) & (t <= hi) & np.isfinite(f) & np.isfinite(t)
    if inside.sum() < 3:
        return None
    tt, ff = t[inside], f[inside]
    if sloped and np.ptp(tt) > 0:
        slope, intercept = np.polyfit(tt, ff, 1)
    else:
        slope, intercept = 0.0, float(np.mean(ff))
    resid = ff - (slope * tt + intercept)
    return {"slope": float(slope), "intercept": float(intercept),
            "sd": float(np.std(resid)), "n": int(inside.sum())}


def side_line(fit, pivot, move=0.0, rotate=0.0):
    """
    One side of the baseline: the fitted line, moved up or down by
    ``move`` at its ``pivot`` time (where it meets the curve) and turned
    about that point by ``rotate`` (added to its slope). Returns {"t",
    "level", "slope", "fit_level", "fit_slope"}.
    """
    pivot = float(pivot)
    level = fit["slope"] * pivot + fit["intercept"]
    return {"t": pivot, "level": float(level + move),
            "slope": float(fit["slope"] + rotate),
            "fit_level": float(level), "fit_slope": float(fit["slope"])}


def line_value(line, t):
    """A side line's value at ``t``."""
    return line["level"] + line["slope"] * (np.asarray(t, float) - line["t"])


def bridged(t, before=None, after=None):
    """
    The baseline at every ``t`` from a line before the curve and a line
    after it (see side_line). With both: the line before up to its point
    (where the curve starts), the line after from its point (where the
    curve ends), and under the curve a straight bridge from one point to
    the other, so each side keeps its own level and tilt. With one: that
    line all the way. None with neither.
    """
    t = np.asarray(t, float)
    if before and after:
        ts, te = before["t"], after["t"]
        out = np.empty_like(t)
        lo, hi = t <= ts, t >= te
        out[lo] = line_value(before, t[lo])
        out[hi] = line_value(after, t[hi])
        mid = ~lo & ~hi
        if te > ts and mid.any():
            out[mid] = before["level"] + (after["level"] - before["level"]) \
                * (t[mid] - ts) / (te - ts)
        return out
    if before:
        return line_value(before, t)
    if after:
        return line_value(after, t)
    return None


def table_to_send(t, f, end_info, zero=0.0, corrected=False, line=None,
                  base=None):
    """Time, force and force less the baseline (``line`` when given, e.g.
    the incline from start to end, else the one after the end), with time
    counted from ``zero``; with ``corrected`` the force less the baseline
    comes first (the column a plot takes)."""
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    out = pd.DataFrame({TIME: t - float(zero), FORCE: f})
    coeffs = line if line is not None else (end_info or {}).get("coeffs")
    if base is not None:
        out[FORCE_LESS_BASE] = f - np.asarray(base, float)
        if corrected:
            out = out[[TIME, FORCE_LESS_BASE, FORCE]]
    elif coeffs is not None:
        out[FORCE_LESS_BASE] = f - np.polyval(coeffs, t)
        if corrected:
            out = out[[TIME, FORCE_LESS_BASE, FORCE]]
    return out


def pad_zero(table, before=0.0, after=0.0):
    """
    The table carried on at zero force for ``before`` seconds ahead of its
    first point and ``after`` seconds past its last, so on a plot beside
    longer recordings the corrected baseline goes on at 0.

    Two points are added on each side (one sample step out, and the far
    end), which a plot joins into a flat line. The force less the baseline
    is 0 there; the force as read carries on along its baseline (the
    straight line between the two columns), or is 0 when there is no
    baseline. The first column is time.
    """
    before = max(float(before or 0.0), 0.0)
    after = max(float(after or 0.0), 0.0)
    if not (before > 0 or after > 0) or len(table) < 2:
        return table
    tcol = table.columns[0]
    t = pd.to_numeric(table[tcol], errors="coerce").to_numpy(float)
    finite = np.isfinite(t)
    if finite.sum() < 2:
        return table
    t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))
    steps = np.diff(np.sort(t[finite]))
    steps = steps[steps > 0]
    dt = float(np.median(steps)) if steps.size else 1e-3
    # the baseline (force as read less force corrected) carries on along a
    # line through its first (or last) fifth: its own end, as drawn
    ends = {}
    if FORCE in table.columns and FORCE_LESS_BASE in table.columns:
        b = (pd.to_numeric(table[FORCE], errors="coerce")
             - pd.to_numeric(table[FORCE_LESS_BASE], errors="coerce")
             ).to_numpy(float)
        ok = finite & np.isfinite(b)
        if ok.sum() >= 4:
            tt, bb = t[ok], b[ok]
            order = np.argsort(tt)
            tt, bb = tt[order], bb[order]
            k = max(2, tt.size // 5)
            ends["before"] = np.polyfit(tt[:k], bb[:k], 1)
            ends["after"] = np.polyfit(tt[-k:], bb[-k:], 1)

    def rows(times, side):
        out = {tcol: times}
        for col in table.columns[1:]:
            if col == FORCE and side in ends:
                out[col] = np.polyval(ends[side], times)
            else:
                out[col] = np.zeros(len(times))
        return pd.DataFrame(out)

    parts = []
    if before > 0:
        parts.append(rows(np.array(sorted({t0 - before,
                                           t0 - min(dt, before)})),
                          "before"))
    parts.append(table)
    if after > 0:
        parts.append(rows(np.array(sorted({t1 + min(dt, after),
                                           t1 + after})), "after"))
    return pd.concat(parts, ignore_index=True)
