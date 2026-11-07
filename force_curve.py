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


def corrected(t, f, end_info, start_info=None):
    """
    The force less the baseline fitted after the end, so the curve ends
    on zero, with both baselines put in those terms (the one after the end
    becomes 0). Returns (force, end info, start info); unchanged when there
    is no baseline.
    """
    coeffs = (end_info or {}).get("coeffs")
    if coeffs is None:
        return np.asarray(f, float), end_info, start_info
    t = np.asarray(t, float)
    coeffs = np.asarray(coeffs, float)
    out = np.asarray(f, float) - np.polyval(coeffs, t)
    end_c = dict(end_info, coeffs=np.array([0.0]), level=0.0)
    start_c = start_info
    if start_info and start_info.get("coeffs") is not None:
        diff = np.polysub(np.asarray(start_info["coeffs"], float), coeffs)
        at = (start_info["start"] if start_info.get("start") is not None
              else start_info["window"][1])
        start_c = dict(start_info, coeffs=diff,
                       level=float(np.polyval(diff, at)))
    return out, end_c, start_c


def table_to_send(t, f, end_info, zero=0.0, corrected=False):
    """Time, force and force less the baseline after the end, with time
    counted from ``zero``; with ``corrected`` the force less the baseline
    comes first (the column a plot takes)."""
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    out = pd.DataFrame({TIME: t - float(zero), FORCE: f})
    coeffs = (end_info or {}).get("coeffs")
    if coeffs is not None:
        out[FORCE_LESS_BASE] = f - np.polyval(coeffs, t)
        if corrected:
            out = out[[TIME, FORCE_LESS_BASE, FORCE]]
    return out
