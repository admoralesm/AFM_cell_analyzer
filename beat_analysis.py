"""
Beat analysis: a signal's excursions above and below its baseline (a
cardiomyocyte's beats seen as force on the cantilever, or as a post's
deflection), each one's area, the excursions paired into beats, and per
beat the area and the spring energy.

Everything here is plain numbers in, numbers (or a plot) out, so it can be
tested without Streamlit.
"""

import re

import numpy as np
import pandas as pd

BASELINES = ("Median", "Most common level", "Typed or dragged")
ORDERS = ("Positive then negative", "Negative then positive", "Either order")
KINDS = ("Full beat", "Positive only", "Negative only")
POS_COLOUR = "#D62728"
NEG_COLOUR = "#1F77B4"

# what one unit of a force is in newtons
NEWTONS = {"N": 1.0, "mN": 1e-3, "µN": 1e-6, "μN": 1e-6, "uN": 1e-6,
           "nN": 1e-9, "pN": 1e-12}
# what one unit of a deflection is in metres
METRES = {"m": 1.0, "mm": 1e-3, "µm": 1e-6, "μm": 1e-6, "um": 1e-6,
          "nm": 1e-9, "pm": 1e-12}
ENERGY_PREFIXES = ((1.0, "J"), (1e-3, "mJ"), (1e-6, "µJ"), (1e-9, "nJ"),
                   (1e-12, "pJ"), (1e-15, "fJ"), (1e-18, "aJ"),
                   (1e-21, "zJ"))
ENERGY_UNITS = tuple(unit for _, unit in ENERGY_PREFIXES)
FORCE_PREFIXES = ((1.0, "N"), (1e-3, "mN"), (1e-6, "µN"), (1e-9, "nN"),
                  (1e-12, "pN"), (1e-15, "fN"))


def unit_of(label):
    """The unit in brackets at the end of a column name: 'Force (µN)' →
    'µN'; '' when there is none."""
    found = re.search(r"\(([^()]*)\)\s*$", str(label or ""))
    return found.group(1).strip() if found else ""


def newtons_per_unit(unit, k=None, invols=None):
    """
    How many newtons one unit of the signal is. A force: its own size. A
    deflection of the cantilever (m, µm, nm, pm): k × its size in metres,
    since F = k·δ. Volts on the photodiode: k × InvOLS (nm/V) × 1e-9.
    None when it is not a force, or cannot be made one without k (or
    InvOLS).
    """
    unit = str(unit or "").strip()
    if unit in NEWTONS:
        return NEWTONS[unit]
    if k is None or not float(k) > 0:
        return None
    if unit in METRES:
        return float(k) * METRES[unit]
    if unit == "V" and invols is not None and float(invols) > 0:
        return float(k) * float(invols) * 1e-9
    return None


def is_force(unit):
    return str(unit or "").strip() in NEWTONS


def energy_scale(values):
    """(factor, unit) that shows these energies (in J) as numbers near 1:
    e.g. (1e-15, 'fJ')."""
    values = np.abs(np.asarray([v for v in values if v is not None], float))
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return 1.0, "J"
    typical = float(np.median(values))
    for factor, unit in ENERGY_PREFIXES:
        if typical >= factor:
            return factor, unit
    return ENERGY_PREFIXES[-1]


def energy_factor(e_unit=None, energies=()):
    """(factor, unit) for energies in J: the unit asked for (one of
    ENERGY_UNITS), or the one that fits these energies when none is."""
    for factor, unit in ENERGY_PREFIXES:
        if unit == e_unit:
            return factor, unit
    return energy_scale(energies)


def force_scale(values):
    """(factor, unit) that shows these forces (in N) as numbers near 1."""
    values = np.abs(np.asarray([v for v in values if v is not None], float))
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return 1.0, "N"
    typical = float(np.median(values))
    for factor, unit in FORCE_PREFIXES:
        if typical >= factor:
            return factor, unit
    return FORCE_PREFIXES[-1]


# how many seconds one unit of x is, for a power (energy per second)
SECONDS = {"s": 1.0, "sec": 1.0, "seconds": 1.0, "ms": 1e-3, "µs": 1e-6,
           "μs": 1e-6, "us": 1e-6, "min": 60.0, "h": 3600.0}


def seconds_per(x_unit):
    """How many seconds one unit of x is; None when x is not a time."""
    return SECONDS.get(str(x_unit or "").strip())


# the numbers describe() gives, in this order
STATS = ("n", "Mean", "SD", "Median", "Min", "Max", "Range")


def describe(values):
    """
    n, mean, SD, median, min, max and range (max − min) of the finite
    values (None counts as missing), as a dict keyed by STATS. NaN where
    there is nothing to say: all of them for no values, the SD for one.
    """
    v = np.asarray([np.nan if x is None else x for x in values], float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0, **{name: np.nan for name in STATS[1:]}}
    return {"n": int(v.size), "Mean": float(np.mean(v)),
            "SD": float(np.std(v, ddof=1)) if v.size > 1 else np.nan,
            "Median": float(np.median(v)), "Min": float(np.min(v)),
            "Max": float(np.max(v)), "Range": float(np.ptp(v))}


def smooth(y, points=1):
    """A running mean over ``points`` points (1: as it is). The ends keep
    their length (the window shrinks there)."""
    y = np.asarray(y, float)
    points = int(points or 1)
    if points <= 1 or y.size < 3:
        return y
    points = min(points, y.size)
    kernel = np.ones(points)
    ok = np.isfinite(y)
    sums = np.convolve(np.where(ok, y, 0.0), kernel, mode="same")
    counts = np.convolve(ok.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def baseline_level(y, method=BASELINES[0]):
    """
    The signal's resting level: its median, or the level it spends most
    time at (the fullest bin of its histogram, refined by the median of
    the points in it), which is better when the beats take up much of the
    recording.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if not y.size:
        return 0.0
    if method != BASELINES[1]:
        return float(np.median(y))
    lo, hi = np.percentile(y, [1, 99])
    if not hi > lo:
        return float(np.median(y))
    counts, edges = np.histogram(y[(y >= lo) & (y <= hi)], bins=200)
    k = int(np.argmax(counts))
    width = edges[1] - edges[0]
    near = y[(y >= edges[k] - width) & (y <= edges[k + 1] + width)]
    return float(np.median(near)) if near.size else float(
        0.5 * (edges[k] + edges[k + 1]))


def noise_sd(y, base):
    """The baseline's noise: a robust SD (1.4826 × the median absolute
    deviation) of the points near the baseline."""
    r = np.asarray(y, float) - float(base)
    r = r[np.isfinite(r)]
    if not r.size:
        return 0.0
    mad = float(np.median(np.abs(r)))
    return 1.4826 * mad


def suggest_thresholds(y, base, times=5.0):
    """Lines ``times`` × the baseline noise above and below it, and never
    less than 2 % of the largest excursion (so a flat, noiseless signal
    still gets lines)."""
    r = np.asarray(y, float) - float(base)
    r = r[np.isfinite(r)]
    if not r.size:
        return 0.0, 0.0
    sd = noise_sd(y, base)
    up = max(times * sd, 0.02 * float(max(r.max(), 0.0)))
    down = max(times * sd, 0.02 * float(max(-r.min(), 0.0)))
    return up, down


def _trapezoid(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2:
        return 0.0
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def _crossing(x, r, i, j):
    """Where the line from point i to point j crosses r = 0."""
    if r[j] == r[i]:
        return float(x[j])
    return float(x[i] + (0.0 - r[i]) * (x[j] - x[i]) / (r[j] - r[i]))


def excursions(x, y, base, up, down, min_width=0.0, merge=0.0):
    """
    Each stretch where the signal leaves its baseline and gets past one of
    the two lines: ``up`` above the baseline (a positive excursion) or
    ``down`` below it (a negative one). A stretch runs from where the
    signal leaves the baseline to where it comes back (the crossings,
    interpolated), and its area is ∫(y − base) dx over it, by trapezoids.

    ``min_width``: shorter stretches are dropped. ``merge``: stretches of
    the same sign less than this apart (end to start) are one excursion,
    with their areas added (the dip between them not counted).

    Returns dicts sorted by start: sign (+1 / −1), start, end, width,
    peak_x, peak (height from the baseline, signed), area (signed), and
    i0, i1 (the first and last points inside).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return []
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    r = y - float(base)
    found = []
    for sign, line in ((1, up), (-1, down)):
        if line is None or not np.isfinite(line) or line <= 0:
            continue
        s = sign * r
        inside = (s > 0).astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, inside, 0]))
        runs = []
        for a, b in zip(edges[0::2], edges[1::2] - 1):
            seg = s[a:b + 1]
            start = _crossing(x, r, a - 1, a) if a > 0 else float(x[a])
            end = _crossing(x, r, b, b + 1) if b + 1 < x.size else float(x[b])
            xs = np.r_[start, x[a:b + 1], end]
            rs = np.r_[0.0, r[a:b + 1], 0.0]
            k = a + int(np.argmax(seg))
            runs.append({"sign": sign, "start": start, "end": end,
                         "i0": int(a), "i1": int(b),
                         "peak_x": float(x[k]), "peak": float(r[k]),
                         "area": _trapezoid(xs, rs),
                         "reaches": bool(seg.max() >= line)})
        # runs that reach the line; a run that does not may still join a
        # neighbour that does, when they are closer than ``merge``
        merged = []
        for run in runs:
            if merged and merge and run["start"] - merged[-1]["end"] < merge:
                last = merged[-1]
                if sign * run["peak"] > sign * last["peak"]:
                    last["peak"], last["peak_x"] = run["peak"], run["peak_x"]
                last["end"], last["i1"] = run["end"], run["i1"]
                last["area"] += run["area"]
                last["reaches"] = last["reaches"] or run["reaches"]
            else:
                merged.append(dict(run))
        for run in merged:
            run["width"] = run["end"] - run["start"]
            if run.pop("reaches") and run["width"] >= float(min_width or 0):
                found.append(run)
    found.sort(key=lambda e: e["start"])
    return found


def pair_beats(events, max_gap=1.0, order=ORDERS[0]):
    """
    Excursions into beats. A positive one followed by a negative one that
    starts within ``max_gap`` of its end is one full beat (or negative
    then positive, or either order); every other excursion stands alone as
    a positive-only or negative-only beat. Numbered 1, 2, … in time.
    """
    out = []
    i = 0
    events = list(events)
    while i < len(events):
        e = events[i]
        nxt = events[i + 1] if i + 1 < len(events) else None
        first = {ORDERS[0]: 1, ORDERS[1]: -1}.get(order, e["sign"])
        if (nxt is not None and e["sign"] == first
                and nxt["sign"] == -e["sign"]
                and nxt["start"] - e["end"] <= float(max_gap)):
            pos, neg = (e, nxt) if e["sign"] > 0 else (nxt, e)
            out.append({"kind": KINDS[0], "pos": pos, "neg": neg,
                        "start": e["start"], "end": nxt["end"]})
            i += 2
            continue
        out.append({"kind": KINDS[1] if e["sign"] > 0 else KINDS[2],
                    "pos": e if e["sign"] > 0 else None,
                    "neg": e if e["sign"] < 0 else None,
                    "start": e["start"], "end": e["end"]})
        i += 1
    for n, beat in enumerate(out, 1):
        beat["n"] = n
    return out


def spring_energy(height, k, newtons):
    """F² / 2k, in J: the energy in a cantilever (spring constant ``k``, in
    N/m) bent by a force ``height`` (in the signal's unit, ``newtons`` N
    each). None without k or a force unit."""
    if not k or not newtons or height is None or not np.isfinite(height):
        return None
    force = float(height) * float(newtons)
    return force * force / (2.0 * float(k))


def peak_power(event, k, newtons, x_unit):
    """
    A peak's power, in W (J/s): its spring energy F²/2k over its rise time,
    from where the signal leaves the baseline to the peak. That is the
    mean rate at which the cantilever was bent to the peak, i.e. the work
    done on it per second. None without k, a force, or a time on x.
    """
    energy = spring_energy(abs(event["peak"]), k, newtons)
    per = seconds_per(x_unit)
    if energy is None or not per:
        return None
    rise = (float(event["peak_x"]) - float(event["start"])) * per
    return energy / rise if rise > 0 else None


def beat_table(beats, x_unit="", y_unit="", k=None, newtons=None,
               e_unit=None):
    """
    One row per beat: its kind, when it starts and ends, the positive
    peak's height and area, the negative peak's depth and area, the net and
    total area, the spring energy of each peak and of the beat (F² / 2k,
    with k), and the time since the previous beat.
    """
    area_unit = f"{y_unit}·{x_unit}" if (x_unit and y_unit) else "area"
    xu = f" ({x_unit})" if x_unit else ""
    yu = f" ({y_unit})" if y_unit else ""
    au = f" ({area_unit})"
    energies = []
    for b in beats:
        for side in ("pos", "neg"):
            if b[side] is not None:
                energies.append(spring_energy(abs(b[side]["peak"]), k,
                                              newtons))
    factor, e_unit = energy_factor(e_unit, energies)
    rows = []
    previous = None
    for b in beats:
        pos, neg = b["pos"], b["neg"]
        e_pos = spring_energy(abs(pos["peak"]), k, newtons) if pos else None
        e_neg = spring_energy(abs(neg["peak"]), k, newtons) if neg else None
        area_pos = pos["area"] if pos else 0.0
        area_neg = neg["area"] if neg else 0.0
        at = (pos or neg)["peak_x"]
        row = {
            "Beat": b["n"], "Kind": b["kind"],
            f"Start{xu}": b["start"], f"End{xu}": b["end"],
            f"Duration{xu}": b["end"] - b["start"],
            f"Positive peak{yu}": pos["peak"] if pos else np.nan,
            f"Positive peak at{xu}": pos["peak_x"] if pos else np.nan,
            f"Positive area{au}": area_pos if pos else np.nan,
            f"Negative peak{yu}": neg["peak"] if neg else np.nan,
            f"Negative peak at{xu}": neg["peak_x"] if neg else np.nan,
            f"Negative area{au}": area_neg if neg else np.nan,
            f"Net area{au}": area_pos + area_neg,
            f"Total area{au}": abs(area_pos) + abs(area_neg),
            f"Since previous beat{xu}": (at - previous
                                        if previous is not None else np.nan),
        }
        if k and newtons:
            row[f"Spring energy, positive ({e_unit})"] = (
                e_pos / factor if e_pos is not None else np.nan)
            row[f"Spring energy, negative ({e_unit})"] = (
                e_neg / factor if e_neg is not None else np.nan)
            row[f"Spring energy, beat ({e_unit})"] = (
                ((e_pos or 0.0) + (e_neg or 0.0)) / factor)
        rows.append(row)
        previous = at
    return pd.DataFrame(rows)


def summary(beats, x_unit="", y_unit="", k=None, newtons=None,
            per_minute=None, e_unit=None):
    """
    The whole recording at a glance: how many beats of each kind, the
    total and mean (± SD) areas, the mean time between beats and the beat
    rate, and the total spring energy (with k). ``per_minute``: how many x
    units make a minute (60 for seconds), for the rate; None leaves it
    out.
    """
    out = {"Full beats": sum(b["kind"] == KINDS[0] for b in beats),
           "Positive only": sum(b["kind"] == KINDS[1] for b in beats),
           "Negative only": sum(b["kind"] == KINDS[2] for b in beats)}
    pos = [b["pos"] for b in beats if b["pos"] is not None]
    neg = [b["neg"] for b in beats if b["neg"] is not None]
    area_unit = f"{y_unit}·{x_unit}" if (x_unit and y_unit) else ""
    out["Positive area, total"] = (sum(e["area"] for e in pos), area_unit)
    out["Negative area, total"] = (sum(e["area"] for e in neg), area_unit)
    out["Net area, total"] = (sum(e["area"] for e in pos + neg), area_unit)
    full = [b for b in beats if b["kind"] == KINDS[0]]
    per_beat = [b["pos"]["area"] - b["neg"]["area"] for b in full]
    if per_beat:
        out["Total area per full beat"] = (
            float(np.mean(per_beat)),
            float(np.std(per_beat, ddof=1)) if len(per_beat) > 1 else 0.0,
            area_unit)
    times = [(b["pos"] or b["neg"])["peak_x"] for b in beats]
    if len(times) > 1:
        gaps = np.diff(times)
        out["Time between beats"] = (float(np.mean(gaps)),
                                     float(np.std(gaps, ddof=1))
                                     if gaps.size > 1 else 0.0, x_unit)
        if per_minute:
            out["Beat rate (per minute)"] = float(per_minute) / float(
                np.mean(gaps))
    if k and newtons:
        energies = [spring_energy(abs(e["peak"]), k, newtons)
                    for e in pos + neg]
        total = float(sum(e for e in energies if e is not None))
        factor, unit = energy_factor(e_unit, energies)
        out["Spring energy, total"] = (total / factor, unit)
        for name, side in (("positive", pos), ("negative", neg)):
            out[f"Spring energy, {name} peaks"] = (
                float(sum(spring_energy(abs(e["peak"]), k, newtons)
                          for e in side)) / factor, unit)
        each = [e / factor for e in energies if e is not None]
        if each:
            out["Spring energy per peak"] = (
                float(np.mean(each)),
                float(np.std(each, ddof=1)) if len(each) > 1 else 0.0, unit)
        beat_energy = [(spring_energy(abs(b["pos"]["peak"]), k, newtons)
                        + spring_energy(abs(b["neg"]["peak"]), k, newtons))
                       / factor for b in full]
        if beat_energy:
            out["Spring energy per full beat"] = (
                float(np.mean(beat_energy)),
                float(np.std(beat_energy, ddof=1))
                if len(beat_energy) > 1 else 0.0, unit)
    return out


def kind_stats(beats, x_unit="", y_unit="", k=None, newtons=None,
               e_unit=None):
    """
    The beats by kind (full, positive only, negative only), each with how
    many there are and its parameters, every one a (name, values, unit)
    to give as mean ± SD and range. Each peak: its height (or depth), width,
    rise (start to peak), fall (peak to end), area and, with k, energy.
    Full beats also: duration, time between their two peaks, total area
    (both without sign) and the beat's energy. Areas have no sign, so a
    negative peak's adds to a positive one's.
    """
    area_unit = f"{y_unit}·{x_unit}" if (x_unit and y_unit) else ""
    energies = [spring_energy(abs(b[side]["peak"]), k, newtons)
                for b in beats for side in ("pos", "neg")
                if b[side] is not None]
    e_factor, e_name = energy_factor(e_unit, energies)
    with_energy = bool(k and newtons)
    with_power = with_energy and bool(seconds_per(x_unit))

    def energy(e):
        return spring_energy(abs(e["peak"]), k, newtons) / e_factor

    out = []
    for kind in KINDS:
        these = [b for b in beats if b["kind"] == kind]
        params = []
        if these and kind == KINDS[0]:
            params.append(("Duration", [b["end"] - b["start"]
                                        for b in these], x_unit))
        for side, name, size in (("pos", "Positive peak", "height"),
                                 ("neg", "Negative peak", "depth")):
            peaks = [b[side] for b in these if b[side] is not None]
            if not peaks:
                continue
            params += [
                (f"{name} {size}", [abs(e["peak"]) for e in peaks], y_unit),
                (f"{name} width", [e["width"] for e in peaks], x_unit),
                (f"{name} rise (start to peak)",
                 [e["peak_x"] - e["start"] for e in peaks], x_unit),
                (f"{name} fall (peak to end)",
                 [e["end"] - e["peak_x"] for e in peaks], x_unit),
                (f"{name} area" + (" (no sign)" if side == "neg" else ""),
                 [abs(e["area"]) for e in peaks], area_unit)]
            if with_energy:
                params.append((f"{name} energy", [energy(e) for e in peaks],
                               e_name))
            if with_power:
                params.append((f"{name} power (energy ÷ rise)",
                               [peak_power(e, k, newtons, x_unit)
                                for e in peaks], "W"))
        if these and kind == KINDS[0]:
            params += [
                ("Time between its two peaks",
                 [abs(b["neg"]["peak_x"] - b["pos"]["peak_x"])
                  for b in these], x_unit),
                ("Total area (|positive| + |negative|)",
                 [abs(b["pos"]["area"]) + abs(b["neg"]["area"])
                  for b in these], area_unit)]
            if with_energy:
                params.append(("Energy of the beat",
                               [energy(b["pos"]) + energy(b["neg"])
                                for b in these], e_name))
        out.append({"kind": kind, "n": len(these),
                    "params": [(n, np.asarray(v, float), u)
                               for n, v, u in params]})
    return out


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pretty(value, digits=4):
    """
    A number as it reads on a slide: 0.0523 or 544.7 as they are, and in
    scientific notation when very small or large: 5.447 × 10⁻¹⁶ (with real
    superscripts, so it pastes into PowerPoint or Word as it reads). ''
    when it is not a number.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return ""
    if v == 0 or 1e-3 <= abs(v) < 1e5:
        return f"{v:.{digits}g}"
    mantissa, exponent = f"{v:.{max(digits - 1, 0)}e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".") if "." in mantissa \
        else mantissa
    return f"{mantissa} × 10{str(int(exponent)).translate(SUPERSCRIPT)}"


def mean_sd_range(values, unit=""):
    """'52.1 ± 1.2 nN, median 52 (50.3 to 54)': mean ± SD, the median,
    and the range from the smallest to the largest. One value as it is.
    Very small or large numbers in scientific notation (5.447 × 10⁻¹⁶ J)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    u = f" {unit}" if unit else ""
    if not v.size:
        return "–"
    if v.size == 1:
        return f"{pretty(v[0])}{u}"
    return (f"{pretty(np.mean(v))} ± {pretty(np.std(v, ddof=1), 2)}{u}, "
            f"median {pretty(np.median(v))} "
            f"({pretty(np.min(v))} to {pretty(np.max(v))})")


def peak_table(beats, x_unit="", y_unit="", k=None, newtons=None,
               e_unit=None):
    """
    One row per peak integrated (each peak in the beats), in time order:
    its sign, the beat it belongs to, where it starts, peaks and ends, its
    height above (or depth below) the baseline and its integral
    ∫(y − baseline) dx, trapezoid by trapezoid from where it leaves the
    baseline to where it comes back. When y is a deflection made a force
    (``newtons`` per unit), the force at the peak and ∫F dx too. With k,
    the energy: ∫F dδ from the baseline to the peak with δ = F/k, which
    comes to F²/2k; and with x a time, the power: that energy over the
    rise time (start to peak), in W.
    """
    peaks = []
    for b in beats:
        for side in ("pos", "neg"):
            if b[side] is not None:
                peaks.append((b[side], b))
    peaks.sort(key=lambda pair: pair[0]["start"])
    xu = f" ({x_unit})" if x_unit else ""
    yu = f" ({y_unit})" if y_unit else ""
    area_unit = f"{y_unit}·{x_unit}" if (x_unit and y_unit) else "y·x"
    as_force = bool(newtons) and not is_force(y_unit)
    f_factor, f_unit = force_scale(
        [abs(e["peak"]) * newtons for e, _ in peaks] if as_force else [])
    energies = [spring_energy(abs(e["peak"]), k, newtons) for e, _ in peaks]
    e_factor, e_name = energy_factor(e_unit, energies)
    rows = []
    for n, ((e, b), energy) in enumerate(zip(peaks, energies), start=1):
        row = {"Peak": n, "Sign": "+" if e["sign"] > 0 else "−",
               "Beat": b["n"], "Beat kind": b["kind"],
               f"Starts{xu}": e["start"], f"Peaks at{xu}": e["peak_x"],
               f"Ends{xu}": e["end"], f"Width{xu}": e["width"],
               f"Rise (start to peak){xu}": e["peak_x"] - e["start"],
               f"Height{yu}": e["peak"],
               f"∫(y − baseline) dx ({area_unit})": e["area"]}
        if as_force:
            row[f"Force at the peak ({f_unit})"] = (
                e["peak"] * newtons / f_factor)
            row[f"∫F dx ({f_unit}·{x_unit or 'x'})"] = (
                e["area"] * newtons / f_factor)
        if k and newtons:
            row[f"Energy F²/2k ({e_name})"] = (
                energy / e_factor if energy is not None else np.nan)
            if seconds_per(x_unit):
                power = peak_power(e, k, newtons, x_unit)
                row["Power, energy ÷ rise (W)"] = (
                    power if power is not None else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _rgba(hex_colour, alpha):
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def beat_figure(x, y, base, up, down, events, beats, left_out=(),
                colour="#222222", pos_colour=POS_COLOUR,
                neg_colour=NEG_COLOUR, x_title="x", y_title="y",
                height=420, line_width=1.4, numbers=True, shade_beats=False,
                clickable=False, font_size=13, peak_click=False):
    """
    The signal with its baseline and the two lines, each excursion's area
    filled (positive and negative in their colours), its peak marked, each
    beat numbered, the excursions left out greyed, and (optionally) each
    full beat on a light band.
    """
    import plotly.graph_objects as go

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    fig = go.Figure()
    if shade_beats:
        for b in beats:
            if b["kind"] == KINDS[0]:
                fig.add_vrect(x0=b["start"], x1=b["end"], line_width=0,
                              fillcolor="#808080", opacity=0.07,
                              layer="below")
    for sign, e_colour in ((1, pos_colour), (-1, neg_colour)):
        polys_x, polys_y = [], []
        for e in events:
            if e["sign"] != sign:
                continue
            seg = slice(e["i0"], e["i1"] + 1)
            xs = np.r_[e["start"], x[seg], e["end"]]
            ys = np.r_[base, y[seg], base]
            polys_x += list(xs) + [None]
            polys_y += list(ys) + [None]
        if polys_x:
            fig.add_trace(go.Scatter(
                x=polys_x, y=polys_y, fill="toself", mode="none",
                fillcolor=_rgba(e_colour, 0.35), hoverinfo="skip",
                showlegend=False,
                name="positive areas" if sign > 0 else "negative areas"))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers" if clickable else "lines",
        name="signal", showlegend=False,
        line=dict(color=colour, width=line_width),
        marker=dict(size=6, opacity=0) if clickable else None,
        # with peak_click only the peak marks take a click
        hoverinfo="skip" if peak_click else None,
        hovertemplate=None if peak_click
        else "%{x:.4g}, %{y:.4g}<extra></extra>"))
    lo, hi = (float(x.min()), float(x.max())) if x.size else (0.0, 1.0)
    for level, dash, line_colour, name in (
            (base, "solid", "#555555", "baseline"),
            (base + up if up else None, "dash", pos_colour, "upper line"),
            (base - down if down else None, "dash", neg_colour,
             "lower line")):
        if level is None:
            continue
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[level, level], mode="lines", name=name,
            showlegend=False, hoverinfo="skip",
            line=dict(color=line_colour, width=1.4, dash=dash)))
    # with peak_click the marks take the mouse (a click on one picks it),
    # larger, and say so when the mouse is over them
    for sign, symbol, e_colour in ((1, "triangle-up", pos_colour),
                                   (-1, "triangle-down", neg_colour)):
        peaks = [e for e in events if e["sign"] == sign]
        if peaks:
            fig.add_trace(go.Scatter(
                x=[e["peak_x"] for e in peaks],
                y=[base + e["peak"] for e in peaks], mode="markers",
                name="positive peaks" if sign > 0 else "negative peaks",
                showlegend=False,
                hoverinfo=None if peak_click else "skip",
                hovertemplate=("%{x:.4g}: click to remove<extra></extra>"
                               if peak_click else None),
                marker=dict(symbol=symbol, size=13 if peak_click else 9,
                            color=e_colour,
                            line=dict(color="white", width=0.8))))
    if left_out:
        fig.add_trace(go.Scatter(
            x=[e["peak_x"] for e in left_out],
            y=[base + e["peak"] for e in left_out], mode="markers",
            name="left out", showlegend=False,
            hoverinfo=None if peak_click else "skip",
            hovertemplate=("%{x:.4g}: removed, click to put back"
                           "<extra></extra>" if peak_click else None),
            marker=dict(symbol="x-thin", size=14 if peak_click else 12,
                        color="#999999",
                        line=dict(color="#999999", width=2))))
    if numbers and beats:
        kind_colour = {KINDS[0]: "#111111", KINDS[1]: pos_colour,
                       KINDS[2]: neg_colour}
        tops = []
        for b in beats:
            e = b["pos"] or b["neg"]
            tops.append((e["peak_x"], base + e["peak"], str(b["n"]),
                         kind_colour[b["kind"]],
                         "top center" if e["sign"] > 0 else "bottom center"))
        fig.add_trace(go.Scatter(
            x=[t[0] for t in tops], y=[t[1] for t in tops], mode="text",
            text=[t[2] for t in tops], textposition=[t[4] for t in tops],
            textfont=dict(size=max(9, font_size - 2),
                          color=[t[3] for t in tops]),
            name="beat numbers", showlegend=False, hoverinfo="skip",
            cliponaxis=False))
    axis = dict(showline=True, linecolor="black", linewidth=1.2, mirror=True,
                ticks="outside", tickfont=dict(size=font_size),
                title_font=dict(size=font_size + 2), zeroline=False)
    fig.update_layout(
        height=int(height), template="plotly_white",
        margin=dict(l=70, r=20, t=30, b=55), hovermode="closest",
        xaxis=dict(title=x_title, **axis), yaxis=dict(title=y_title, **axis),
        font=dict(size=font_size))
    if peak_click:
        fig.update_layout(dragmode="zoom", clickmode="event+select")
    return fig
