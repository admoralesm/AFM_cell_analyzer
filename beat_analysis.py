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
# a peak's timing: it starts where, on its rise, the signal gets out past a
# level near the baseline and ends where, on its fall, it is back inside it.
# The level is a number of noise SDs (2 unless changed: the peak starts
# where it leaves the noise and ends where it is back in it) or a part of
# the peak's height (10 %: 10 % up to 90 % back). Where the signal crosses
# the baseline itself is lost in the noise and in any slow drift; these
# levels are not. The width at half height is read the same way at 50 %.
TIMINGS = ("Back within × noise SD", "At a % of the peak's height")
TIMING_NOISE = 2.0
TIMING_PERCENT = 10.0
TIMING_LEVEL = TIMING_PERCENT / 100
HALF_LEVEL = 0.50


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


def area_word(x_unit, y_unit):
    """What the area under a peak is called: 'Impulse' for a force against
    time (force × time, e.g. µN·s), else 'Area' (∫ of y over x)."""
    return ("Impulse" if is_force(y_unit) and seconds_per(x_unit)
            else "Area")


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
    sd = float(np.std(v, ddof=1)) if v.size > 1 else np.nan
    if sd < 1e-9 * float(np.max(np.abs(v))):
        sd = 0.0                    # equal values, but for rounding
    return {"n": int(v.size), "Mean": float(np.mean(v)), "SD": sd,
            "Median": float(np.median(v)), "Min": float(np.min(v)),
            "Max": float(np.max(v)),
            "Range": float(np.ptp(v)) if sd else 0.0}


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


def noise_stats(x, y, base, start=None, end=None, events=()):
    """
    The noise of a signal, measured one of two ways:

    - manually, on a quiet stretch: the points with ``start`` ≤ x ≤ ``end``;
    - automatically, between the peaks: every point outside the spans
      (start to end) of ``events``, the peaks found past the lines. With
      fewer than 10 such points, all of them.

    Returns None when there are fewer than 3 points, else a dict: ``n``
    points, ``x0``/``x1`` the stretch used, ``sd`` (about its own mean),
    ``robust_sd`` (1.4826 × the median absolute deviation from its median,
    unmoved by a stray spike), ``rms`` (about the baseline), ``p2p`` (peak
    to peak, largest minus smallest), ``offset`` (its mean minus the
    baseline: how far the baseline sits from the quiet level), and
    ``noise``, the one to use: the plain SD for a stretch chosen by hand,
    the robust SD for the automatic one.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    manual = start is not None and end is not None
    if manual:
        lo, hi = sorted((float(start), float(end)))
        keep &= (x >= lo) & (x <= hi)
    else:
        between = keep.copy()
        for e in events or ():
            between &= ~((x >= e["start"]) & (x <= e["end"]))
        if between.sum() >= 10:
            keep = between
    v, xs = y[keep], x[keep]
    if v.size < 3:
        return None
    median = float(np.median(v))
    out = {"n": int(v.size), "x0": float(xs.min()), "x1": float(xs.max()),
           "sd": float(np.std(v, ddof=1)),
           "robust_sd": 1.4826 * float(np.median(np.abs(v - median))),
           "rms": float(np.sqrt(np.mean((v - float(base)) ** 2))),
           "p2p": float(np.max(v) - np.min(v)),
           "offset": float(np.mean(v) - float(base)),
           "manual": bool(manual)}
    out["noise"] = out["sd"] if manual else out["robust_sd"]
    return out


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


def _level_times(xs, ss, kk, level):
    """
    Where a peak's signal ``ss`` (made positive, 0 at both ends), going
    out from its top at index ``kk``, first drops under ``level``: on the
    rise (back in time) and on the fall, each interpolated between the
    points either side. (x on the rise, x on the fall.)
    """
    def cross(a, b):
        if ss[b] == ss[a]:
            return float(xs[b])
        return float(xs[a] + (level - ss[a]) * (xs[b] - xs[a])
                     / (ss[b] - ss[a]))
    under = np.flatnonzero(ss[:kk] < level)
    a = int(under[-1]) if under.size else 0
    rise = cross(a, a + 1) if under.size else float(xs[0])
    under = np.flatnonzero(ss[kk + 1:] < level)
    b = kk + 1 + int(under[0]) if under.size else len(ss) - 1
    fall = cross(b - 1, b) if under.size else float(xs[-1])
    return rise, fall


def timing_words(rule=None, value=None):
    """What a duration runs between, in a few words: "within 2 × noise SD"
    or "10% up to 90% back"."""
    if rule == TIMINGS[1]:
        p = float(TIMING_PERCENT if value is None else value)
        return f"{p:g}% up to {100 - p:g}% back"
    k = float(TIMING_NOISE if value is None else value)
    return f"within {k:g} × noise SD"


def timing_level(event, rule=None, value=None, noise=None):
    """
    The level (from the baseline, as a size) at which a peak starts and
    ends: ``value`` noise SDs (2 unless given) or ``value`` % of its height
    (10 unless given). With no noise measured, the baseline itself. Never
    more than half the peak's height.
    """
    height = abs(float(event["peak"]))
    if rule == TIMINGS[1]:
        p = float(TIMING_PERCENT if value is None else value)
        level = p / 100 * height
    else:
        k = float(TIMING_NOISE if value is None else value)
        level = k * float(noise) if noise and noise > 0 else 0.0
    return min(max(level, 0.0), HALF_LEVEL * height)


def retime(events, x, y, base, rule=None, value=None, noise=None):
    """Each peak's timing again with the rule chosen (``timing_level``)."""
    words = timing_words(rule, value)
    if rule != TIMINGS[1] and not (noise and noise > 0):
        words = "at the baseline (no noise)"
    for e in events:
        add_timing(e, x, y, base, timing_level(e, rule, value, noise), words)
    return events


def timing_of(beats):
    """The words of the timing rule the beats' peaks were timed with."""
    for b in beats:
        for side in ("pos", "neg"):
            if b.get(side) is not None and b[side].get("timing"):
                return b[side]["timing"]
    return "at the baseline"


def add_timing(event, x, y, base, level=None, words=None):
    """
    Adds a peak's timing, read going out from its top to where the signal
    is ``level`` from the baseline (10 % of its height unless given):
    ``up`` on the rise, ``back`` on the fall, ``up50``/``back50`` the same
    at half height, ``dur`` (up to back), ``half_width`` (up50 to back50),
    ``rise`` (up to the peak) and ``recovery`` (the peak to back). Going
    out from the peak, a slow drift before the beat or a slow return after
    it, inside the level, does not stretch it.
    """
    x = np.asarray(x, float)
    s = event["sign"] * (np.asarray(y, float) - float(base))
    i0, i1 = int(event["i0"]), int(event["i1"])
    xs = np.r_[event["start"], x[i0:i1 + 1], event["end"]]
    ss = np.r_[0.0, s[i0:i1 + 1], 0.0]
    kk = int(np.argmax(ss))
    height = float(ss[kk])
    if not height > 0:
        return event
    if level is None:
        level = TIMING_LEVEL * height
        words = words or timing_words(TIMINGS[1], TIMING_PERCENT)
    level = min(max(float(level), 0.0), HALF_LEVEL * height)
    up, back = _level_times(xs, ss, kk, level)
    up50, back50 = _level_times(xs, ss, kk, HALF_LEVEL * height)
    event.update(up=up, back=back, level=level, timing=words,
                 up50=up50, back50=back50,
                 dur=back - up, half_width=back50 - up50,
                 rise=float(xs[kk]) - up, recovery=back - float(xs[kk]))
    return event


def peak_duration(event):
    """A peak's duration: from where it starts to where it ends, by the
    timing rule (at the baseline when its timing is not there)."""
    return float(event.get("dur", event["end"] - event["start"]))


def rise_time(event):
    """A peak's rise: from where it starts (by the timing rule) to the
    peak."""
    return float(event.get("rise", event["peak_x"] - event["start"]))


def recovery_time(event):
    """A peak's recovery: the peak to where it ends (by the timing
    rule)."""
    return float(event.get("recovery", event["end"] - event["peak_x"]))


def beat_duration(beat):
    """A beat's duration: from where its first peak starts to where its
    last one ends, by the timing rule. A full beat, Positive then negative:
    the positive peak's start to the negative peak's end."""
    return float(beat.get("duration", beat["end"] - beat["start"]))


def _ratio(a, b):
    """a ÷ b, NaN when either is missing or b is 0."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return float("nan")
    return a / b


def beat_symmetry(beat):
    """
    How a beat's positive peak compares with its negative one (positive ÷
    negative, as the positive is usually the larger), and how lopsided each
    peak is in time:

    - ``depth_ratio``: the positive peak's height ÷ the negative's depth;
    - ``impulse_ratio``: the positive peak's area ÷ the negative's (both
      without sign); 1 when the area above the baseline matches the area
      below it;
    - ``duration_ratio``: the positive peak's duration ÷ the negative's
      (by the timing rule);
    - ``balance``: (P − N) ÷ (P + N) of the two areas: 0 when they match,
      +1 all above the baseline, −1 all below;
    - ``pos_shape`` / ``neg_shape``: each peak's recovery ÷ its rise
      (peak to end ÷ start to peak): 1 when it falls as fast as it rises,
      above 1 for a slow fall, a tail or a shoulder after the peak.

    The ratios between the two peaks need both (a full beat); NaN
    otherwise.
    """
    pos, neg = beat.get("pos"), beat.get("neg")
    nan = float("nan")
    out = {"depth_ratio": nan, "impulse_ratio": nan, "duration_ratio": nan,
           "balance": nan, "pos_shape": nan, "neg_shape": nan}
    if pos is not None:
        out["pos_shape"] = _ratio(recovery_time(pos), rise_time(pos))
    if neg is not None:
        out["neg_shape"] = _ratio(recovery_time(neg), rise_time(neg))
    if pos is not None and neg is not None:
        P, N = abs(float(pos["area"])), abs(float(neg["area"]))
        out["depth_ratio"] = _ratio(abs(pos["peak"]), abs(neg["peak"]))
        out["impulse_ratio"] = _ratio(P, N)
        out["duration_ratio"] = _ratio(peak_duration(pos),
                                       peak_duration(neg))
        out["balance"] = (P - N) / (P + N) if P + N > 0 else nan
    return out


# A beat drawn in the plane of time and force can only have the symmetries
# of the Klein four-group V = {e, T, S, TS} ≅ Z₂ × Z₂ (abelian): e leaves it
# alone, T mirrors time about a moment c (t → 2c − t), S flips the force's
# sign (F → −F), and TS does both, a half-turn about (c, baseline). No graph
# but a flat line is unchanged by S, so a peak can only be mirror-symmetric
# (T) about its top and a full beat only half-turn symmetric (TS) about the
# moment it crosses the baseline. As V is abelian, each element splits a
# waveform g into a part it leaves unchanged and a part it reverses,
# g = (g + Tg)/2 + (g − Tg)/2, which are orthogonal; the share of ∫g² dt in
# the unchanged part measures the symmetry: (1 + c)/2 for T and (1 − c)/2
# for TS, c = ∫g(t) g(2c − t) dt / ∫g² dt. 1: exactly symmetric; 0.5: no
# more symmetric than not; 0: exactly the opposite.


def symmetry_share(x, r, centre, a, b, turn=False, points=801):
    """
    The share of ``r`` (the signal less its baseline), on [a, b] and nought
    outside, that is left unchanged by mirroring time about ``centre``
    (``turn`` False: g(2c − t) = g(t)) or by a half-turn about (centre, 0)
    (``turn`` True: g(2c − t) = −g(t)): the squared size of its symmetric
    part over its own. NaN when there is nothing there.
    """
    half = max(float(centre) - float(a), float(b) - float(centre))
    if not half > 0:
        return float("nan")
    grid = np.linspace(float(centre) - half, float(centre) + half, points)
    g = np.interp(grid, x, r)
    g[(grid < a) | (grid > b)] = 0.0
    total = float(np.dot(g, g))
    if not total > 0:
        return float("nan")
    c = float(np.dot(g, g[::-1])) / total
    return (1.0 - c) / 2.0 if turn else (1.0 + c) / 2.0


def add_symmetry(beats, x, y, base):
    """
    Each peak's mirror symmetry about its top (``mirror``: on its own side
    of the baseline, from where it leaves it to where it comes back) and
    each full beat's half-turn symmetry about where it crosses the baseline
    between its two peaks (``point``, from the first peak's start to the
    second's end), as shares in [0, 1] (``symmetry_share``).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    order = np.argsort(x[ok], kind="stable")
    x, r = x[ok][order], y[ok][order] - float(base)
    for b in beats:
        for side in ("pos", "neg"):
            e = b.get(side)
            if e is not None:
                own = np.clip(e["sign"] * r, 0.0, None)
                e["mirror"] = symmetry_share(x, own, e["peak_x"], e["start"],
                                             e["end"])
        if b.get("pos") is not None and b.get("neg") is not None:
            first, second = sorted((b["pos"], b["neg"]),
                                   key=lambda e: e["start"])
            centre = (float(first["end"]) + float(second["start"])) / 2.0
            b["point"] = symmetry_share(x, r, centre, first["start"],
                                        second["end"], turn=True)
    return beats


def beat_span(beat):
    """Where a beat starts and ends: its first peak leaving the baseline
    to its last one coming back (by the timing rule when it has been
    applied, else where the signal crosses the baseline). (None, None) for
    a beat with no peak."""
    parts = [e for e in (beat.get("pos"), beat.get("neg")) if e]
    starts = [e.get("up", e["start"]) for e in parts]
    ends = [e.get("back", e["end"]) for e in parts]
    starts = [float(v) for v in starts if v is not None and np.isfinite(v)]
    ends = [float(v) for v in ends if v is not None and np.isfinite(v)]
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def _loop_area(a, b):
    """The area a closed loop of points encloses (the shoelace formula),
    whichever way round it goes."""
    return float(abs(np.dot(a, np.roll(b, -1)) - np.dot(b, np.roll(a, -1)))
                 / 2.0)


def beat_z(beats, x, f, z, base=0.0):
    """
    Each beat with the Z piezo's movement through it, for a force recorded
    with the Z piezo beside it (an AFM's Z sensor or height channel, at the
    same times ``x``). Over each beat, from where it starts to where it
    ends (beat_span): how far Z moved (its largest less its smallest), and
    the loop the force (``f`` less ``base``) draws against Z, with its
    area, force × Z.

    That area is the net work done on the probe over the beat: the tip
    moves with Z plus the deflection, and over a closed loop the
    cantilever's own spring gives back all it took (∮F d(deflection) = 0,
    as F = k × deflection), so ∮F d(tip) = ∮F dZ. Turning Z round (a
    sensor that counts the other way) mirrors the loop and keeps its area.

    One dict per beat with at least 3 points: n, t0, t1, z_move,
    loop_area, loop_z (Z from where the beat starts), loop_f.
    """
    x = np.asarray(x, float)
    f = np.asarray(f, float) - float(base)
    z = np.asarray(z, float)
    good = np.isfinite(x) & np.isfinite(f) & np.isfinite(z)
    out = []
    for b in beats:
        t0, t1 = beat_span(b)
        if t0 is None:
            continue
        inside = good & (x >= t0) & (x <= t1)
        if inside.sum() < 3:
            continue
        zz, ff = z[inside], f[inside]
        out.append({"n": b.get("n"), "t0": t0, "t1": t1,
                    "z_move": float(np.ptp(zz)),
                    "loop_area": _loop_area(zz, ff),
                    "loop_z": zz - zz[0], "loop_f": ff})
    return out


def _edge(e, key, fallback):
    v = e.get(key)
    return float(v) if v is not None and np.isfinite(v) else float(e[fallback])


def peak_z(beats, x, f, z):
    """
    Each peak (positive and negative, on its own) with the Z piezo through
    it, both measured from where the peak starts: the force (or deflection)
    and Z there are taken off, so every peak's path starts at (0, 0) and
    the peaks lie over one another, whatever the drift between them. From
    the peak's start to its end (``up`` and ``back`` by the timing rule
    when set, else where it leaves and meets the baseline).

    One dict per peak with at least 3 points: n (its beat's number), side
    ("pos" or "neg"), t0, t1, z_move (Z's largest less its smallest),
    z_net (Z at the end less at the start), loop_area (the area the path
    encloses, closed back to its start: force × Z, the work the peak's own
    loop holds), loop_z and loop_f (the path, from (0, 0)).
    """
    x = np.asarray(x, float)
    f = np.asarray(f, float)
    z = np.asarray(z, float)
    good = np.isfinite(x) & np.isfinite(f) & np.isfinite(z)
    xs, fs, zs = x[good], f[good], z[good]
    out = []
    if xs.size < 3:
        return out
    for b in beats:
        for side in ("pos", "neg"):
            e = b.get(side)
            if not e:
                continue
            t0, t1 = _edge(e, "up", "start"), _edge(e, "back", "end")
            inside = (xs > t0) & (xs < t1)
            if inside.sum() < 2:
                continue
            z0, f0 = np.interp(t0, xs, zs), np.interp(t0, xs, fs)
            z1, f1 = np.interp(t1, xs, zs), np.interp(t1, xs, fs)
            pz = np.r_[z0, zs[inside], z1] - z0
            pf = np.r_[f0, fs[inside], f1] - f0
            out.append({"n": b.get("n"), "side": side, "t0": t0, "t1": t1,
                        "z_move": float(np.ptp(pz)),
                        "z_net": float(pz[-1]),
                        "loop_area": _loop_area(pz, pf),
                        "loop_z": pz, "loop_f": pf})
    return out


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
    i0, i1 (the first and last points inside), with the timing read on the
    peak's height (``add_timing``: 10 % up, 90 % back, half height; see
    ``retime`` for another rule).
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
                found.append(add_timing(run, x, y, base))
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
                        "start": e["start"], "end": nxt["end"],
                        # the first peak's start to the second's end,
                        # by the timing rule
                        "duration": (nxt.get("back", nxt["end"])
                                     - e.get("up", e["start"]))})
            i += 2
            continue
        out.append({"kind": KINDS[1] if e["sign"] > 0 else KINDS[2],
                    "pos": e if e["sign"] > 0 else None,
                    "neg": e if e["sign"] < 0 else None,
                    "start": e["start"], "end": e["end"],
                    "duration": peak_duration(e)})
        i += 1
    for n, beat in enumerate(out, 1):
        beat["n"] = n
    return out


# The rhythm: what each beat is, by when it comes and what it looks like
# (classify_rhythm), in the order a beat is called: a run of early beats
# first, then one early or late beat, then its shape, then alternans
RHYTHMS = ("Regular", "Premature", "After a pause", "Fast run",
           "Fibrillation-like", "Alternans", "Positive only",
           "Negative only")
RHYTHM_COLOURS = {"Regular": "#2ca02c", "Premature": "#ff7f0e",
                  "After a pause": "#9467bd", "Fast run": "#8c564b",
                  "Fibrillation-like": "#d62728", "Alternans": "#17becf",
                  "Positive only": "#e377c2", "Negative only": "#7f7f7f"}
# how a beat is sorted unless set otherwise (fractions of the usual time
# between beats, or of the usual positive peak; runs in beats)
RHYTHM_DEFAULTS = {"early": 0.75, "late": 1.5, "run": 3, "irregular": 0.25,
                   "alternans": 0.15, "alt_run": 4}


def beat_time(beat):
    """When a beat happens: its positive peak's top (its negative peak's,
    for a negative-only beat), as the time between beats is measured."""
    return float((beat["pos"] or beat["neg"])["peak_x"])


def usual_interval(beats):
    """
    The usual time between beats: the median of the times between two
    full beats in a row (between any two beats in a row when there are
    fewer than three of those). NaN with fewer than two beats.
    """
    t = np.array([beat_time(b) for b in beats], float)
    if t.size < 2:
        return float("nan")
    rr = np.diff(t)
    full = [rr[i - 1] for i in range(1, len(beats))
            if beats[i]["kind"] == KINDS[0]
            and beats[i - 1]["kind"] == KINDS[0]]
    return float(np.median(full if len(full) >= 3 else rr))


def _alternating(heights, step, least):
    """The stretches (first, last index, inclusive) of at least ``least``
    heights in a row that go up and down in turn, each change at least
    ``step``."""
    out, start = [], 0
    for k in range(1, len(heights) + 1):
        ok = False
        if k < len(heights):
            d = heights[k] - heights[k - 1]
            ok = abs(d) >= step and (
                k - start < 2
                or np.sign(d) == -np.sign(heights[k - 1] - heights[k - 2]))
        if ok:
            continue
        if k - start >= least:
            out.append((start, k - 1))
        if k < len(heights):
            d = heights[k] - heights[k - 1]
            start = k - 1 if abs(d) >= step else k
    return out


def classify_rhythm(beats, usual=None, early=RHYTHM_DEFAULTS["early"],
                    late=RHYTHM_DEFAULTS["late"], run=RHYTHM_DEFAULTS["run"],
                    irregular=RHYTHM_DEFAULTS["irregular"],
                    alternans=RHYTHM_DEFAULTS["alternans"],
                    alt_run=RHYTHM_DEFAULTS["alt_run"]):
    """
    What each beat is (one of RHYTHMS), from the time since the beat
    before it against the usual time between beats (``usual``, or the
    median between full beats, usual_interval) and from its shape:

    - Fast run: ``run`` or more beats in a row, each early (sooner than
      ``early`` × usual after the one before), the times between them
      steady; Fibrillation-like: such a run whose times vary (their
      coefficient of variation at least ``irregular``): rapid and chaotic.
    - Premature: an early beat on its own (or two in a row).
    - After a pause: later than ``late`` × usual after the one before.
    - Positive only, Negative only: on time, but half a beat.
    - Alternans: ``alt_run`` or more full beats in a row on time whose
      positive peaks go large, small, large… each change at least
      ``alternans`` × the usual positive peak.
    - Regular: a full beat on time, in none of those.

    The first beat, with none before it, is judged by its shape. -> a dict:
    labels (one per beat), rr (the time since the beat before, NaN for the
    first), usual, usual_auto (True when worked out), usual_height (the
    median positive peak of the full beats) and episodes: the runs, the
    alternans and the pauses, each a dict with kind, start, end (in x),
    beats (their indices) and, for a run, cv; for a pause, rr (the time
    between the beats either side, its length).
    """
    n = len(beats)
    t = np.array([beat_time(b) for b in beats], float)
    rr = np.full(n, np.nan)
    if n > 1:
        rr[1:] = np.diff(t)
    usual_auto = not (usual is not None and np.isfinite(usual)
                      and float(usual) > 0)
    usual = usual_interval(beats) if usual_auto else float(usual)
    tall = [abs(b["pos"]["peak"]) for b in beats if b["kind"] == KINDS[0]]
    if not tall:
        tall = [abs((b["pos"] or b["neg"])["peak"]) for b in beats]
    usual_height = float(np.median(tall)) if tall else float("nan")
    labels = [None] * n
    episodes = []
    if n > 1 and np.isfinite(usual) and usual > 0:
        with np.errstate(invalid="ignore"):
            is_early = rr < float(early) * usual
            is_late = rr > float(late) * usual
        i = 1
        while i < n:
            if not is_early[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and is_early[j + 1]:
                j += 1
            members = list(range(i, j + 1))
            if len(members) >= max(int(run), 1):
                gaps = rr[i:j + 1]
                cv = (float(np.std(gaps, ddof=1) / np.mean(gaps))
                      if gaps.size > 1 and np.mean(gaps) > 0 else 0.0)
                kind = "Fibrillation-like" if cv >= float(irregular) \
                    else "Fast run"
                for m in members:
                    labels[m] = kind
                episodes.append({"kind": kind, "start": beats[i]["start"],
                                 "end": beats[j]["end"], "beats": members,
                                 "cv": cv})
            else:
                for m in members:
                    labels[m] = "Premature"
            i = j + 1
        for i in range(1, n):
            if labels[i] is None and is_late[i]:
                labels[i] = "After a pause"
                # shaded from the beat before's end to this one's start;
                # as long as the time between the two
                episodes.append({"kind": "Pause", "start": beats[i - 1]["end"],
                                 "end": beats[i]["start"], "beats": [i],
                                 "rr": float(rr[i])})
    for i, b in enumerate(beats):
        if labels[i] is None and b["kind"] != KINDS[0]:
            labels[i] = b["kind"]
    # alternans, among the full beats on time, in stretches with no other
    # beat between them
    if (alternans and alternans > 0 and np.isfinite(usual_height)
            and usual_height > 0):
        stretch = []
        for i in list(range(n)) + [None]:
            if i is not None and labels[i] is None:
                stretch.append(i)
                continue
            heights = [abs(beats[m]["pos"]["peak"]) for m in stretch]
            for a, b_ in _alternating(heights, float(alternans)
                                      * usual_height, max(int(alt_run), 2)):
                members = stretch[a:b_ + 1]
                for m in members:
                    labels[m] = "Alternans"
                episodes.append({"kind": "Alternans",
                                 "start": beats[members[0]]["start"],
                                 "end": beats[members[-1]]["end"],
                                 "beats": members})
            stretch = []
    labels = [lab or RHYTHMS[0] for lab in labels]
    episodes.sort(key=lambda e: e["start"])
    return {"labels": labels, "rr": rr, "usual": usual,
            "usual_auto": usual_auto, "usual_height": usual_height,
            "episodes": episodes}


def rhythm_numbers(rhythm, span=None):
    """
    The rhythm over a recording, as (name, value, unit) rows: how many
    beats of each type (and the regular ones' share), the pauses and the
    longest, the runs, the time fibrillation-like (and its share of
    ``span``, the recording's length, when given), and how irregular the
    times between beats are: their coefficient of variation, and the
    beat-to-beat change (RMSSD) over their mean. Units: "n", "%", "x"
    (the x unit) or "" (a ratio).
    """
    labels = list(rhythm["labels"])
    count = {name: labels.count(name) for name in RHYTHMS}
    total = len(labels)
    eps = rhythm["episodes"]
    pauses = [e.get("rr", e["end"] - e["start"]) for e in eps
              if e["kind"] == "Pause"]
    fib = [e["end"] - e["start"] for e in eps
           if e["kind"] == "Fibrillation-like"]
    rr = np.asarray(rhythm["rr"], float)
    rr = rr[np.isfinite(rr)]
    mean = float(np.mean(rr)) if rr.size else float("nan")
    cv = (float(np.std(rr, ddof=1)) / mean
          if rr.size > 1 and mean > 0 else float("nan"))
    rmssd = (float(np.sqrt(np.mean(np.diff(rr) ** 2))) / mean
             if rr.size > 2 and mean > 0 else float("nan"))
    fib_time = float(sum(fib))
    return [
        ("Regular beats", count["Regular"], "n"),
        ("Regular beats, % of all",
         100.0 * count["Regular"] / total if total else float("nan"), "%"),
        ("Premature beats", count["Premature"], "n"),
        ("Beats after a pause", count["After a pause"], "n"),
        ("Longest pause", float(max(pauses)) if pauses else 0.0, "x"),
        ("Fast runs", sum(e["kind"] == "Fast run" for e in eps), "n"),
        ("Beats in fast runs", count["Fast run"], "n"),
        ("Fibrillation-like episodes", len(fib), "n"),
        ("Fibrillation-like beats", count["Fibrillation-like"], "n"),
        ("Fibrillation-like time", fib_time, "x"),
        ("Fibrillation-like time, % of the recording",
         100.0 * fib_time / float(span) if span else float("nan"), "%"),
        ("Alternans beats", count["Alternans"], "n"),
        ("Rhythm irregularity (CV of the time between beats)", cv, ""),
        ("Beat-to-beat change (RMSSD ÷ mean time between beats)", rmssd,
         "")]


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
    from 10 % of its height on the rise to the peak. That is the mean rate
    at which the cantilever was bent to the peak, i.e. the work done on it
    per second. None without k, a force, or a time on x.
    """
    energy = spring_energy(abs(event["peak"]), k, newtons)
    per = seconds_per(x_unit)
    if energy is None or not per:
        return None
    rise = rise_time(event) * per
    return energy / rise if rise > 0 else None


def beat_table(beats, x_unit="", y_unit="", k=None, newtons=None,
               e_unit=None, timing=None):
    """
    One row per beat: its kind, when it starts and ends, the positive
    peak's height and area, the negative peak's depth and area, the net and
    total area, how the two peaks compare (``beat_symmetry``: depth ÷
    height, area ÷ area, duration ÷ duration, the balance of the areas, and
    each peak's recovery ÷ rise), the spring energy of each peak and of the
    beat (F² / 2k, with k), and the time since the previous beat.
    """
    area_unit = f"{y_unit}·{x_unit}" if (x_unit and y_unit) else "area"
    xu = f" ({x_unit})" if x_unit else ""
    yu = f" ({y_unit})" if y_unit else ""
    au = f" ({area_unit})"
    w = area_word(x_unit, y_unit).lower()
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
        sym = beat_symmetry(b)
        row = {
            "Beat": b["n"], "Kind": b["kind"],
            f"Leaves baseline{xu}": b["start"],
            f"Back at baseline{xu}": b["end"],
            f"Duration, {timing or timing_of(beats)}{xu}": beat_duration(b),
            f"Positive peak{yu}": pos["peak"] if pos else np.nan,
            f"Positive peak at{xu}": pos["peak_x"] if pos else np.nan,
            f"Positive {w}{au}": area_pos if pos else np.nan,
            f"Negative peak{yu}": neg["peak"] if neg else np.nan,
            f"Negative peak at{xu}": neg["peak_x"] if neg else np.nan,
            f"Negative {w}{au}": area_neg if neg else np.nan,
            f"Net {w}{au}": area_pos + area_neg,
            f"Total {w}{au}": abs(area_pos) + abs(area_neg),
            "Positive ÷ negative peak": sym["depth_ratio"],
            f"Positive ÷ negative {w}": sym["impulse_ratio"],
            "Positive ÷ negative duration": sym["duration_ratio"],
            f"{w.capitalize()} balance (P − N) ÷ (P + N)": sym["balance"],
            "Positive peak recovery ÷ rise": sym["pos_shape"],
            "Negative peak recovery ÷ rise": sym["neg_shape"],
            # the group's two possible symmetries, as shares (add_symmetry)
            "Half-turn symmetry about the crossing": b.get("point", np.nan),
            "Positive peak mirror symmetry": pos.get("mirror", np.nan)
            if pos else np.nan,
            "Negative peak mirror symmetry": neg.get("mirror", np.nan)
            if neg else np.nan,
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
    w = area_word(x_unit, y_unit)
    out[f"Positive {w.lower()}, total"] = (sum(e["area"] for e in pos),
                                           area_unit)
    out[f"Negative {w.lower()}, total"] = (sum(e["area"] for e in neg),
                                           area_unit)
    out[f"Net {w.lower()}, total"] = (sum(e["area"] for e in pos + neg),
                                      area_unit)
    full = [b for b in beats if b["kind"] == KINDS[0]]
    per_beat = [b["pos"]["area"] - b["neg"]["area"] for b in full]
    if per_beat:
        out[f"Total {w.lower()} per full beat"] = (
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
               e_unit=None, timing=None):
    """
    The beats by kind (full, positive only, negative only), each with how
    many there are and its parameters, every one a (name, values, unit)
    to give as mean ± SD and range. Each peak: its height (or depth),
    duration (by the timing rule, ``timing`` its words), width at half
    height, rise (start to peak), recovery (peak to end), area and, with k,
    energy.
    Full beats also: duration, time between their two peaks, total area
    (both without sign) and the beat's energy. Areas have no sign, so a
    negative peak's adds to a positive one's.
    """
    area_unit = f"{y_unit}·{x_unit}" if (x_unit and y_unit) else ""
    w = area_word(x_unit, y_unit)
    timing = timing or timing_of(beats)
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
            params.append((f"Duration ({timing})",
                           [beat_duration(b) for b in these], x_unit))
        for side, name, size in (("pos", "Positive peak", "height"),
                                 ("neg", "Negative peak", "depth")):
            peaks = [b[side] for b in these if b[side] is not None]
            if not peaks:
                continue
            params += [
                (f"{name} {size}", [abs(e["peak"]) for e in peaks], y_unit),
                (f"{name} duration ({timing})",
                 [peak_duration(e) for e in peaks], x_unit),
                (f"{name} width at half height",
                 [e.get("half_width", np.nan) for e in peaks], x_unit),
                (f"{name} rise (start to peak)",
                 [rise_time(e) for e in peaks], x_unit),
                (f"{name} recovery (peak to end)",
                 [recovery_time(e) for e in peaks], x_unit),
                (f"{name} {w.lower()}"
                 + (" (no sign)" if side == "neg" else ""),
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
                (f"Total {w.lower()} (|positive| + |negative|)",
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
    A number as Excel writes it: 0.0523 or 544.7 as they are, and in
    scientific notation when very small or large: 5.447E-16, 2.5E+07, so
    it pastes into Excel as a number (and reads the same anywhere else).
    '' when it is not a number.
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
    return f"{mantissa}E{int(exponent):+03d}"


def mean_sd_range(values, unit=""):
    """'52.1 ± 1.2 nN, median 52 (50.3 to 54)': mean ± SD, the median,
    and the range from the smallest to the largest. One value as it is.
    Very small or large numbers as Excel writes them (5.447E-16 J)."""
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
               e_unit=None, timing=None):
    """
    One row per peak integrated (each peak in the beats), in time order:
    its sign, the beat it belongs to, where it leaves the baseline, gets to
    start, peaks, ends (by the timing rule, ``timing`` its words) and meets
    the baseline again; its duration, width at half height, rise and
    recovery; its
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
    timing = timing or timing_of(beats)
    xu = f" ({x_unit})" if x_unit else ""
    yu = f" ({y_unit})" if y_unit else ""
    area_unit = f"{y_unit}·{x_unit}" if (x_unit and y_unit) else "y·x"
    # a force against time: its area is an impulse
    area_col = (f"Impulse ∫(F − baseline) dt ({area_unit})"
                if area_word(x_unit, y_unit) == "Impulse"
                else f"∫(y − baseline) dx ({area_unit})")
    as_force = bool(newtons) and not is_force(y_unit)
    f_factor, f_unit = force_scale(
        [abs(e["peak"]) * newtons for e, _ in peaks] if as_force else [])
    energies = [spring_energy(abs(e["peak"]), k, newtons) for e, _ in peaks]
    e_factor, e_name = energy_factor(e_unit, energies)
    rows = []
    for n, ((e, b), energy) in enumerate(zip(peaks, energies), start=1):
        row = {"Peak": n, "Sign": "+" if e["sign"] > 0 else "−",
               "Beat": b["n"], "Beat kind": b["kind"],
               f"Leaves baseline{xu}": e["start"],
               f"Starts at{xu}": e.get("up", np.nan),
               f"Peaks at{xu}": e["peak_x"],
               f"Ends at{xu}": e.get("back", np.nan),
               f"Back at baseline{xu}": e["end"],
               f"Duration, {timing}{xu}": peak_duration(e),
               f"Width at half height{xu}": e.get("half_width", np.nan),
               # a full beat: its first peak's start to its second's end
               f"Full beat duration{xu}": (beat_duration(b)
                                           if b["kind"] == KINDS[0]
                                           else np.nan),
               f"Rise, start to peak{xu}": rise_time(e),
               f"Recovery, peak to end{xu}": recovery_time(e),
               # 1: falls as fast as it rises; above 1: a slow fall, a tail
               # or a shoulder
               "Recovery ÷ rise": _ratio(recovery_time(e), rise_time(e)),
               # the share of the peak unchanged by mirroring time about
               # its top (add_symmetry): 1 exactly symmetric
               "Mirror symmetry about the peak": e.get("mirror", np.nan),
               "Beat's half-turn symmetry": b.get("point", np.nan),
               f"Height{yu}": e["peak"],
               area_col: e["area"]}
        if as_force:
            row[f"Force at the peak ({f_unit})"] = (
                e["peak"] * newtons / f_factor)
            row[(f"Impulse ∫F dt ({f_unit}·{x_unit})" if seconds_per(x_unit)
                 else f"∫F dx ({f_unit}·{x_unit or 'x'})")] = (
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
    timed = [e for e in events if "up" in e]
    if timed:
        fig.add_trace(go.Scatter(
            x=[v for e in timed for v in (e["up"], e["back"])],
            y=[base + e["sign"] * e["level"] for e in timed
               for _ in (0, 1)],
            mode="markers", name="starts and ends", showlegend=False,
            hoverinfo="skip" if peak_click else None,
            hovertemplate=(None if peak_click else
                           "%{x:.4g}: a peak starts or ends<extra></extra>"),
            marker=dict(symbol="line-ns-open", size=12, color="#111111",
                        line=dict(color="#111111", width=2))))
    if left_out:
        fig.add_trace(go.Scatter(
            x=[e["peak_x"] for e in left_out],
            y=[base + e["peak"] for e in left_out], mode="markers",
            name="left out", showlegend=False,
            text=[e.get("auto_out") or "removed by hand" for e in left_out],
            hoverinfo=None if peak_click else "skip",
            hovertemplate=("%{x:.4g}: left out (%{text}), click to put "
                           "back<extra></extra>" if peak_click else None),
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
