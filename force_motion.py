"""
Force and motion: an AFM force trace and a video length-change trace of the
same cell, recorded at the same time, put on one clock and compared beat by
beat.

The two recordings start at different moments, so the first job is the
offset between their clocks: the shift that best lines the force up with
the length change (a cross-correlation over every shift, or near one that
is typed in). Then, for each beat found in the force, the length change
in the same stretch of time: how far the cell shortened, when, how long
after the force, the force per % of shortening, and the loop that force
against shortening draws through the beat.

Times are in seconds. Force in whatever unit it comes in (µN), length
change in % (or µm). Pure numpy, so it can be tested on its own:
python tests_force_motion.py
"""

import numpy as np

SIGNS = ("Auto", "Length decreases", "Length increases")


def clean(t, y):
    """The finite points, sorted by time, with repeated times dropped."""
    t = np.asarray(t, float).ravel()
    y = np.asarray(y, float).ravel()
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    if t.size:
        keep = np.r_[True, np.diff(t) > 0]
        t, y = t[keep], y[keep]
    return t, y


def step_of(t):
    """The typical time between points (the median step)."""
    t = np.asarray(t, float)
    return float(np.median(np.diff(t))) if t.size > 1 else float("nan")


def best_offset(tf, f, tm, m, dt=None, centre=None, window=None,
                min_overlap=0.3):
    """
    The shift of the motion's clock onto the force's: a frame at motion
    time t happened at force time t + offset. Found as the shift where
    the force and the length change are most alike (the largest
    correlation, of either sign, of the two resampled on one grid of step
    ``dt``), over every shift where at least ``min_overlap`` of the shorter
    recording overlaps the other, or only within ``window`` s of
    ``centre`` when those are given.

    Returns offset (s), r (the correlation there; negative when the length
    falls as the force rises), sign (+1/−1), overlap (s), and lags / rs:
    the correlation at every shift tried.
    """
    tf, f = clean(tf, f)
    tm, m = clean(tm, m)
    if tf.size < 4 or tm.size < 4:
        return None
    if dt is None:
        dt = max(min(step_of(tm), step_of(tf) * 4) / 2.0, 1e-4)
    gf = np.arange(tf[0], tf[-1], dt)
    gm = np.arange(tm[0], tm[-1], dt)
    if gf.size < 4 or gm.size < 4:
        return None
    a = np.interp(gf, tf, f)
    b = np.interp(gm, tm, m)
    a = (a - a.mean()) / (a.std() or 1.0)
    b = (b - b.mean()) / (b.std() or 1.0)
    n = gf.size + gm.size - 1
    size = 1 << int(np.ceil(np.log2(n)))
    # c[k]: sum of a[i] b[i − k'] for every shift; by FFT
    full = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)),
                        size)
    # shifts k from −(len b − 1) to len a − 1: a[i] lines up with b[i − k]
    c = np.r_[full[-(gm.size - 1):], full[:gf.size]]
    ks = np.arange(-(gm.size - 1), gf.size)
    # the points that overlap at each shift, to make a mean of the products
    count = np.minimum(gf.size, ks + gm.size) - np.maximum(0, ks)
    rs = np.where(count > 0, c / np.maximum(count, 1), 0.0)
    # motion grid point j is at gm[0] + j·dt; it lines up with force grid
    # point j + k, at gf[0] + (j + k)·dt: offset = gf[0] − gm[0] + k·dt
    lags = gf[0] - gm[0] + ks * dt
    ok = count >= min_overlap * min(gf.size, gm.size)
    if centre is not None and window is not None:
        ok &= np.abs(lags - float(centre)) <= float(window)
    if not ok.any():
        return None
    i = int(np.flatnonzero(ok)[np.argmax(np.abs(rs[ok]))])
    return {"offset": float(lags[i]), "r": float(rs[i]),
            "sign": 1 if rs[i] >= 0 else -1,
            "overlap": float(count[i] * dt),
            "lags": lags[ok], "rs": rs[ok]}


def crossing(t, y, level, after=None, before=None, rising=True):
    """The first time ``y`` passes ``level`` going up (or down), between
    ``after`` and ``before``, interpolated between the two points either
    side. None if it does not."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    sel = np.ones(t.size, bool)
    if after is not None:
        sel &= t >= after
    if before is not None:
        sel &= t <= before
    idx = np.flatnonzero(sel)
    for i, j in zip(idx[:-1], idx[1:]):
        a, b = y[i] - level, y[j] - level
        if (a < 0 <= b) if rising else (a > 0 >= b):
            return float(t[i] + (t[j] - t[i]) * (-a) / (b - a))
    return None


def loop_area(x, y):
    """The area a closed loop of points encloses (the shoelace formula),
    whatever way round it goes."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan")
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
                 / 2.0)


def shortening(tm, m, sign):
    """The length change turned into shortening (larger when the cell is
    shorter): −m when a contraction makes the length fall (sign −1), m when
    it makes it rise."""
    return -np.asarray(m, float) if sign < 0 else np.asarray(m, float)


def beat_motion(beats, tf, f, tm, m, offset, sign, pad=0.3, before=1.0):
    """
    Each force beat with the cell's shortening in the same stretch of time.

    ``beats``: dicts with t0 and t1 (where the beat starts and ends, s on
    the force clock), t_peak (the force peak), t_half (the force at half
    its height on the rise) and any values to carry along. ``tf``/``f``:
    the force against time, its baseline taken away. ``tm``/``m``: the
    length change on the motion clock; ``offset`` puts it on the force
    clock; ``sign`` is −1 when a contraction makes the length fall.

    For each beat, in [t0 − pad, t1 + pad]: the shortening (its largest
    value less the median over ``before`` s before the beat), when it is
    largest, when it is half way there, the delays after the force (peak
    to peak, half to half), the force per % of shortening, and the loop
    (force against shortening at each frame) with its area. Beats with no
    frame in their stretch get NaN.
    """
    tf, f = clean(tf, f)
    tm, m = clean(tm, m)
    tmo = tm + float(offset)
    s = shortening(tmo, m, sign)
    out = []
    for b in beats:
        row = dict(b)
        a, z = float(b["t0"]) - pad, float(b["t1"]) + pad
        inside = (tmo >= a) & (tmo <= z)
        prior = (tmo >= float(b["t0"]) - before) & (tmo < float(b["t0"]))
        nan = float("nan")
        row.update(shortening=nan, t_short=nan, t_short_half=nan,
                   delay_peak=nan, delay_half=nan, force_per_short=nan,
                   loop_area=nan, loop_x=np.array([]), loop_y=np.array([]),
                   frames=int(inside.sum()))
        if inside.sum() >= 2:
            rest = float(np.median(s[prior])) if prior.sum() >= 2 \
                else float(np.median(s[inside]))
            ts, ss = tmo[inside], s[inside] - rest
            k = int(np.argmax(ss))
            amp = float(ss[k])
            row["shortening"] = amp
            row["t_short"] = float(ts[k])
            if amp > 0:
                half = crossing(ts, ss, amp / 2.0, before=ts[k])
                row["t_short_half"] = half if half is not None else nan
                if b.get("force") is not None and np.isfinite(b["force"]):
                    row["force_per_short"] = float(b["force"]) / amp
            # (rounded to 1 ns: the clocks' sums leave dust like 6E-12;
            # + 0.0 makes −0 plain 0)
            if b.get("t_peak") is not None:
                row["delay_peak"] = round(
                    row["t_short"] - float(b["t_peak"]), 9) + 0.0
            if b.get("t_half") is not None and np.isfinite(row["t_short_half"]):
                row["delay_half"] = round(
                    row["t_short_half"] - float(b["t_half"]), 9) + 0.0
            fx = np.interp(ts, tf, f, left=nan, right=nan)
            row["loop_x"], row["loop_y"] = ss, fx
            row["loop_area"] = loop_area(ss, fx)
        out.append(row)
    return out


def correlation(x, y):
    """Pearson's r of the finite pairs and how many there were; r is NaN
    with fewer than three or no spread."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), int(ok.sum())
