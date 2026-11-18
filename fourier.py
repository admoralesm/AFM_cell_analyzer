"""
Fourier analysis of a signal against time, for the 〰 Fourier tab.

A recording is put on an even time grid first (an FFT needs equal steps;
an AFM trace has gaps where the scan turns round, a video may skip
frames), then:

- its spectrum: the amplitude of each frequency, in the signal's own unit
  (a sine of amplitude A shows a peak of height A), or its power spectral
  density, in unit²/Hz; optionally averaged over overlapping pieces
  (Welch) to steady a noisy one;
- its peaks: the strongest frequencies, each with its period, its rate per
  minute and which harmonic of the strongest one it is;
- a spectrogram: the spectrum of piece after piece, to see the beating
  rate change over time;
- a band-pass filter: the signal with only a band of frequencies kept.

Time in seconds gives frequencies in Hz; any other x gives cycles per
unit of x.
"""

import numpy as np

WINDOWS = ("Hann", "Hamming", "Blackman", "None (rectangular)")
DETRENDS = ("Mean", "Straight line", "None")
KINDS = ("Amplitude", "Power density")
# seconds per unit of time, so a spectrum is always in Hz
TIME_UNITS = {"s": 1.0, "sec": 1.0, "ms": 1e-3, "µs": 1e-6, "us": 1e-6,
              "μs": 1e-6, "min": 60.0, "h": 3600.0}


def seconds_per(unit):
    """How many seconds one unit of x is, or None when x is not a time."""
    return TIME_UNITS.get(str(unit or "").strip())


def to_uniform(x, y, step=None):
    """
    (xs, ys, step, filled): the signal on an evenly spaced grid, at its
    median step (or ``step``), gaps (missing points, NaN) bridged by
    straight lines. ``filled`` is the share of grid points that fell in a
    gap, i.e. more than 1.5 steps from any measured point.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    x, first = np.unique(x, return_index=True)
    y = y[first]
    if x.size < 8:
        raise ValueError("Too few points for a spectrum (8 at least).")
    steps = np.diff(x)
    step = float(step or np.median(steps[steps > 0]))
    if not step > 0:
        raise ValueError("The x values do not increase.")
    n = int(np.floor((x[-1] - x[0]) / step + 1e-9)) + 1
    xs = x[0] + step * np.arange(n)
    ys = np.interp(xs, x, y)
    at = np.clip(np.searchsorted(x, xs), 1, x.size - 1)
    near = np.minimum(np.abs(xs - x[at - 1]), np.abs(x[at] - xs))
    filled = float(np.mean(near > 1.5 * step))
    return xs, ys, step, filled


def window(name, n):
    if name == "Hann":
        return np.hanning(n)
    if name == "Hamming":
        return np.hamming(n)
    if name == "Blackman":
        return np.blackman(n)
    return np.ones(n)


def detrend(xs, ys, how="Mean"):
    """The signal less its mean, or less its straight-line fit, or as it
    is."""
    if how == "Mean":
        return ys - np.mean(ys)
    if how == "Straight line":
        slope, cut = np.polyfit(xs - xs[0], ys, 1)
        return ys - (slope * (xs - xs[0]) + cut)
    return ys


def _one_spectrum(piece, step, win, kind, nfft):
    """Single-sided amplitude (a sine of amplitude A gives A) or power
    density of one piece, windowed."""
    w = window(win, piece.size)
    spec = np.fft.rfft(piece * w, nfft)
    fs = 1.0 / step
    if kind == KINDS[1]:
        out = np.abs(spec) ** 2 / (fs * np.sum(w ** 2))
        out[1:] *= 2.0
        if nfft % 2 == 0:
            out[-1] /= 2.0
        return out
    out = 2.0 * np.abs(spec) / np.sum(w)
    out[0] /= 2.0
    if nfft % 2 == 0:
        out[-1] /= 2.0
    return out


def spectrum(x, y, kind=KINDS[0], win=WINDOWS[0], how=DETRENDS[0], pad=1,
             pieces=1, x_scale=1.0):
    """
    The spectrum of y against x: dict with ``freq`` (Hz when x is in
    seconds, x × ``x_scale`` being seconds), ``value`` (amplitude in y's
    unit, or power density in y's unit² per Hz), and how it was made:
    ``step`` (s), ``fs`` (Hz), ``n`` points, ``resolution`` (Hz, the true
    one: 1 / the length of a piece), ``nyquist`` (Hz), ``filled``.
    ``pad`` × more points (zeros) draws the spectrum finer without adding
    resolution; ``pieces`` > 1 averages that many half-overlapping pieces
    (Welch): steadier, coarser.
    """
    xs, ys, step, filled = to_uniform(np.asarray(x, float) * x_scale, y)
    n = ys.size
    pieces = max(1, int(pieces))
    seg = n if pieces == 1 else max(8, int(2 * n / (pieces + 1)))
    hop = max(1, seg // 2)
    starts = list(range(0, n - seg + 1, hop))[:pieces] or [0]
    nfft = int(2 ** np.ceil(np.log2(seg * max(1, int(pad)))))
    total = None
    for start in starts:
        piece = detrend(xs[start:start + seg], ys[start:start + seg], how)
        one = _one_spectrum(piece, step, win, kind, nfft)
        if kind == KINDS[0]:
            one = one ** 2          # amplitudes are averaged as power
        total = one if total is None else total + one
    value = total / len(starts)
    if kind == KINDS[0]:
        value = np.sqrt(value)
    freq = np.fft.rfftfreq(nfft, step)
    return {"freq": freq, "value": value, "step": step, "fs": 1.0 / step,
            "n": n, "seg": seg, "pieces": len(starts),
            "resolution": 1.0 / (seg * step), "nyquist": 0.5 / step,
            "filled": filled, "x0": xs[0], "x1": xs[-1]}


def find_peaks(freq, value, count=5, fmin=None, fmax=None, apart=None):
    """
    The strongest peaks of a spectrum (local maxima), largest first, as
    dicts with ``freq`` (refined between the bins by a parabola through
    the top three), ``value`` and ``rank``. Below ``fmin`` (by default the
    first bin past 0, so the mean is never a peak) and above ``fmax`` they
    are not looked for; peaks closer than ``apart`` (by default 3 bins) to
    a stronger one are left out.
    """
    f = np.asarray(freq, float)
    v = np.asarray(value, float)
    if f.size < 3:
        return []
    df = f[1] - f[0]
    lo = fmin if fmin is not None else df * 1.5
    hi = fmax if fmax is not None else f[-1]
    apart = apart if apart is not None else 3 * df
    inner = np.arange(1, f.size - 1)
    is_max = (v[inner] >= v[inner - 1]) & (v[inner] > v[inner + 1])
    cand = inner[is_max & (f[inner] >= lo) & (f[inner] <= hi)]
    cand = cand[np.argsort(v[cand])[::-1]]
    found = []
    for i in cand:
        a, b, c = v[i - 1], v[i], v[i + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        shift = float(np.clip(shift, -0.5, 0.5))
        fp = f[i] + shift * df
        if any(abs(fp - p["freq"]) < apart for p in found):
            continue
        found.append({"freq": float(fp),
                      "value": float(b - 0.25 * (a - c) * shift),
                      "bin": int(i)})
        if len(found) >= count:
            break
    for rank, p in enumerate(found, start=1):
        p["rank"] = rank
    return found


def harmonic_of(peaks, tolerance=0.06):
    """For each peak, which multiple it is of the strongest (fundamental)
    one: 1, 2, 3… when within ``tolerance`` of a whole number, else None.
    A peak below the strongest one is matched as a fraction (½, ⅓)."""
    if not peaks:
        return []
    base = peaks[0]["freq"]
    out = []
    for p in peaks:
        ratio = p["freq"] / base if base else np.nan
        near = round(ratio) if ratio >= 1 else 0
        if near >= 1 and abs(ratio - near) <= tolerance * near:
            out.append(f"×{near}")
        elif 0 < ratio < 1 and abs(1 / ratio - round(1 / ratio)) <= tolerance \
                * round(1 / ratio):
            out.append(f"×1/{round(1 / ratio)}")
        else:
            out.append("")
    return out


def spectrogram(x, y, piece_s, overlap=0.5, win=WINDOWS[0], how=DETRENDS[0],
                kind=KINDS[0], x_scale=1.0, pad=2, fmin=None, fmax=None):
    """
    The spectrum of piece after piece (each ``piece_s`` long, overlapping
    by ``overlap``): dict with ``times`` (the middle of each piece, in x's
    own unit), ``freq``, ``value`` (frequency × time) and ``peak``, the
    strongest frequency of each piece between ``fmin`` and ``fmax``.
    """
    xs, ys, step, _ = to_uniform(np.asarray(x, float) * x_scale, y)
    seg = max(8, int(round(piece_s / step)))
    if seg > ys.size:
        raise ValueError("The piece is longer than the signal.")
    hop = max(1, int(round(seg * (1 - overlap))))
    nfft = int(2 ** np.ceil(np.log2(seg * max(1, int(pad)))))
    freq = np.fft.rfftfreq(nfft, step)
    use = np.ones(freq.size, bool)
    use[0] = False
    if fmin is not None:
        use &= freq >= fmin
    if fmax is not None:
        use &= freq <= fmax
    times, columns, tops = [], [], []
    for start in range(0, ys.size - seg + 1, hop):
        piece = detrend(xs[start:start + seg], ys[start:start + seg], how)
        one = _one_spectrum(piece, step, win, kind, nfft)
        columns.append(one)
        times.append((xs[start] + xs[start + seg - 1]) / 2 / x_scale)
        tops.append(float(freq[use][np.argmax(one[use])]) if use.any()
                    else np.nan)
    return {"times": np.asarray(times), "freq": freq,
            "value": np.asarray(columns).T, "peak": np.asarray(tops),
            "seg_s": seg * step}


def bandpass(x, y, low=None, high=None, x_scale=1.0):
    """
    (xs, filtered) with only the frequencies from ``low`` to ``high`` (Hz)
    kept, by zeroing the rest of its spectrum (no phase shift). With no
    ``low`` (or 0) the mean is kept too. On the even grid of to_uniform,
    xs in x's own unit.
    """
    xs, ys, step, _ = to_uniform(np.asarray(x, float) * x_scale, y)
    spec = np.fft.rfft(ys)
    freq = np.fft.rfftfreq(ys.size, step)
    keep = np.ones(freq.size, bool)
    if low:
        keep &= freq >= low
    if high:
        keep &= freq <= high
    if not low:
        keep[0] = True
    return xs / x_scale, np.fft.irfft(spec * keep, ys.size)
