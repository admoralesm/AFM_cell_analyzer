"""Tests for fourier.py: known sines in, their frequencies and amplitudes
out. Run: python tests_fourier.py"""

import sys

import numpy as np

import fourier as ft

FAILURES = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name
          + ("" if cond else f"  [{detail}]"))
    if not cond:
        FAILURES.append(name)


# 0.25 Hz beats (15 per minute) of amplitude 0.05, a 0.75 Hz overtone of
# 0.02, an offset and a little noise, 120 s at 50 Hz
t = np.arange(0, 120, 0.02)
rng = np.random.default_rng(3)
y = (0.087 + 0.05 * np.sin(2 * np.pi * 0.25 * t)
     + 0.02 * np.sin(2 * np.pi * 0.75 * t + 1.0)
     + rng.normal(0, 0.002, t.size))

print("even grid")
xs, ys, step, filled = ft.to_uniform(t, y)
check("an even signal stays as it is", abs(step - 0.02) < 1e-12
      and xs.size == t.size and filled == 0.0, (step, xs.size, filled))
gappy = y.copy()
gappy[1000:1100] = np.nan
xs, ys, step, filled = ft.to_uniform(t, gappy)
check("a gap is bridged and counted", xs.size == t.size
      and 0.015 < filled < 0.02 and np.all(np.isfinite(ys)), filled)
try:
    ft.to_uniform([0, 1, 2], [1, 2, 3]); check("too short is refused", False)
except ValueError:
    check("too short is refused", True)

print("amplitude spectrum")
sp = ft.spectrum(t, y)
peaks = ft.find_peaks(sp["freq"], sp["value"], count=4)
check("the strongest frequency is the beat, 0.25 Hz",
      abs(peaks[0]["freq"] - 0.25) < 0.003, peaks[0])
check("its height is the sine's amplitude (Hann window)",
      abs(peaks[0]["value"] - 0.05) < 0.003, peaks[0]["value"])
check("the overtone next, at 0.75 Hz, amplitude 0.02",
      abs(peaks[1]["freq"] - 0.75) < 0.003
      and abs(peaks[1]["value"] - 0.02) < 0.002, peaks[1])
check("the mean is not a peak", all(p["freq"] > 0.05 for p in peaks))
check("named as harmonics of the strongest",
      ft.harmonic_of(peaks)[:2] == ["×1", "×3"], ft.harmonic_of(peaks))
check("resolution and Nyquist", abs(sp["resolution"] - 1 / 120) < 1e-3
      and abs(sp["nyquist"] - 25) < 1e-9, (sp["resolution"], sp["nyquist"]))
flat = ft.spectrum(t, y, win=ft.WINDOWS[-1])
check("no window: the same amplitude for a whole number of cycles",
      abs(ft.find_peaks(flat["freq"], flat["value"])[0]["value"] - 0.05)
      < 0.003)
ms = ft.spectrum(t * 1000, y, x_scale=1e-3)
check("x in ms, the frequency still in Hz",
      abs(ft.find_peaks(ms["freq"], ms["value"])[0]["freq"] - 0.25) < 0.003)

print("power density and averaging")
psd = ft.spectrum(t, y, kind=ft.KINDS[1])
df = psd["freq"][1] - psd["freq"][0]
band = (psd["freq"] > 0.2) & (psd["freq"] < 0.3)
check("the power in the beat's band is A²/2",
      abs(np.sum(psd["value"][band]) * df - 0.05 ** 2 / 2) < 0.1 * 0.05 ** 2 / 2,
      np.sum(psd["value"][band]) * df)
welch = ft.spectrum(t, y, pieces=4)
check("four pieces: coarser, the beat still found",
      welch["pieces"] == 4 and welch["resolution"] > sp["resolution"]
      and abs(ft.find_peaks(welch["freq"], welch["value"])[0]["freq"] - 0.25)
      < 0.02, (welch["pieces"], welch["resolution"]))
padded = ft.spectrum(t, y, pad=4)
check("padding draws it finer, the same resolution",
      padded["freq"].size > sp["freq"].size
      and padded["resolution"] == sp["resolution"])

print("spectrogram")
# the rate goes from 0.25 Hz to 0.5 Hz half way
chirp = np.where(t < 60, np.sin(2 * np.pi * 0.25 * t),
                 np.sin(2 * np.pi * 0.5 * t))
sg = ft.spectrogram(t, chirp, piece_s=20, overlap=0.5, fmin=0.1, fmax=2)
check("the strongest frequency follows the rate",
      abs(sg["peak"][0] - 0.25) < 0.03 and abs(sg["peak"][-1] - 0.5) < 0.03,
      sg["peak"])
check("one column per piece", sg["value"].shape == (sg["freq"].size,
                                                    sg["times"].size))

print("band-pass")
xs, kept = ft.bandpass(t, y, low=0.5, high=1.0)
check("only the overtone is left",
      abs(np.std(kept) - 0.02 / np.sqrt(2)) < 0.002 and abs(np.mean(kept))
      < 1e-3, np.std(kept))
xs, low = ft.bandpass(t, y, high=0.4)
check("a low-pass keeps the mean and the beat",
      abs(np.mean(low) - 0.087) < 1e-3
      and abs(np.std(low) - 0.05 / np.sqrt(2)) < 0.002)

print()
if FAILURES:
    print(f"{len(FAILURES)} failing: {FAILURES}")
    sys.exit(1)
print("all passing")
