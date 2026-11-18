"""Tests for beat_analysis.py: python tests_beat_analysis.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beat_analysis as ba  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else f"  {detail}"))
    if not ok:
        FAILED.append(name)


def tri(t, at, half, height):
    """A triangle on t: 0 outside [at − half, at + half]."""
    return np.clip(1 - np.abs(t - at) / half, 0, None) * height


t = np.arange(0, 100, 0.001)
base = 0.02
y = np.full(t.size, base)
# three full beats: a positive peak, then a negative one right after
for at in (10.0, 30.0, 50.0):
    y += tri(t, at, 0.5, 0.10) - tri(t, at + 1.0, 0.4, 0.03)
y += tri(t, 70.0, 0.5, 0.20)            # positive only
y -= tri(t, 85.0, 0.4, 0.05)            # negative only
y += tri(t, 92.0, 0.05, 0.002)          # too small: under the line

print("units")
check("the unit at the end of a name", ba.unit_of("Force (µN)") == "µN"
      and ba.unit_of("Plot 1 (raw)") == "raw" and ba.unit_of("length") == "")
check("newtons per unit", ba.newtons_per_unit("µN") == 1e-6
      and ba.newtons_per_unit("nN") == 1e-9
      and ba.newtons_per_unit("raw") is None)
check("energies shown near 1", ba.energy_scale([3e-15, 5e-15]) == (1e-15, "fJ"))

print("baseline and lines")
check("the median finds the resting level",
      abs(ba.baseline_level(y) - base) < 1e-6, ba.baseline_level(y))
check("so does the most common level",
      abs(ba.baseline_level(y, ba.BASELINES[1]) - base) < 1e-3,
      ba.baseline_level(y, ba.BASELINES[1]))
rng = np.random.default_rng(0)
noisy = y + rng.normal(0, 0.001, y.size)
up, down = ba.suggest_thresholds(noisy, base)
check("lines a few noise SDs from the baseline",
      0.004 < up < 0.006 and 0.004 < down < 0.006, (up, down))

print("excursions")
events = ba.excursions(t, y, base, 0.01, 0.01)
signs = [e["sign"] for e in events]
check("every peak past a line, in time order",
      signs == [1, -1, 1, -1, 1, -1, 1, -1], signs)
first = events[0]
check("from where it leaves the baseline to where it comes back",
      abs(first["start"] - 9.5) < 2e-3 and abs(first["end"] - 10.5) < 2e-3,
      (first["start"], first["end"]))
check("its area is that of the triangle (½ × 1 s × 0.1)",
      abs(first["area"] - 0.05) < 1e-4, first["area"])
check("a negative one has a negative area and depth",
      abs(events[1]["area"] + 0.5 * 0.8 * 0.03) < 1e-4
      and abs(events[1]["peak"] + 0.03) < 1e-6, events[1])
check("the peak is where it is highest",
      abs(first["peak_x"] - 10.0) < 2e-3 and abs(first["peak"] - 0.1) < 1e-6)
check("a bump under the line is not an excursion",
      not any(abs(e["peak_x"] - 92.0) < 0.2 for e in events))
wide = ba.excursions(t, y, base, 0.01, 0.01, min_width=0.9)
check("too narrow ones are dropped",
      [e["sign"] for e in wide] == [1, 1, 1, 1], [e["sign"] for e in wide])
twin = np.full(t.size, base) + tri(t, 10.0, 0.3, 0.1) + tri(t, 10.8, 0.3, 0.1)
apart = ba.excursions(t, twin, base, 0.01, 0.01)
together = ba.excursions(t, twin, base, 0.01, 0.01, merge=0.5)
check("two close peaks are one when merged, their areas added",
      len(apart) == 2 and len(together) == 1
      and abs(together[0]["area"] - 2 * 0.03) < 1e-4, (len(apart), together))

print("beats")
beats = ba.pair_beats(events, max_gap=1.0)
kinds = [b["kind"] for b in beats]
check("a positive then a negative is one full beat; the rest alone",
      kinds == ["Full beat"] * 3 + ["Positive only", "Negative only"], kinds)
check("numbered in time", [b["n"] for b in beats] == [1, 2, 3, 4, 5])
check("too far apart, they stay apart",
      [b["kind"] for b in ba.pair_beats(events, max_gap=0.01)][:2]
      == ["Positive only", "Negative only"])
check("negative then positive pairs the other way",
      [b["kind"] for b in ba.pair_beats(events, 1.0, ba.ORDERS[1])][:2]
      == ["Positive only", "Negative only"])

print("table and summary")
table = ba.beat_table(beats, "s", "µN", k=0.1, newtons=1e-6)
check("one row per beat", len(table) == 5, len(table))
check("areas with their unit", "Positive area (µN·s)" in table.columns
      and abs(table["Positive area (µN·s)"][0] - 0.05) < 1e-4, list(table.columns))
check("total area of a full beat is both peaks' areas",
      abs(table["Total area (µN·s)"][0] - (0.05 + 0.012)) < 2e-4,
      table["Total area (µN·s)"][0])
energy_cols = [c for c in table.columns if c.startswith("Spring energy, beat")]
check("spring energy F²/2k per beat", len(energy_cols) == 1)
unit = energy_cols[0].split("(")[1].rstrip(")")
factor = dict((u, f) for f, u in ba.ENERGY_PREFIXES)[unit]
expected = ((0.1e-6) ** 2 + (0.03e-6) ** 2) / (2 * 0.1)
check("…and its value", abs(table[energy_cols[0]][0] * factor - expected)
      < 1e-3 * expected, (table[energy_cols[0]][0] * factor, expected))
check("no k, no spring energy",
      not any("Spring" in c for c in ba.beat_table(beats, "s", "µN").columns))
check("time since the previous beat",
      abs(table["Since previous beat (s)"][1] - 20.0) < 2e-3)
info = ba.summary(beats, "s", "µN", k=0.1, newtons=1e-6, per_minute=60)
check("counts of each kind", (info["Full beats"], info["Positive only"],
                              info["Negative only"]) == (3, 1, 1), info)
check("total positive area", abs(info["Positive area, total"][0]
                                 - (3 * 0.05 + 0.1)) < 5e-4,
      info["Positive area, total"])
check("a rate in beats per minute", "Beat rate (per minute)" in info)
check("total spring energy", info["Spring energy, total"][0] > 0
      and "Spring energy per full beat" in info)

print("units: a deflection made a force, a fixed energy unit")
check("a deflection in nm is k × 1e-9 N per nm",
      abs(ba.newtons_per_unit("nm", k=0.05) - 0.05e-9) < 1e-24)
check("volts go through InvOLS (nm/V) first",
      abs(ba.newtons_per_unit("V", k=0.05, invols=40.0)
          - 0.05 * 40e-9) < 1e-24)
check("without k a deflection is not a force",
      ba.newtons_per_unit("nm") is None
      and ba.newtons_per_unit("V", k=0.05) is None)
check("an energy unit asked for is kept", ba.energy_factor("aJ", [3e-15])
      == (1e-18, "aJ") and ba.energy_factor(None, [3e-15]) == (1e-15, "fJ"))
fixed = ba.summary(beats, "s", "µN", k=0.1, newtons=1e-6, e_unit="aJ")
check("the summary in the unit asked for",
      fixed["Spring energy, total"][1] == "aJ"
      and abs(fixed["Spring energy, total"][0] * 1e-18
              - info["Spring energy, total"][0]
              * dict((u, f) for f, u in ba.ENERGY_PREFIXES)[
                  info["Spring energy, total"][1]]) < 1e-24)
check("positive and negative energies add up to the total",
      abs(fixed["Spring energy, positive peaks"][0]
          + fixed["Spring energy, negative peaks"][0]
          - fixed["Spring energy, total"][0]) < 1e-9
      * fixed["Spring energy, total"][0])
check("net area is positive plus negative",
      abs(fixed["Net area, total"][0] - fixed["Positive area, total"][0]
          - fixed["Negative area, total"][0]) < 1e-12)

print("integration results, one row per peak")
peaks = ba.peak_table(beats, "s", "µN", k=0.1, newtons=1e-6, e_unit="fJ")
check("every peak in the beats, in time order", len(peaks) == 8
      and list(peaks["Sign"]) == ["+", "−"] * 3 + ["+", "−"]
      and list(peaks["Beat"]) == [1, 1, 2, 2, 3, 3, 4, 5], peaks)
check("its integral with its unit",
      abs(peaks["∫(y − baseline) dx (µN·s)"][0] - 0.05) < 1e-4,
      list(peaks.columns))
check("its energy F²/2k in the unit asked for",
      abs(peaks["Energy F²/2k (fJ)"][0] - (0.1e-6) ** 2 / 0.2 / 1e-15)
      < 1e-3 * (0.1e-6) ** 2 / 0.2 / 1e-15, peaks["Energy F²/2k (fJ)"][0])
check("a force has no second force column",
      not any(c.startswith("Force at") for c in peaks.columns))
defl = ba.peak_table(beats, "s", "nm", k=0.05, e_unit="aJ",
                     newtons=ba.newtons_per_unit("nm", k=0.05))
check("a deflection gets its force and ∫F dx",
      "Force at the peak (pN)" in defl.columns
      and abs(defl["Force at the peak (pN)"][0] - 0.1 * 0.05 * 1e3) < 1e-9
      and abs(defl["∫F dx (pN·s)"][0] - 0.05 * 0.05 * 1e3) < 1e-5,
      list(defl.columns))
check("…and the energy k·δ²/2",
      abs(defl["Energy F²/2k (aJ)"][0] * 1e-18 - 0.05 * (0.1e-9) ** 2 / 2)
      < 1e-24, defl["Energy F²/2k (aJ)"][0])
check("no k, no energy column",
      not any("Energy" in c for c in ba.peak_table(beats, "s", "µN").columns))

print("beats by kind, every parameter")
kinds = ba.kind_stats(beats, "s", "µN", k=0.1, newtons=1e-6, e_unit="fJ")
by = {group["kind"]: group for group in kinds}
check("three kinds, counted", [g["kind"] for g in kinds] == list(ba.KINDS)
      and [by[kd]["n"] for kd in ba.KINDS] == [3, 1, 1],
      [(g["kind"], g["n"]) for g in kinds])
full = {n: (v, u) for n, v, u in by[ba.KINDS[0]]["params"]}
check("a full beat's peaks: height, width, rise, area, energy, in units",
      abs(np.mean(full["Positive peak height"][0]) - 0.1) < 1e-3
      and full["Positive peak height"][1] == "µN"
      and abs(np.mean(full["Positive peak width"][0]) - 1.0) < 0.02
      and abs(np.mean(full["Positive peak rise (start to peak)"][0]) - 0.5)
      < 0.02 and full["Positive peak area"][1] == "µN·s"
      and full["Positive peak energy"][1] == "fJ", sorted(full))
check("…and the beat's own: duration, time between its peaks, energy",
      abs(np.mean(full["Time between its two peaks"][0]) - 1.0) < 0.02
      and abs(np.mean(full["Energy of the beat"][0])
              - np.mean(full["Positive peak energy"][0])
              - np.mean(full["Negative peak energy"][0])) < 1e-9,
      full["Time between its two peaks"][0])
alone = [n for n, _, _ in by[ba.KINDS[1]]["params"]]
check("a positive-only beat has only its peak's parameters",
      alone and all(n.startswith("Positive peak") for n in alone), alone)
check("mean ± SD (range), or the one value",
      ba.mean_sd_range([1.0, 2.0, 3.0], "nN") == "2 ± 1 nN (1 to 3)"
      and ba.mean_sd_range([5.0], "s") == "5 s"
      and ba.mean_sd_range([]) == "–")

print("plot")
fig = ba.beat_figure(t, y, base, 0.01, 0.01, events, beats,
                     left_out=[events[0]], shade_beats=True)
names = [tr.name for tr in fig.data]
check("areas, signal, lines, peaks, numbers and what is left out",
      all(n in names for n in ("positive areas", "negative areas", "signal",
                               "baseline", "upper line", "lower line",
                               "positive peaks", "negative peaks",
                               "beat numbers", "left out")), names)

print()
print("all passing" if not FAILED else f"{len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
