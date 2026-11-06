"""
Checks for force_curve.py, the Single force curve engine.

Each check writes a force recording the way an Asylum instrument does
(channels Raw, Defl and ZSnsr, time as the wave's x scaling, the spring
constant and segment boundaries in the note), or as a plain CSV, and asks
the engine to read it, turn deflection into force and find where the
curve ends against the baseline after it.

    python tests_force_curve.py
"""
import os
import pathlib
import struct
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import force_curve as fc  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name
          + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def write_ibw5(path, arr, name="wave0", note="", labels=None, dx=1.0,
               x_unit=""):
    """A minimal Igor Binary Wave, version 5, little endian, with the first
    axis scaled by ``dx`` in ``x_unit``."""
    arr = np.asarray(arr, dtype=np.float32)
    data = arr.tobytes(order="F")
    dims = list(arr.shape) + [0] * (4 - arr.ndim)
    note_b = note.encode("latin-1")
    blocks, sizes = [], [0, 0, 0, 0]
    for d, names in (labels or {}).items():
        block = b"".join(n.encode()[:31].ljust(32, b"\0")
                         for n in [""] + list(names))
        blocks.append((d, block))
        sizes[d] = len(block)
    wave = bytearray(320)
    struct.pack_into("<l", wave, 12, int(arr.size))
    struct.pack_into("<h", wave, 16, 2)                 # NT_FP32
    struct.pack_into("<h", wave, 26, 1)
    wave[28:60] = name.encode()[:31].ljust(32, b"\0")
    struct.pack_into("<4l", wave, 68, *dims)
    struct.pack_into("<4d", wave, 84, dx, 1, 1, 1)      # sfA
    struct.pack_into("<4d", wave, 116, 0, 0, 0, 0)      # sfB
    wave[152:156] = x_unit.encode()[:4].ljust(4, b"\0")  # dimUnits[0]
    head = bytearray(64)
    struct.pack_into("<h", head, 0, 5)
    struct.pack_into("<l", head, 4, 320 + len(data))
    struct.pack_into("<l", head, 12, len(note_b))
    struct.pack_into("<4l", head, 36, *sizes)
    with open(path, "wb") as fh:
        fh.write(head + wave + data + note_b)
        for d in range(4):
            for dd, block in blocks:
                if dd == d:
                    fh.write(block)


def compression(rate=2000.0, seed=0):
    """A cell squeezed and let go: flat, a rise to 5 nN, relaxation while
    held, the probe pulled off with a small adhesion dip, flat again. The
    curve is over at 3.40 s."""
    t = np.arange(0, 5.0, 1.0 / rate)
    f = np.zeros_like(t)
    rise = (t >= 1.0) & (t < 1.5)
    f[rise] = 5.0 * (t[rise] - 1.0) / 0.5
    hold = (t >= 1.5) & (t < 3.0)
    f[hold] = 3.0 + 2.0 * np.exp(-(t[hold] - 1.5) / 0.4)
    off = (t >= 3.0) & (t < 3.2)
    f[off] = 3.0 * (1 - (t[off] - 3.0) / 0.2) - 0.6 * (t[off] - 3.0) / 0.2
    back = (t >= 3.2) & (t < 3.4)
    f[back] = -0.6 * (1 - (t[back] - 3.2) / 0.2)
    f += np.random.default_rng(seed).normal(0, 0.02, t.size)
    return t, f


def main():
    folder = tempfile.mkdtemp()
    k = 0.08                                   # N/m
    rate = 2000.0
    t, force = compression(rate)
    defl_m = force * 1e-9 / k                  # nN -> m of deflection
    z = np.linspace(0, 3e-6, t.size)
    arr = np.column_stack([z * 0.98, defl_m, z])
    note = (f"SpringConstant: {k}\rInvOLS: 4.2e-08\r"
            f"NumPtsPerSec: {rate:g}\rIndexes: 0,{int(1.5 * rate)},"
            f"{int(3.0 * rate)},{t.size - 1}\r")

    print("an Asylum force file with its time scaling")
    path = os.path.join(folder, "cell01_force.ibw")
    write_ibw5(path, arr, "cell01_force", note,
               {1: ["Raw", "Defl", "ZSnsr"]}, dx=1.0 / rate, x_unit="s")
    item = fc.read_force_file(open(path, "rb").read(), "cell01_force.ibw")
    check("reads the three channels by name",
          list(item["channels"]) == ["Raw", "Defl", "ZSnsr"],
          str(list(item["channels"])))
    check("takes the time step from the wave", abs(item["dt"] - 1 / rate)
          < 1e-12 and "wave" in item["time_from"], str(item["time_from"]))
    check("finds k, InvOLS (in nm/V) and the segment boundaries",
          abs(item["k"] - k) < 1e-12 and abs(item["invols"] - 42) < 1e-6
          and item["segments"] == [3000, 6000], f"{item['k']} "
          f"{item['invols']} {item['segments']}")
    defl = item["channels"]["Defl"]
    check("the deflection is seen to be in metres",
          fc.deflection_unit(defl) == "m")
    got = fc.to_nanonewton(defl, "m", item["k"])
    check("deflection × k gives the force in nN",
          np.allclose(got, force, atol=1e-4), f"{np.abs(got - force).max()}")

    print("the same file with no time unit, the rate only in the note")
    path2 = os.path.join(folder, "plain.ibw")
    write_ibw5(path2, arr, "plain", note, {1: ["Raw", "Defl", "ZSnsr"]})
    item2 = fc.read_force_file(open(path2, "rb").read(), "plain.ibw")
    check("takes the time step from NumPtsPerSec",
          abs(item2["dt"] - 1 / rate) < 1e-12
          and "NumPtsPerSec" in item2["time_from"], str(item2["time_from"]))
    path3 = os.path.join(folder, "bare.ibw")
    write_ibw5(path3, arr, "bare", "", {1: ["Raw", "Defl", "ZSnsr"]})
    item3 = fc.read_force_file(open(path3, "rb").read(), "bare.ibw")
    check("and says so when nothing gives the time step",
          item3["dt"] is None and item3["k"] is None)

    print("where the curve ends")
    window = fc.default_window(t)
    check("the default baseline window is the last 15 %",
          abs(window[0] - (t[-1] - 0.15 * t[-1])) < 1e-9
          and window[1] == t[-1], str(window))
    after = fc.baseline_after(t, force, window)
    check("the end is found where it really is (3.40 s), adhesion dip "
          "included", after["end"] is not None
          and abs(after["end"] - 3.40) < 0.02, str(after["end"]))
    check("the baseline after it is at 0 with the noise as its SD",
          abs(after["level"]) < 0.01 and 0.01 < after["sd"] < 0.03,
          f"{after['level']} {after['sd']}")
    up = fc.baseline_after(t, force, window, direction="up")
    check("counting only rises above the baseline, it ends as the force "
          "drops (3.17 s)", up["end"] is not None
          and abs(up["end"] - 3.17) < 0.03, str(up["end"]))
    tilted = force + 0.3 * t
    sloped = fc.baseline_after(t, tilted, window, order=1)
    check("a sloped baseline follows a drift", sloped["end"] is not None
          and abs(sloped["end"] - 3.40) < 0.03, str(sloped["end"]))
    before = fc.baseline_before(t, force, fc.default_window(t, after=False))
    check("and the start is found (1.00 s)", before["start"] is not None
          and abs(before["start"] - 1.0) < 0.02, str(before["start"]))
    lines, nums = fc.curve_stats(t, force, after, before)
    check("the summary has the end, peak, duration and area",
          abs(nums["peak"] - 5.0) < 0.1 and abs(nums["peak_t"] - 1.5) < 0.01
          and abs(nums["duration_s"] - 2.40) < 0.04
          and nums["area"] > 0
          and [lab for lab, _ in lines][:3] == [
              "Curve ends at", "Baseline after the end", "Peak force"],
          f"{nums}")
    table = fc.table_to_send(t, force, after, zero=after["end"])
    check("the table sent on counts time from the end and adds the force "
          "less the baseline",
          list(table.columns) == [fc.TIME, fc.FORCE, fc.FORCE_LESS_BASE]
          and abs(np.interp(0.0, table[fc.TIME], table[fc.TIME])) < 1e-9
          and table[fc.TIME].iloc[-1] > 0, str(list(table.columns)))

    print("a CSV export")
    csv = os.path.join(folder, "curve.csv")
    import pandas as pd
    pd.DataFrame({"Time (s)": t, "Force (nN)": force}).to_csv(csv,
                                                              index=False)
    item4 = fc.read_force_file(open(csv, "rb").read(), "curve.csv")
    cols = list(item4["frame"].columns)
    check("reads the table and picks time and force",
          fc.guess_time(cols) == "Time (s)"
          and fc.guess_force(cols, "Time (s)") == "Force (nN)", str(cols))
    check("a table with no time column is said to have none",
          fc.guess_time(["Defl (nm)", "Z (um)"]) is None)

    print("\nall passing" if not FAILED else f"\n{len(FAILED)} failing")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
