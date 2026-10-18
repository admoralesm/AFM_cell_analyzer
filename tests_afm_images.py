"""
Checks for afm_images.py, the AFM image engine.

Each check writes a small Igor .ibw image the way an Asylum instrument does
(4 channels with their names stored in the wave, a scan note with the line
rate and the cantilever calibration) and asks the engine to read it back,
build the signal against time from its scan lines and process it.

    python tests_afm_images.py
"""
import io
import os
import pathlib
import struct
import sys
import tempfile
import zipfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import afm_images as ai  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name
          + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def write_ibw5(path, arr, name="wave0", note="", labels=None):
    """A minimal Igor Binary Wave, version 5, little endian."""
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
    struct.pack_into("<4d", wave, 84, 1, 1, 1, 1)
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


def asylum_scan(ny=96, nx=128, line_hz=2.0, beat_hz=0.7, seed=0):
    """Height/Deflection trace and retrace of a beating cell, shaped
    (points, lines, channels) like an Asylum file. Heights in metres."""
    rng = np.random.default_rng(seed)
    xx = np.linspace(-1, 1, nx)
    out = np.zeros((nx, ny, 4))
    T = 1.0 / line_hz
    bump = 400e-9 * np.exp(-(xx / 0.45) ** 2)
    for j in range(ny):
        t_tr = j * T + np.arange(nx) * (T / 3) / nx
        t_re = j * T + T / 3 + np.arange(nx)[::-1] * (T / 3) / nx
        beat = lambda t: np.maximum(0, np.sin(2 * np.pi * beat_hz * t)) ** 4  # noqa
        out[:, j, 0] = bump + rng.normal(0, 2e-9, nx)
        out[:, j, 1] = bump + rng.normal(0, 2e-9, nx)
        out[:, j, 2] = 0.5 + 0.3 * beat(t_tr) + rng.normal(0, 0.002, nx)
        out[:, j, 3] = 0.5 + 0.3 * beat(t_re) + rng.normal(0, 0.002, nx)
    note = (f"ImagingMode: Contact\rScanSize: 2e-05\rScanRate: {line_hz:g}\r"
            f"ScanPoints: {nx}\rScanLines: {ny}\rSpringConstant: 0.08\r"
            "InvOLS: 6.5e-08\rDeflectionSetpointVolts: 0.5\r")
    labels = {2: ["HeightTrace", "HeightRetrace", "DeflectionTrace",
                  "DeflectionRetrace"]}
    return out, note, labels


def main():
    folder = tempfile.mkdtemp(prefix="afm_img_test_")
    path = os.path.join(folder, "cell0001.ibw")
    arr, note, labels = asylum_scan()
    write_ibw5(path, arr, "cell0001", note, labels)

    print("reading the file")
    item = ai.load_ibw_bytes(open(path, "rb").read(), "cell0001.ibw")
    check("the data come back as written",
          np.allclose(item["arr"], arr.astype(np.float32)))
    axis = ai.guess_channel_axis(item["arr"])
    names = [ai.channel_display_name(item["meta"], axis, i) for i in range(4)]
    check("the channels are found, with their own names",
          axis == 2 and names == labels[2], str(names))
    check("the note gives the line rate",
          item["guess"].get("scan_hz") == 2.0, str(item["guess"]))
    k, invols, sp = ai.note_calibration(item["note_dict"])
    check("and the calibration, InvOLS turned from m/V into nm/V",
          (k, invols, sp) == (0.08, 65.0, 0.5), str((k, invols, sp)))
    check("and the scan size in µm",
          ai.scan_size_um(item["note_dict"]) == (20.0, 20.0))

    print("the image")
    img = ai.orient_image(ai.channel_image(item["arr"], axis, 0) * 1e9,
                          "Rotate 90° CW")
    stats = ai.image_stats(img)
    check("heights in nm: the 400 nm cell is the tallest thing",
          390 < stats["max"] < 412 and stats["min"] < 5, str(stats["max"]))
    check("rotated 90° clockwise, (points, lines) becomes (lines, points)",
          img.shape == (96, 128), str(img.shape))
    lo, hi = ai.contrast_limits(img, "percentile", 1, 99)
    check("percentile contrast sits inside the range",
          stats["min"] <= lo < hi <= stats["max"])

    print("signal against time")
    A = ai.channel_image(item["arr"], axis, 2).T
    B = ai.channel_image(item["arr"], axis, 3).T
    t, y = ai.build_continuous_time(A, B, 2.0, 1 / 3, 1 / 3, 1 / 3)
    check("96 lines at 2 Hz last 48 s", abs(np.nanmax(t) - 48.0) < 0.01,
          f"{np.nanmax(t):.3f}")
    check("a third of the time is the move between lines (a gap)",
          abs(np.mean(np.isnan(y)) - 1 / 3) < 0.01)
    # the retrace is reversed, so it runs forward in time: the beat it
    # carries must line up with the trace's, not mirror it
    good = np.isfinite(y)
    beat = 0.5 + 0.3 * np.maximum(0, np.sin(2 * np.pi * 0.7 * t)) ** 4
    check("trace and retrace together follow the beat",
          np.corrcoef(y[good], beat[good])[0, 1] > 0.98)

    print("processing")
    f = ai.force_convert(y, "Defl (V) → force (nN)", k=k, invols=invols,
                         setpoint=sp, subtract_setpoint=True)
    check("deflection to force, relative to the setpoint",
          abs(np.nanmedian(f)) < 0.5 and abs(np.nanmax(f) - 0.3 * 65 * 0.08)
          < 0.2, f"{np.nanmax(f):.3f}")
    # between beats (sin < 0: 0.71 to 1.43 s) the cell is at rest at 0.5
    a, b = ai.fit_baseline(t, y, 0.8, 1.3, "Offset")
    check("an offset baseline is the median of the region",
          abs(b - 0.5) < 0.01 and a == 0.0, f"{b:.4f}")
    slope = 0.01 * t + 2.0
    a, b = ai.fit_baseline(t, slope, 0, 48, "Linear")
    check("a linear baseline recovers a straight line",
          abs(a - 0.01) < 1e-9 and abs(b - 2.0) < 1e-9)
    res, err, _ = ai.apply_ymath(t, y, 1000.0, -500.0, "")
    check("multiply then add", not err and np.allclose(
        res[good], y[good] * 1000 - 500))
    res, err, _ = ai.apply_ymath(t, y, formula="y^2")
    check("a formula, written the Excel way", not err and np.allclose(
        res[good], y[good] ** 2))
    res, err, _ = ai.apply_ymath(t, y, formula="y1 - y2", which=1,
                                 others={"y2": y * 0})
    check("a formula combining the two plots",
          not err and np.allclose(res[good], y[good]))
    _, err, _ = ai.apply_ymath(t, y, formula="__import__('os')")
    check("a formula cannot reach outside", bool(err), err)
    xs, ys = ai.smooth_xy(t, y, "Savitzky-Golay", 21, 3)
    check("smoothing keeps the gaps as gaps",
          np.isnan(ys).sum() >= 95 and np.nanstd(ys) < np.nanstd(y))
    pos, labels_ = ai.symlog_ticks(-1000.0, 1000.0, 1.0)
    check("symlog ticks are symmetric", "0" in labels_ and "1000" in labels_
          and "-1000" in labels_, str(labels_))

    print("saving")
    base = ai.default_save_base("cell0001.ibw", (2, 3), (0, None),
                                labels=names)
    check("files are named after the file and its own channel names",
          base == "cell0001_DeflectionTrace_DeflectionRetrace_vs_"
                  "HeightTrace_vs_time", base)
    import pandas as pd
    data, notes = ai.write_excel_bytes(
        [("TimeSeries", pd.DataFrame({"Time_s": t, "Signal": y}))])
    z = zipfile.ZipFile(io.BytesIO(data))
    check("the workbook is complete and has the gaps as empty cells",
          "xl/workbook.xml" in z.namelist() and not notes)

    print()
    if FAILED:
        print(f"{len(FAILED)} failing: {FAILED}")
        return 1
    print("all passing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
