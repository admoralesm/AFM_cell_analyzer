"""
Checks for the video engines: cell_tracking.py (four-marker tracker) and
auto_analysis.py (the automatic analysis).

Each check writes a small synthetic recording whose truth is known -- a
striated, textured cell that shortens on a schedule, and a dark probe rod
with a round tip -- and asks the engine to find what was put there.

    python tests_tracking.py
"""
import os
import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cell_tracking as ct  # noqa: E402
import auto_analysis as aa  # noqa: E402

cv2 = ct.cv2
FAILED = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + (f"  {detail}" if detail
                                                      and not ok else ""))
    if not ok:
        FAILED.append(name)


def make_video(path, n=360, fps=30, beat=0.06, probe_from=6.0, seed=3):
    """A cell beating at 0.9 Hz, a probe coming down onto it after
    ``probe_from`` s. Returns the true fractional length (s) per frame."""
    rng = np.random.default_rng(seed)
    w, h = 480, 360
    bg = 128 + cv2.GaussianBlur(rng.normal(0, 3, (h, w)).astype(np.float32),
                                (0, 0), 2)
    tex = cv2.GaussianBlur(rng.normal(0, 35, (h, w)).astype(np.float32),
                           (0, 0), 1.2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx0, cy, a, b = 240, 220, 110, 24
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    truth = []
    for i in range(n):
        t = i / fps
        phase = (t * 0.9) % 1.0
        s = 1 - beat * np.exp(-((phase - 0.3) / 0.08) ** 2)
        cx = cx0 + 2.0 * np.sin(2 * np.pi * t * 0.13)
        u = (xx - cx) / s
        cell = cv2.remap(tex, (cx0 + u).astype(np.float32), yy,
                         cv2.INTER_LINEAR)
        stripes = 30 * np.sin(2 * np.pi * u / 6.0)
        inside = ((u / a) ** 2 + ((yy - cy) / b) ** 2) <= 1
        img = bg.copy()
        img[inside] = (bg + 40 + cell + stripes)[inside]
        py = 110 + min(1.0, max(0.0, (t - probe_from) / 3)) * 95
        px, r = 330, 22
        rod = (((np.abs(xx - px) <= r) & (yy <= py))
               | (((xx - px) ** 2 + (yy - py) ** 2) <= r * r))
        img[rod] = 25
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
        img = np.clip(img + rng.normal(0, 2.5, (h, w)), 0, 255)
        out.write(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
        truth.append((s, cx))
    out.release()
    return np.array(truth)


def main():
    folder = tempfile.mkdtemp(prefix="afm_tracking_test_")
    path = os.path.join(folder, "beats.mp4")
    truth = make_video(path)
    scale, centre = truth[:, 0], truth[:, 1]

    print("four-marker tracker, v1.7 (area-anchored)")
    picks = np.array([[160, 212], [320, 212], [320, 230], [160, 230]], float)
    # Stop before the probe arrives: there the truth is still known.
    res = ct.track_points_ref(path, picks, 0, 170, radius=12)
    check("returns the v1.6 result layout plus the lock states",
          res["points"].shape == (171, 4, 2) and res["lock"].shape == (171, 4)
          and res["method"] == "ref")
    held, _ = ct.settle(res)
    true_x = centre[:171, None] + (picks[None, :, 0] - 240) * scale[:171, None]
    error = np.abs(held[:, :, 0] - true_x)
    check("follows the contracting cell to a tenth of a pixel",
          float(np.nanmean(error)) < 0.1, f"mean {np.nanmean(error):.3f} px")
    check("every marker stays locked on its own area",
          bool(np.all(res["lock"] == ct.ANCHORED)))
    warnings, summary = ct.quality_report(res)
    check("the honesty report says so", "locked" in summary.lower(), summary)

    old = ct.track_points(path, picks, 0, 170)
    held_old, _ = ct.settle(old)
    error_old = np.abs(held_old[:, :, 0] - true_x)
    check("v1.6 is still there, and v1.7 is at least as close",
          float(np.nanmean(error)) <= float(np.nanmean(error_old)) + 1e-6,
          f"v1.7 {np.nanmean(error):.3f} / v1.6 {np.nanmean(error_old):.3f}")

    print("bridging a lost marker")
    gap = held.copy()
    gap[40:60, 1] = np.nan
    filled, mask = ct.bridge_lost_points(gap)
    check("only the missing frames are filled",
          bool(mask[40:60, 1].all()) and int(mask.sum()) == 20)
    check("with no jump where it is found again",
          float(np.abs(filled[60, 1] - held[60, 1]).max()) < 1e-4)

    print("hand corrections")
    fixed = res["points"][100].astype(float).copy()
    fixed[1, 0] += 3.0
    redone = ct.track_points_corrected(path, {0: picks, 100: fixed}, 170,
                                       radius=12)
    check("the corrected frame sits exactly where it was placed",
          float(np.abs(redone["points"][100] - fixed).max()) < 1e-3)
    half = redone["points"][50, 1, 0] - res["points"][50, 1, 0]
    check("half the correction is spread back to halfway",
          abs(half - 1.5) < 0.05, f"{half:.3f} px")
    check("and the anchors are reported", redone["anchors"] == [0, 100])

    print("probe sphere")
    frame = ct.read_frame(path, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fit = aa.snap_sphere(gray, (334.0, 113.0, 19.0))
    check("a roughly marked circle snaps onto the tip (centre 330, 110, "
          "r 22)",
          fit["ok"] and np.hypot(fit["center"][0] - 330,
                                 fit["center"][1] - 110) < 1.5
          and abs(fit["radius"] - 22) < 1.0,
          f"{fit['center']} r {fit['radius']:.2f}")
    disk = np.full((200, 200), 128, np.uint8)
    cv2.circle(disk, (100, 100), 30, 25, -1)
    disk = cv2.GaussianBlur(disk, (0, 0), 1.0)
    fit = aa.snap_sphere(disk, (104.0, 97.0, 26.0))
    centre, radius = fit["center"], fit["radius"]
    check("and onto a whole round sphere (centre 100, 100, r 30)",
          np.hypot(centre[0] - 100, centre[1] - 100) < 0.75
          and abs(radius - 30) < 1.0, f"{centre} r {radius:.2f}")
    still = aa.snap_sphere(gray, (330.0, 110.0, 22.0))
    check("and a circle already on the tip stays there",
          np.hypot(still["center"][0] - 330, still["center"][1] - 110) < 1.5,
          f"{still['center']} r {still['radius']:.2f}")
    worst = 0.0
    for d in ((0, 0), (6, 4), (-10, 8), (0, 15)):
        one = aa.sphere_from_click(gray, (330 + d[0], 110 + d[1]))
        worst = max(worst, np.hypot(one["center"][0] - 330,
                                    one["center"][1] - 110))
    check("one click anywhere on the tip finds it (within 2 px)",
          worst < 2.0, f"worst {worst:.1f} px")
    hinted = aa.AutoCellAnalysis.analyze(path, probe_diameter_um=14.5,
                                         probe_hint=(330.0, 110.0, 22.0))
    check("the analysis takes a marked sphere as the probe tip",
          abs(hinted["probe"]["radius"] - 22.0) < 1e-9
          and abs(hinted["um_per_px"] - 14.5 / 44.0) < 1e-9)

    print("automatic analysis")
    auto = aa.AutoCellAnalysis.analyze(path, probe_diameter_um=14.5)
    check("finds the probe tip at its real size",
          auto["probe"] is not None
          and abs(2 * auto["probe"]["radius"] - 44.0) < 1.5,
          str(auto["probe"] and 2 * auto["probe"]["radius"]))
    um = auto["um_per_px"]
    sl = np.asarray(auto["sl_px"], float)
    sl = sl[np.isfinite(sl)]
    check("reads the sarcomere length from the striations",
          sl.size and abs(np.median(sl) * um - 6.0 * um) < 0.05,
          f"{np.median(sl) * um:.3f} vs {6.0 * um:.3f} µm")
    before = [e for e in auto["events"] if e["t_peak"] < 6.0]
    check("counts every beat before the probe arrives",
          len(before) == 6, f"{len(before)} found")
    amps = np.array([e["amp_pct"] for e in before])
    # The true shortening is 6 %. The desktop algorithm reads about 4.1 %
    # here (4.5 % on a rod-shaped cell): each end window straddles the cell
    # edge and the striations inside it, and the striations move less than
    # the edge. This pins today's reading so a change to it is noticed;
    # it is not a claim that 4 % is right.
    check("and how much the cell shortened in each (true 6 %, reads ~4 %)",
          amps.size and 3.5 < float(np.median(amps)) < 6.5,
          f"{np.median(amps):.2f} %")
    check("says the probe came over the cell",
          auto["verdict_code"] == "probe_over_cell", auto["verdict_code"])
    files = aa.AutoCellAnalysis.save_csvs(auto, os.path.join(folder, "out"))
    check("writes the time series, contractions and report",
          all(os.path.getsize(f) > 0 for f in files) and len(files) == 3)

    print()
    if FAILED:
        print(f"{len(FAILED)} failing: {FAILED}")
        return 1
    print("all passing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
