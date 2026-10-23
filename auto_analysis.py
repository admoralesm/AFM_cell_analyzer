"""
Automatic analysis of a 10x compression clip: nothing is picked by hand.

This is the desktop Video Tracker's v1.8 to v1.9.7 automatic analysis
("AutoCellAnalysis"), with the Tk window stripped off. It takes a path and
gives back a dict of arrays and plain-words notes, so the Streamlit "Cell
motion" tab, a script or a batch job can all drive it.

What it does, in order:
  1. Builds a clean reference picture from the first few seconds.
  2. Finds the probe (the big dark object) and its round tip. If no scale
     is given, the tip diameter gives µm/px.
  3. Finds the cell: the largest sharply textured object, preferring the
     one nearest the probe tip (or the outline / click it is given), and
     rebuilds the part hidden under the probe.
  4. On every frame: follows the cell, measures how much its length
     changed along the long axis, the sarcomere length from the
     striations, follows the probe tip and measures the probe-to-cell gap
     and how much of the cell the probe covers.
  5. Lists every contraction (time, size, duration, speeds) and says in
     plain words whether the probe ever came over the cell.

Frames the file will not decode (v1.9.4) and pictures stored twice
(v1.9.5) are found and left out; a wrong frame rate in the file header is
corrected from the frames' own time stamps.

The class body below is the desktop code unchanged, so results match the
desktop app frame for frame. Only numpy + OpenCV are used; the summary
figure (save_figure / save_all) additionally needs matplotlib and is not
used by the web app, which draws its own charts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as _exc:  # pragma: no cover - only on a broken deploy
    cv2 = None
    CV2_ERROR = str(_exc)
else:
    CV2_ERROR = ""

import cell_tracking as _ct
from cell_tracking import FPS_FIX, RobustCapture  # noqa: F401  (re-exported)

VERSION = "1.9.7"


class Config:
    """The desktop app's settings that the analysis reads."""

    DEFAULT_FPS = _ct.DEFAULT_FPS
    SCALE_BAR_UM = 40.0

    # All sizes are video pixels unless noted. The cut-offs adapt to each
    # video's own noise; these are only the floors and multipliers.
    AUTO_PROBE_DIAMETER_UM = 40.0     # default probe-tip diameter
    AUTO_REF_SECONDS = 3.0            # reference picture = first N seconds
    AUTO_REF_MAX_FRAMES = 31
    AUTO_PROBE_DARK_FRAC = 0.50       # probe = darker than this x background
    AUTO_PROBE_MIN_AREA_FRAC = 0.0008  # of the frame area
    AUTO_CELL_FINE_SIGMA = 1.0
    AUTO_CELL_ENERGY_SIGMA = 3.0
    AUTO_CELL_K_MAD = 8.0             # texture cut-off above background noise
    AUTO_CELL_MIN_ENERGY = 0.006
    AUTO_CELL_MIN_GRAD = 0.008
    AUTO_CELL_MIN_AREA_PX = 150
    AUTO_CELL_MIN_AREA_FRAC = 0.002   # of the frame area (debris is smaller)
    AUTO_CELL_KEEP_FRAC = 0.25        # candidates >= this x the largest
    AUTO_CELL_SEARCH_PAD = 12         # px searched around the last position
    AUTO_CELL_RECOVER_PAD = 45
    AUTO_CELL_MATCH_CONF = 0.45
    AUTO_END_MIN_CONF = 0.45          # min match to trust a cell end
    AUTO_EVENT_MIN_PCT = 0.5          # smallest shortening counted, % length
    AUTO_EVENT_K_SIGMA = 6.0
    AUTO_EVENT_BASELINE_S = 5.0       # running-baseline window
    AUTO_EVENT_MERGE_S = 0.25
    AUTO_PROBE_MOVE_MIN_PX = 3.0      # less than this = "probe did not move"
    AUTO_OVERLAP_MIN_FRAC = 0.05      # of the cell area under the probe
    AUTO_SPEED_SMOOTH_FRAMES = 5      # speed window (frames)
    AUTO_HINT_SNAP_PX = 6             # outline may be this far off the edge
    AUTO_SL_MIN_UM = 1.3              # sarcomere band searched
    AUTO_SL_MAX_UM = 3.0
    AUTO_SL_EDGE_FRAC = 0.15          # cell ends left out (each side)
    AUTO_SL_REF_SNR = 6.0             # striation peak / noise needed
    AUTO_SL_MIN_SNR = 3.0             # per frame
    AUTO_SL_DARK_FRAC = 0.88          # darker than this x the cell = covered
    AUTO_SL_SPIKE_FRAC = 0.08         # one-frame jump rejected above this
    AUTO_SPEED_POLYORDER = 4          # 4th order keeps fast peaks
    AUTO_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm")

    # The probe sphere: snapping a marked circle onto its edge, and finding
    # it without being shown (SphereEdgeRefiner, desktop v1.6+).
    SPHERE_REFINE_NUM_RAYS = 36
    SPHERE_REFINE_BAND_PX = 12
    SPHERE_REFINE_MIN_GRAD = 6.0
    SPHERE_REFINE_MIN_VALID_RAYS_FRAC = 0.30
    SPHERE_REFINE_BLUR_KSIZE = 3
    SPHERE_DETECT_NUM_FRAMES = 7
    SPHERE_DETECT_FRAME_STRIDE = 3
    SPHERE_DETECT_RADIUS_FRAC_MIN = 0.04
    SPHERE_DETECT_RADIUS_FRAC_MAX = 0.55
    SPHERE_DETECT_HOUGH_PARAM1 = 100
    SPHERE_DETECT_HOUGH_PARAM2 = 35
    SPHERE_DETECT_BLUR_KSIZE = 7
    SPHERE_DETECT_MEDIAN_KSIZE = 5
    SPHERE_DETECT_AGREEMENT_PX = 12.0
    SPHERE_DETECT_HINT_RADIUS_FRAC = 0.55
    SPHERE_DETECT_HINT_RMIN_FRAC = 0.10
    SPHERE_DETECT_EDGE_SHARPNESS_MIN = 0.55
    SPHERE_DETECT_EDGE_PROBE_BAND_FRAC = 0.12
    SPHERE_DETECT_EDGE_PROBE_RAYS = 24
    SPHERE_DETECT_TOP_K_BEFORE_SCORE = 8
    SPECULAR_THRESH = 235
    SPECULAR_INPAINT_RADIUS = 4


class Utils:
    clamp = staticmethod(_ct.clamp)

    @staticmethod
    def format_hhmmss(seconds):
        seconds = max(0.0, float(seconds))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:06.3f}"
        return f"{m:02d}:{s:06.3f}"


class Signal:
    smooth_derivative = staticmethod(_ct.smooth_derivative)
    plain_derivative = staticmethod(_ct.plain_derivative)


# --------------------------------------------- the desktop class, as is ---

class AutoCellAnalysis:
    """v1.8 — automatic analysis of a 10x clip. Nothing is picked by hand.

    What it does, in order:
      1. Builds a clean reference picture from the first few seconds.
      2. Finds the probe (the big dark object) and its round tip. If no
         scale is locked yet, the tip diameter gives µm/px.
      3. Finds the cell: the largest sharply textured object, preferring
         the one nearest the probe tip. Measures length, width, area and
         the direction of its long axis.
      4. On every frame: follows the cell (so stage movement is allowed),
         measures how much its length changed along the long axis, follows
         the probe tip, and measures the probe-to-cell gap and how much of
         the cell is covered by the probe.
      5. Lists every contraction (time, size, duration) and says in plain
         words whether the probe ever came over the cell.

    Only numpy + OpenCV are used. All cut-offs are derived from the video's
    own noise; the floors / multipliers are the Config.AUTO_* values.
    """

    # ------------------------------ helpers ------------------------------

    @staticmethod
    def norm_gray(bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Grey picture divided by its own background level (so lamp
        flicker / auto-exposure does not matter). Background ~ 1.0."""
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        g = g.astype(np.float32)
        bg = float(np.median(g[::4, ::4]))
        if not np.isfinite(bg) or bg < 1.0:
            bg = 1.0
        return g / bg, bg

    @staticmethod
    def _robust(v: np.ndarray) -> Tuple[float, float]:
        v = np.asarray(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0.0, 0.0
        med = float(np.median(v))
        return med, float(1.4826 * np.median(np.abs(v - med)))

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        m = (np.asarray(mask) > 0).astype(np.uint8) * 255
        pad = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
        pad[1:-1, 1:-1] = m
        ff = pad.copy()
        cv2.floodFill(ff, None, (0, 0), 128)
        return ((pad > 0) | (ff == 0))[1:-1, 1:-1]

    @staticmethod
    def _odd(k: float, lo: int = 3) -> int:
        k = max(int(lo), int(round(float(k))))
        return k if k % 2 == 1 else k + 1

    @staticmethod
    def _rolling_median(y: np.ndarray, win: int) -> np.ndarray:
        win = AutoCellAnalysis._odd(win)
        n = y.shape[0]
        if n == 0:
            return y.copy()
        if win >= n:
            return np.full(n, float(np.median(y)))
        h = win // 2
        # mirrored ends: a value sitting right at the start/end of the
        # range (e.g. a contraction cut by the range) must not dominate
        yp = np.pad(y, (h, h), mode="reflect")
        try:
            sw = np.lib.stride_tricks.sliding_window_view(yp, win)
            return np.median(sw, axis=1)
        except AttributeError:        # numpy older than 1.20
            return np.array([np.median(yp[i:i + win]) for i in range(n)])

    # ------------------------------ probe ------------------------------

    @staticmethod
    def find_probe(norm: np.ndarray) -> Optional[dict]:
        """Largest dark object. If it reaches the picture border it is a
        rod with a round tip at its free end ('rod'); otherwise a free
        disk ('disk'). Returns centre/radius of the tip, or None."""
        h, w = norm.shape[:2]
        g = cv2.GaussianBlur(norm, (0, 0), 2.0)
        mask = (g < float(Config.AUTO_PROBE_DARK_FRAC)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        k = 1 + int(np.argmax(areas))
        if float(areas[k - 1]) < float(Config.AUTO_PROBE_MIN_AREA_FRAC) * h * w:
            return None
        comp = (lab == k)
        ys, xs = np.nonzero(comp)
        on_border = (xs == 0) | (ys == 0) | (xs == w - 1) | (ys == h - 1)
        dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
        if on_border.any():
            bx = float(xs[on_border].mean())
            by = float(ys[on_border].mean())
            j = int(np.argmax((xs - bx) ** 2 + (ys - by) ** 2))
            ex, ey = float(xs[j]), float(ys[j])
            ux, uy = ex - bx, ey - by
            nrm = math.hypot(ux, uy)
            if nrm < 1.0:
                return None
            ux, uy = ux / nrm, uy / nrm
            rmax = float(dt.max())
            near = ((xs - ex) ** 2 + (ys - ey) ** 2) <= (3.0 * rmax) ** 2
            r = float(dt[ys[near], xs[near]].max())
            probe = {"kind": "rod", "axis": (ux, uy), "radius": r,
                     "center": (ex - r * ux, ey - r * uy), "mask": comp}
            # sub-pixel tip position + tip diameter from the soft edge
            m = AutoCellAnalysis._measure_probe(norm, probe,
                                                *probe["center"])
            if m is not None:
                probe["center"] = (m[0], m[1])
                d = AutoCellAnalysis._rod_tip_diameter(norm, probe)
                if d is not None and 0.6 * 2 * r <= d <= 1.6 * 2 * r:
                    probe["radius"] = 0.5 * d
                    m = AutoCellAnalysis._measure_probe(
                        norm, probe, *probe["center"])
                    if m is not None:
                        probe["center"] = (m[0], m[1])
            return probe
        area = float(comp.sum())
        probe = {"kind": "disk", "axis": (0.0, 0.0),
                 "radius": math.sqrt(area / math.pi),
                 "center": (float(xs.mean()), float(ys.mean())),
                 "mask": comp}
        m = AutoCellAnalysis._measure_probe(norm, probe, *probe["center"])
        if m is not None:
            probe["center"] = (m[0], m[1])
        return probe

    @staticmethod
    def _rod_patch(norm, probe, cx, cy):
        """Upright patch aligned with the rod: columns run ALONG the rod
        (towards the tip), rows ACROSS it. Returns (darkness 0..1, half_s,
        half_q) where darkness is 0.5 exactly on the 50 % edge."""
        r = float(probe["radius"])
        ux, uy = probe["axis"]
        vx, vy = -uy, ux
        half_s = int(math.ceil(2.5 * r))
        half_q = int(math.ceil(1.7 * r))
        W, H = 2 * half_s + 1, 2 * half_q + 1
        M = np.float32([
            [ux, vx, cx - half_s * ux - half_q * vx],
            [uy, vy, cy - half_s * uy - half_q * vy]])
        patch = cv2.warpAffine(
            norm, M, (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE)
        patch = cv2.GaussianBlur(patch, (0, 0), 1.5)
        thr = float(Config.AUTO_PROBE_DARK_FRAC)
        dark = np.clip(0.5 + (thr - patch) / 0.5, 0.0, 1.0)
        return dark, half_s, half_q

    @staticmethod
    def probe_from_hint(norm, detected, cx, cy, r) -> dict:
        """v1.9.7: the probe tip from the sphere you marked in Step 2
        (centre + radius as you set them). The automatic search tends to
        take the blurred dark halo for part of the sphere and so makes it
        too big; your circle is used as it is. A rod found leading to it
        is kept (it is dark too, never cell)."""
        h, w = norm.shape[:2]
        cx, cy, r = float(cx), float(cy), max(2.0, float(r))
        yy, xx = np.ogrid[0:h, 0:w]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = d2 <= r * r
        kind, axis = "disk", (0.0, 0.0)
        if detected is not None:
            dcx, dcy = detected["center"]
            near = math.hypot(dcx - cx, dcy - cy) <= 1.5 * max(
                r, float(detected["radius"]))
            if near:
                # What hides the cell is the dark probe itself, which the
                # search outlines well; only its centre and size (used for
                # the scale and for following the tip) come from your
                # circle. A circle set a little too big or too small then
                # does not change which part of the cell counts as hidden.
                kind, axis = detected["kind"], detected["axis"]
                mask = np.asarray(detected["mask"], bool)
        return {"kind": kind, "axis": axis, "radius": r,
                "center": (cx, cy), "mask": mask, "source": "step2"}

    @staticmethod
    def _measure_probe(norm, probe, cx, cy):
        """Sub-pixel tip centre near (cx, cy) on this frame -> (x, y,
        edge_sharpness) or None when the tip is not there."""
        r = float(probe["radius"])
        if probe["kind"] == "rod":
            dark, half_s, half_q = AutoCellAnalysis._rod_patch(
                norm, probe, cx, cy)
            ux, uy = probe["axis"]
            vx, vy = -uy, ux
            W, H = dark.shape[1], dark.shape[0]
            b = max(1, int(round(0.5 * r)))
            band = dark[half_q - b:half_q + b + 1, :].mean(axis=0)
            idx = np.nonzero(band >= 0.5)[0]
            if idx.size == 0:
                return None
            i = int(idx[-1])
            if i >= W - 1 or i < 2:
                return None
            den = float(band[i] - band[i + 1])
            s_apex = i + (float(band[i]) - 0.5) / den if den > 1e-9 else float(i)
            s0 = int(max(0, round(s_apex - 1.5 * r)))
            s1 = int(max(s0 + 1, round(s_apex - 0.5 * r)))
            col = dark[:, s0:s1].mean(axis=1)
            tot = float(col.sum())
            if tot < 1e-6:
                return None
            q_c = float((col * np.arange(H)).sum() / tot)
            ds = (s_apex - r) - half_s
            dq = q_c - half_q
            lo, hi = max(0, i - 4), min(W - 1, i + 5)
            sharp = float(np.max(np.abs(np.diff(band[lo:hi + 1]))))
            return (cx + ds * ux + dq * vx, cy + ds * uy + dq * vy, sharp)

        # free disk: darkness-weighted centroid, re-centred a few times
        h, w = norm.shape[:2]
        R = int(math.ceil(1.6 * r))
        sharp = float("nan")
        for _ in range(4):
            x0, x1 = int(round(cx)) - R, int(round(cx)) + R + 1
            y0, y1 = int(round(cy)) - R, int(round(cy)) + R + 1
            xa, xb = max(0, x0), min(w, x1)
            ya, yb = max(0, y0), min(h, y1)
            if xb - xa < 5 or yb - ya < 5:
                return None
            crop = cv2.GaussianBlur(norm[ya:yb, xa:xb], (0, 0), 1.5)
            dark = np.clip(
                0.5 + (float(Config.AUTO_PROBE_DARK_FRAC) - crop) / 0.5,
                0.0, 1.0)
            tot = float(dark.sum())
            if tot < 0.15 * math.pi * r * r:
                return None
            yy, xx = np.mgrid[ya:yb, xa:xb]
            nx = float((dark * xx).sum() / tot)
            ny = float((dark * yy).sum() / tot)
            moved = math.hypot(nx - cx, ny - cy)
            cx, cy = nx, ny
            gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            edge = (dark > 0.2) & (dark < 0.8)
            if edge.any():
                sharp = float(np.hypot(gx, gy)[edge].mean())
            if moved < 0.05:
                break
        return (cx, cy, sharp)

    @staticmethod
    def _rod_tip_diameter(norm, probe) -> Optional[float]:
        """Widest part of the rod within one diameter of its free end,
        measured edge-to-edge at half darkness (sub-pixel)."""
        cx, cy = probe["center"]
        r = float(probe["radius"])
        dark, half_s, half_q = AutoCellAnalysis._rod_patch(
            norm, probe, cx, cy)
        H = dark.shape[0]
        widths = []
        for s in range(int(round(half_s - r)), int(round(half_s + 0.6 * r))):
            if s < 0 or s >= dark.shape[1]:
                continue
            col = dark[:, s]
            idx = np.nonzero(col >= 0.5)[0]
            if idx.size < 3:
                continue
            a, b = int(idx[0]), int(idx[-1])
            if a <= 0 or b >= H - 1:
                continue
            ea = a - (float(col[a]) - 0.5) / max(1e-9, float(col[a] - col[a - 1]))
            eb = b + (float(col[b]) - 0.5) / max(1e-9, float(col[b] - col[b + 1]))
            widths.append(eb - ea)
        if len(widths) < 5:
            return None
        w = np.sort(np.array(widths))
        return float(np.mean(w[-max(3, len(w) // 5):]))

    # ------------------------------ cell ------------------------------

    @staticmethod
    def find_cell(norm: np.ndarray, probe: Optional[dict] = None,
                  hint_xy: Optional[Tuple[float, float]] = None
                  ) -> Optional[dict]:
        """The cell = a big, sharply textured object. Two steps: (1) a
        fine-texture map locates the candidates (the blurred probe halo is
        handled by raising the cut-off ring by ring around the probe);
        (2) the outline is tightened on the edge strength."""
        h, w = norm.shape[:2]
        s1 = float(Config.AUTO_CELL_FINE_SIGMA)
        f1 = cv2.GaussianBlur(norm, (0, 0), s1)
        bp = f1 - cv2.GaussianBlur(norm, (0, 0), 2.0 * s1)
        en = np.sqrt(cv2.GaussianBlur(
            bp * bp, (0, 0), float(Config.AUTO_CELL_ENERGY_SIGMA)))

        pm = None
        far = np.ones((h, w), dtype=bool)
        dist = None
        if probe is not None:
            pm = probe["mask"]
            dist = cv2.distanceTransform(
                (~pm).astype(np.uint8), cv2.DIST_L2, 5)
            far = dist > 2.0 * float(probe["radius"])
        med, sig = AutoCellAnalysis._robust(en[far][::7])
        base_thr = max(float(Config.AUTO_CELL_MIN_ENERGY),
                       med + float(Config.AUTO_CELL_K_MAD) * sig)
        thr = np.full((h, w), base_thr, dtype=np.float32)
        if probe is not None:
            dmax = 2.0 * float(probe["radius"])
            d = 0.0
            while d < dmax:
                ring = (dist > d) & (dist <= d + 2.0)
                if ring.any():
                    thr[ring] = max(base_thr, 3.0 * float(np.median(en[ring])))
                d += 2.0
        coarse = en > thr
        if pm is not None:
            coarse &= ~pm
        coarse = cv2.morphologyEx(
            coarse.astype(np.uint8), cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        coarse = AutoCellAnalysis._fill_holes(coarse).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(coarse, 8)
        if n <= 1:
            return None
        # debris is far smaller than a cell: ask for a real share of the
        # picture, unless the user pointed at the cell (hint)
        min_area = float(Config.AUTO_CELL_MIN_AREA_PX)
        if hint_xy is None:
            min_area = max(min_area,
                           float(Config.AUTO_CELL_MIN_AREA_FRAC) * h * w)
        cand = [k for k in range(1, n)
                if stats[k, cv2.CC_STAT_AREA] >= min_area]
        if not cand:
            return None
        amax = max(int(stats[k, cv2.CC_STAT_AREA]) for k in cand)
        cand = [k for k in cand if stats[k, cv2.CC_STAT_AREA]
                >= float(Config.AUTO_CELL_KEEP_FRAC) * amax]

        def _dist_to(k, xy):
            ys_, xs_ = np.nonzero(lab == k)
            return float(np.min(np.hypot(xs_ - xy[0], ys_ - xy[1])))

        if hint_xy is not None:
            best = min(cand, key=lambda k: _dist_to(k, hint_xy))
        elif probe is not None and len(cand) > 1:
            best = min(cand, key=lambda k: _dist_to(k, probe["center"]))
        else:
            best = max(cand, key=lambda k: int(stats[k, cv2.CC_STAT_AREA]))
        blob = (lab == best)

        # v1.9.6: the probe lies ON the cell. Where it covers the middle of
        # the cell, the visible cell falls apart into pieces on either side
        # of it: join pieces that sit on the opposite side of the probe.
        probe_zone = None
        if probe is not None:
            pr = float(probe["radius"])
            kz = int(round(0.25 * pr)) + 2
            probe_zone = cv2.dilate(pm.astype(np.uint8), cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * kz + 1, 2 * kz + 1))) > 0
            kt = kz + int(round(0.3 * pr)) + 3
            touch = cv2.dilate(pm.astype(np.uint8), cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * kt + 1, 2 * kt + 1))) > 0
            if (blob & touch).any():
                pcx, pcy = probe["center"]
                by, bx = np.nonzero(blob)
                vb = np.array([bx.mean() - pcx, by.mean() - pcy])
                amin = max(30.0, 0.08 * float(blob.sum()))
                for k in range(1, n):
                    if k == best or stats[k, cv2.CC_STAT_AREA] < amin:
                        continue
                    comp = lab == k
                    if not (comp & touch).any():
                        continue
                    vk = np.array([cents[k][0] - pcx, cents[k][1] - pcy])
                    cosang = float(vb @ vk) / max(
                        1e-9, np.linalg.norm(vb) * np.linalg.norm(vk))
                    if cosang < 0.0:            # other side of the probe
                        blob |= comp

        # ---- tighten the outline on the edge strength ----
        gx = cv2.Sobel(f1, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(f1, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        gm = np.hypot(gx, gy)
        gmed, gsig = AutoCellAnalysis._robust(gm[far][::7])
        gthr = max(float(Config.AUTO_CELL_MIN_GRAD),
                   gmed + float(Config.AUTO_CELL_K_MAD) * gsig,
                   0.3 * float(np.percentile(gm[blob], 90)))
        zone = cv2.dilate(blob.astype(np.uint8),
                          np.ones((5, 5), np.uint8)) > 0
        fine = (gm > gthr) & zone
        if pm is not None:
            fine &= ~pm
        fine = cv2.morphologyEx(
            fine.astype(np.uint8), cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        fine = AutoCellAnalysis._fill_holes(fine).astype(np.uint8)
        # the edge-strength skirt reaches ~1 px outside the true edge
        fine = cv2.erode(fine, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)))
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(fine, 8)
        if n2 <= 1:
            mask = blob
        else:
            amax2 = int(st2[1:, cv2.CC_STAT_AREA].max())
            mask = np.zeros_like(blob)
            for k2 in range(1, n2):
                comp = lab2 == k2
                a2 = int(st2[k2, cv2.CC_STAT_AREA])
                if a2 == amax2 or (a2 >= 0.1 * amax2 and
                                   int((comp & blob).sum()) >= 0.5 * a2
                                   and probe_zone is not None):
                    mask |= comp
            if int(mask.sum()) < 0.3 * int(blob.sum()):
                mask = blob

        hidden = None
        if probe_zone is not None:
            mask, hidden = AutoCellAnalysis.fill_under_probe(mask, probe_zone)

        ys, xs = np.nonzero(mask)
        if xs.size < int(Config.AUTO_CELL_MIN_AREA_PX) // 2:
            return None
        cx, cy = float(xs.mean()), float(ys.mean())
        cov = np.cov(np.vstack([xs - cx, ys - cy]))
        evals, evecs = np.linalg.eigh(cov)
        ax, ay = float(evecs[0, 1]), float(evecs[1, 1])
        if ay < 0 or (abs(ay) < 1e-9 and ax < 0):
            ax, ay = -ax, -ay
        p = (xs - cx) * ax + (ys - cy) * ay
        q = -(xs - cx) * ay + (ys - cy) * ax
        length = float(p.max() - p.min() + 1.0)
        wmax = float(q.max() - q.min() + 1.0)
        # re-centre on the middle of the long/short extents
        pc = 0.5 * float(p.max() + p.min())
        qc = 0.5 * float(q.max() + q.min())
        ccx = cx + pc * ax - qc * ay
        ccy = cy + pc * ay + qc * ax
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        contour = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(
            np.float32)
        area = float(mask.sum())
        return {
            "mask": mask, "contour": contour, "ys": ys, "xs": xs,
            "center": (ccx, ccy), "centroid": (cx, cy), "axis": (ax, ay),
            "length_px": length, "width_max_px": wmax,
            "width_mean_px": area / max(1.0, length), "area_px": area,
            "angle_deg": float(math.degrees(math.atan2(ay, ax)) % 180.0),
            "bbox": (int(xs.min()), int(ys.min()),
                     int(xs.max()) + 1, int(ys.max()) + 1),
            "n_candidates": len(cand),
            "hidden_mask": hidden,
            "hidden_px": float(hidden.sum()) if hidden is not None else 0.0,
        }

    @staticmethod
    def hidden_contours(cell: dict) -> list:
        """Outlines (Nx2 x,y) of the part rebuilt under the probe."""
        if "hidden_contours" in cell:
            return cell["hidden_contours"]
        hm = cell.get("hidden_mask")
        out = []
        if hm is not None and np.any(hm):
            cs, _ = cv2.findContours(np.asarray(hm, np.uint8),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
            out = [c.reshape(-1, 2).astype(np.float32) for c in cs
                   if len(c) >= 5]
        cell["hidden_contours"] = out
        return out

    @staticmethod
    def fill_under_probe(mask: np.ndarray, probe_zone: np.ndarray):
        """v1.9.6: rebuild the part of the cell hidden under the probe.
        The probe lies on top of the cell, so the cell continues beneath
        it. Along the cell's long axis, every cross-section that is clear
        of the probe gives the cell's two side edges there; across the
        covered stretch both edges are carried straight from the last clear
        section before the probe to the first one after it. Only pixels
        inside the probe's footprint are changed; the visible outline stays
        as it was found. (If too little of the cell is clear on both sides
        of the probe, the outline's convex hull is used instead.)
        Returns (new mask, added pixels or None)."""
        mask = np.asarray(mask, dtype=bool)
        zone = np.asarray(probe_zone, dtype=bool)
        near = cv2.dilate(zone.astype(np.uint8),
                          np.ones((5, 5), np.uint8)) > 0
        if not (mask & near).any():
            return mask, None
        vis = mask & ~zone
        ys, xs = np.nonzero(vis)
        if xs.size < 30:
            return mask, None
        cx, cy = float(xs.mean()), float(ys.mean())
        evals, evecs = np.linalg.eigh(np.cov(np.vstack([xs - cx, ys - cy])))
        ax, ay = float(evecs[0, 1]), float(evecs[1, 1])
        u = (xs - cx) * ax + (ys - cy) * ay
        v = -(xs - cx) * ay + (ys - cy) * ax
        ub = np.round(u).astype(int)
        u0, u1 = int(ub.min()), int(ub.max())
        h, w = mask.shape
        clean_u, lo_l, hi_l = [], [], []
        for k in range(u0, u1 + 1):
            sel = ub == k
            if sel.sum() < 2:
                continue
            vlo, vhi = float(v[sel].min()), float(v[sel].max())
            # is the whole section (with a margin) clear of the probe?
            vv = np.arange(vlo - 3.0, vhi + 3.01, 1.0)
            px = np.clip(np.round(cx + k * ax - vv * ay).astype(int), 0, w - 1)
            py = np.clip(np.round(cy + k * ay + vv * ax).astype(int), 0, h - 1)
            if zone[py, px].any():
                continue
            clean_u.append(k)
            lo_l.append(vlo)
            hi_l.append(vhi)
        clean_u = np.asarray(clean_u, dtype=float)
        new = None
        if clean_u.size >= 8:
            lo_a = np.asarray(lo_l)
            hi_a = np.asarray(hi_l)
            # steady edges: median over 5 sections
            if lo_a.size >= 5:
                pad = 2
                lo_p = np.pad(lo_a, pad, mode="edge")
                hi_p = np.pad(hi_a, pad, mode="edge")
                lo_a = np.array([np.median(lo_p[i:i + 5])
                                 for i in range(lo_a.size)])
                hi_a = np.array([np.median(hi_p[i:i + 5])
                                 for i in range(hi_a.size)])
            # the covered stretch must have clear sections on both sides
            gaps = np.diff(clean_u)
            if gaps.size and gaps.max() > 2:
                yz, xz = np.nonzero(zone)
                uz = (xz - cx) * ax + (yz - cy) * ay
                vz = -(xz - cx) * ay + (yz - cy) * ax
                inside_u = (uz >= clean_u[0]) & (uz <= clean_u[-1])
                lo_i = np.interp(uz, clean_u, lo_a)
                hi_i = np.interp(uz, clean_u, hi_a)
                # only where the section is NOT clear (covered stretch)
                k_near = np.searchsorted(clean_u, uz)
                k_near = np.clip(k_near, 1, clean_u.size - 1)
                in_gap = (clean_u[k_near] - clean_u[k_near - 1]) > 2
                band = inside_u & in_gap & (vz >= lo_i - 0.5) & \
                    (vz <= hi_i + 0.5)
                bm = np.zeros_like(mask)
                bm[yz[band], xz[band]] = True
                if bm.sum() >= 10:
                    new = (mask & ~zone) | bm
        if new is None:
            hull = cv2.convexHull(np.column_stack(
                np.nonzero(mask)[::-1]).astype(np.int32))
            hm = np.zeros(mask.shape, np.uint8)
            cv2.fillConvexPoly(hm, hull, 1)
            add = (hm > 0) & zone & ~mask
            if int(add.sum()) < 10:
                return mask, None
            new = mask | add
        new = cv2.morphologyEx(new.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(
                                   cv2.MORPH_ELLIPSE, (5, 5))) > 0
        new = AutoCellAnalysis._fill_holes(new.astype(np.uint8)) > 0
        n3, lab3, st3, _ = cv2.connectedComponentsWithStats(
            new.astype(np.uint8), 8)
        if n3 > 2:
            new = lab3 == (1 + int(np.argmax(st3[1:, cv2.CC_STAT_AREA])))
        hidden = new & ~mask & zone        # what was under the probe
        return new, (hidden if hidden.any() else None)

    @staticmethod
    def _strip_geometry(cell: dict) -> dict:
        """Size of the upright cell patch and of the two end windows used
        to follow the cell's ends (see analyze)."""
        L = float(cell["length_px"])
        maxs = int(max(8, round(0.10 * L)))      # largest end movement
        m = maxs + 6                             # background kept past ends
        H = int(round(L)) + 2 * m
        Wp = int(round(cell["width_max_px"])) + 4
        hw = int(max(6, round(0.10 * H)))        # end-window half length
        hw = int(min(hw, max(3, (H - 2 * maxs - 3) // 4)))
        return {"H": H, "W": Wp, "margin": m, "maxs": maxs, "hw": hw}

    @staticmethod
    def _strip(norm, cell, geo, dx=0.0, dy=0.0) -> np.ndarray:
        """Upright patch of the cell: rows run ALONG the long axis."""
        ax, ay = cell["axis"]
        bx, by = -ay, ax
        cx, cy = cell["center"]
        cx, cy = cx + dx, cy + dy
        H, Wp = geo["H"], geo["W"]
        M = np.float32([
            [bx, ax, cx - 0.5 * (Wp - 1) * bx - 0.5 * (H - 1) * ax],
            [by, ay, cy - 0.5 * (Wp - 1) * by - 0.5 * (H - 1) * ay]])
        return cv2.warpAffine(
            norm, M, (Wp, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _highpass_rows(P: np.ndarray, k: int) -> np.ndarray:
        k = AutoCellAnalysis._odd(k)
        h = k // 2
        Pp = np.pad(P, ((0, 0), (h, h)), mode="reflect")
        c = np.cumsum(np.pad(Pp, ((0, 0), (1, 0))), axis=1)
        return P - (c[:, k:] - c[:, :-k]) / float(k)

    @staticmethod
    def _local_shifts(P: np.ndarray, ref: np.ndarray, centers, hw: int,
                      maxs: int) -> Tuple[np.ndarray, np.ndarray]:
        """For each window along the cell: where did the piece of the
        REFERENCE profile centred there go on every frame? -> (shift px
        (N, K), match score (N, K)). Sub-pixel, brightness-independent."""
        N = P.shape[0]
        K = len(centers)
        shifts = np.zeros((N, K), dtype=np.float64)
        scores = np.zeros((N, K), dtype=np.float64)
        lags = np.arange(-maxs, maxs + 1)
        rows = np.arange(N)
        for k, c in enumerate(centers):
            c = int(round(c))
            a = ref[c - hw:c + hw + 1].astype(np.float64)
            a = a - a.mean()
            na = math.sqrt(float((a * a).sum())) + 1e-12
            cors = np.empty((N, lags.size), dtype=np.float64)
            for j, s in enumerate(lags):
                b = P[:, c - hw + s:c + hw + 1 + s].astype(np.float64)
                b = b - b.mean(axis=1, keepdims=True)
                nb = np.sqrt((b * b).sum(axis=1)) + 1e-12
                cors[:, j] = (b @ a) / (na * nb)
            jb = np.argmax(cors, axis=1)
            y1 = cors[rows, jb]
            jm = np.clip(jb - 1, 0, lags.size - 1)
            jp = np.clip(jb + 1, 0, lags.size - 1)
            y0, y2 = cors[rows, jm], cors[rows, jp]
            den = y0 - 2.0 * y1 + y2
            d = np.where((np.abs(den) > 1e-12) & (jb > 0)
                         & (jb < lags.size - 1),
                         0.5 * (y0 - y2) / np.where(
                             np.abs(den) > 1e-12, den, 1.0), 0.0)
            shifts[:, k] = lags[jb] + np.clip(d, -0.5, 0.5)
            scores[:, k] = y1
        return shifts, scores

    # ------------------------------ events ------------------------------

    @staticmethod
    def detect_events(t: np.ndarray, pct: np.ndarray) -> dict:
        """Contractions = dips of the length trace below its own running
        baseline. Returns {"events": [...], "baseline", "sigma", "thr"}."""
        t = np.asarray(t, dtype=np.float64)
        pct = np.asarray(pct, dtype=np.float64)
        n = pct.shape[0]
        valid = np.isfinite(pct)
        out = {"events": [], "baseline": np.full(n, np.nan),
               "sigma": float("nan"), "thr": float("nan")}
        if n < 8 or int(valid.sum()) < 8:
            return out
        y = pct.copy()
        if not valid.all():
            y[~valid] = np.interp(t[~valid], t[valid], pct[valid])
        dt_med = float(np.median(np.diff(t))) if n > 1 else 1.0
        dt_med = dt_med if dt_med > 1e-6 else 1.0 / Config.DEFAULT_FPS
        win = AutoCellAnalysis._odd(
            float(Config.AUTO_EVENT_BASELINE_S) / dt_med, lo=5)
        base = AutoCellAnalysis._rolling_median(y, win)
        d = y - base
        _, sig = AutoCellAnalysis._robust(d[valid])
        thr = max(float(Config.AUTO_EVENT_MIN_PCT),
                  float(Config.AUTO_EVENT_K_SIGMA) * sig)
        out.update(baseline=base, sigma=sig, thr=thr)

        below = (d < -0.5 * thr) & valid
        runs = []
        i = 0
        while i < n:
            if below[i]:
                j = i
                while j + 1 < n and below[j + 1]:
                    j += 1
                runs.append([i, j])
                i = j + 1
            else:
                i += 1
        merged = []
        for r in runs:
            if merged and (t[r[0]] - t[merged[-1][1]]
                           < float(Config.AUTO_EVENT_MERGE_S)):
                merged[-1][1] = r[1]
            else:
                merged.append(r)

        def _cross(i_from, step, level):
            """Walk from i_from in direction step until d rises above
            `level`; return the interpolated time (or None at the end)."""
            i = i_from
            while 0 <= i + step < n:
                a, b = d[i], d[i + step]
                if b >= level > a or (a <= level <= b):
                    if abs(b - a) < 1e-12:
                        return float(t[i + step])
                    f = (level - a) / (b - a)
                    return float(t[i] + f * (t[i + step] - t[i]))
                i += step
            return None

        vel = np.gradient(y, t) if n > 2 else np.zeros(n)
        events = []
        for i0, i1 in merged:
            ip = i0 + int(np.argmin(d[i0:i1 + 1]))
            amp = float(-d[ip])
            if amp < thr:
                continue
            t_on = _cross(ip, -1, -0.1 * amp)
            t_off = _cross(ip, +1, -0.1 * amp)
            t_h0 = _cross(ip, -1, -0.5 * amp)
            t_h1 = _cross(ip, +1, -0.5 * amp)
            partial = (t_on is None) or (t_off is None)
            a_idx = int(np.searchsorted(t, t_on)) if t_on is not None else 0
            b_idx = (int(np.searchsorted(t, t_off)) if t_off is not None
                     else n - 1)
            a_idx = int(np.clip(a_idx - 1, 0, ip))
            b_idx = int(np.clip(b_idx, ip, n - 1))
            events.append({
                "index": ip, "t_peak": float(t[ip]), "amp_pct": amp,
                "t_onset": t_on, "t_end": t_off,
                "time_to_peak": (float(t[ip]) - t_on
                                 if t_on is not None else float("nan")),
                "relax90": (t_off - float(t[ip])
                            if t_off is not None else float("nan")),
                "dur50": (t_h1 - t_h0 if (t_h0 is not None
                                          and t_h1 is not None)
                          else float("nan")),
                "v_short_pct_s": float(np.min(vel[a_idx:ip + 1])),
                "v_relax_pct_s": float(np.max(vel[ip:b_idx + 1])),
                "partial": bool(partial),
            })
        for k, ev in enumerate(events):
            ev["interval"] = (ev["t_peak"] - events[k - 1]["t_peak"]
                              if k > 0 else float("nan"))
        out["events"] = events
        return out

    # ------------------------------ main ------------------------------

    @staticmethod
    def analyze(video_path: str, start_frame: int = 0,
                end_frame: Optional[int] = None,
                um_per_px: Optional[float] = None,
                probe_diameter_um: Optional[float] = None,
                hint_xy: Optional[Tuple[float, float]] = None,
                progress_callback=None,
                cell_outline=None, probe_hint=None) -> Optional[dict]:
        """Run the whole automatic analysis. Returns a result dict, or None
        if cancelled. Raises RuntimeError with a plain-words reason when no
        cell can be found."""
        A = AutoCellAnalysis
        cap = RobustCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video for automatic analysis")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or Config.DEFAULT_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                raise RuntimeError("The video reports no frames.")
            start_frame = int(Utils.clamp(start_frame, 0, total - 1))
            if end_frame is None:
                end_frame = total - 1
            end_frame = int(Utils.clamp(end_frame, start_frame, total - 1))
            n_total = end_frame - start_frame + 1

            # ---------- 1) reference picture ----------
            span = int(min(n_total, max(5, round(
                float(Config.AUTO_REF_SECONDS) * fps))))
            step = max(1, int(math.ceil(
                span / float(Config.AUTO_REF_MAX_FRAMES))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            stack = []
            bg_levels = []
            for i in range(span):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if i % step == 0:
                    nrm, bg = A.norm_gray(frame)
                    stack.append(nrm)
                    bg_levels.append(bg)
            if not stack:
                raise RuntimeError("Could not read the start of the range.")
            ref = (np.median(np.stack(stack), axis=0).astype(np.float32)
                   if len(stack) > 1 else stack[0])
            del stack
            h, w = ref.shape[:2]
            bg0 = float(np.median(bg_levels))

            # ---------- 2) probe, 3) cell ----------
            probe = A.find_probe(ref)
            if probe_hint is not None:
                # v1.9.7: the sphere you marked in Step 2 IS the probe tip
                probe = A.probe_from_hint(ref, probe, *probe_hint)
            if cell_outline is not None and len(cell_outline) >= 3:
                # v1.9: your outline is the cell (snapped to its edges
                # where it is visible)
                cell = A.cell_from_outline(ref, cell_outline, probe)
                if cell is None:
                    raise RuntimeError(
                        "The cell outline you drew is too small. Draw it "
                        "around the whole cell (at least 3 clicks).")
            else:
                cell = A.find_cell(ref, probe, hint_xy)
                if cell is not None:
                    cell["source"] = "click" if hint_xy is not None \
                        else "auto"
            if cell is None:
                raise RuntimeError(
                    "No cell found in the first "
                    f"{Config.AUTO_REF_SECONDS:g} s of the range. Make sure "
                    "the cell is in focus there, or pick 4 points on it "
                    "first (they are used as a hint) and run again.")

            scale_source = "none (pixels)"
            if um_per_px is not None and np.isfinite(um_per_px) \
                    and um_per_px > 0:
                um = float(um_per_px)
                scale_source = "scale locked in Step 2"
            elif (probe is not None and probe_diameter_um is not None
                  and probe_diameter_um > 0):
                um = float(probe_diameter_um) / (2.0 * float(probe["radius"]))
                scale_source = (f"probe tip = {probe_diameter_um:g} µm "
                                f"({2.0 * probe['radius']:.1f} px wide)")
            else:
                um = None

            # ---------- 4) every frame ----------
            geo = A._strip_geometry(cell)
            x0, y0, x1, y1 = cell["bbox"]
            pad = 4
            tx0, ty0 = max(0, x0 - pad), max(0, y0 - pad)
            tx1, ty1 = min(w, x1 + pad), min(h, y1 + pad)
            # The cell is followed as THREE pieces (thirds along its longer
            # side). When the probe hides one piece the others still say
            # where the cell is, so it can be followed under the probe.
            parts = []
            if (ty1 - ty0) >= (tx1 - tx0):
                cuts = np.linspace(ty0, ty1, 4).round().astype(int)
                for j in range(3):
                    parts.append((tx0, int(cuts[j]), tx1, int(cuts[j + 1])))
            else:
                cuts = np.linspace(tx0, tx1, 4).round().astype(int)
                for j in range(3):
                    parts.append((int(cuts[j]), ty0, int(cuts[j + 1]), ty1))
            templs = [np.ascontiguousarray(ref[b_:d_, a_:c_])
                      for (a_, b_, c_, d_) in parts]

            def locate(nrm, templ, lx, ly, spad):
                """Template position near (lx, ly) -> (x, y, score)."""
                th, tw = templ.shape[:2]
                sx0 = int(max(0, round(lx) - spad))
                sy0 = int(max(0, round(ly) - spad))
                sx1 = int(min(w, round(lx) + tw + spad))
                sy1 = int(min(h, round(ly) + th + spad))
                if sx1 - sx0 < tw or sy1 - sy0 < th:
                    return lx, ly, 0.0
                resm = cv2.matchTemplate(
                    nrm[sy0:sy1, sx0:sx1], templ, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(resm)
                if not np.isfinite(mx):
                    return lx, ly, 0.0
                px, py = float(loc[0]), float(loc[1])
                if 0 < loc[0] < resm.shape[1] - 1:
                    a_, b_, c_ = (resm[loc[1], loc[0] - 1], mx,
                                  resm[loc[1], loc[0] + 1])
                    den = a_ - 2 * b_ + c_
                    if abs(den) > 1e-9:
                        px += float(np.clip(0.5 * (a_ - c_) / den, -.5, .5))
                if 0 < loc[1] < resm.shape[0] - 1:
                    a_, b_, c_ = (resm[loc[1] - 1, loc[0]], mx,
                                  resm[loc[1] + 1, loc[0]])
                    den = a_ - 2 * b_ + c_
                    if abs(den) > 1e-9:
                        py += float(np.clip(0.5 * (a_ - c_) / den, -.5, .5))
                return sx0 + px, sy0 + py, float(mx)

            def follow_cell(nrm, dx, dy, spad):
                """Shift of the cell vs the reference near (dx, dy) ->
                (dx, dy, best score, n pieces found)."""
                found = []
                for (a_, b_, c_, d_), tm in zip(parts, templs):
                    x, y, sc = locate(nrm, tm, a_ + dx, b_ + dy, spad)
                    if sc >= conf:
                        found.append((x - a_, y - b_, sc))
                if not found:
                    return dx, dy, 0.0, 0
                f_ = np.array(found)
                if len(found) == 3:
                    ndx, ndy = float(np.median(f_[:, 0])), float(
                        np.median(f_[:, 1]))
                else:
                    wsum = float(f_[:, 2].sum())
                    ndx = float((f_[:, 0] * f_[:, 2]).sum() / wsum)
                    ndy = float((f_[:, 1] * f_[:, 2]).sum() / wsum)
                return ndx, ndy, float(f_[:, 2].max()), len(found)

            sub = slice(None, None, 2)
            cys = cell["ys"][sub].astype(np.float32)
            cxs = cell["xs"][sub].astype(np.float32)
            cont = cell["contour"][::2]

            frames_idx = np.zeros(n_total, dtype=np.int64)
            t_ms = np.full(n_total, np.nan)
            prof = np.zeros((n_total, geo["H"]), dtype=np.float32)
            cdx = np.zeros(n_total)
            cdy = np.zeros(n_total)
            cscore = np.zeros(n_total)
            pxs = np.full(n_total, np.nan)
            pys = np.full(n_total, np.nan)
            psharp = np.full(n_total, np.nan)
            gap = np.full(n_total, np.nan)
            overlap = np.zeros(n_total)

            chg = np.full(n_total, np.nan)    # v1.9.5: change vs previous
            prev_small = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ldx, ldy = 0.0, 0.0
            pcx, pcy = (probe["center"] if probe is not None
                        else (float("nan"), float("nan")))
            dark_frac = float(Config.AUTO_PROBE_DARK_FRAC)
            conf = float(Config.AUTO_CELL_MATCH_CONF)
            n_done = 0
            for i in range(n_total):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                t_ms[i] = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                frames_idx[i] = start_frame + i
                nrm, _bg = A.norm_gray(frame)

                # -- cell position (rigid shift vs the reference) --
                ndx, ndy, sc, nf = follow_cell(
                    nrm, ldx, ldy, int(Config.AUTO_CELL_SEARCH_PAD))
                if nf == 0:
                    ndx, ndy, sc, nf = follow_cell(
                        nrm, ldx, ldy, int(Config.AUTO_CELL_RECOVER_PAD))
                if nf > 0:
                    ldx, ldy = ndx, ndy
                cdx[i], cdy[i], cscore[i] = ldx, ldy, sc
                prof[i] = A._strip(nrm, cell, geo, cdx[i], cdy[i]).mean(axis=1)

                # -- probe tip --
                small = cv2.resize(nrm, (w // 2, h // 2),
                                   interpolation=cv2.INTER_AREA)
                if prev_small is not None:
                    chg[i] = float(np.mean(np.abs(small - prev_small)))
                prev_small = small
                pmask = small < dark_frac
                if probe is not None:
                    m = None
                    if np.isfinite(pcx):
                        m = A._measure_probe(nrm, probe, pcx, pcy)
                    if m is None and i % 5 == 0:
                        p2 = A.find_probe(nrm)
                        if p2 is not None:
                            m = A._measure_probe(nrm, probe, *p2["center"])
                    if m is not None:
                        pcx, pcy = m[0], m[1]
                        pxs[i], pys[i], psharp[i] = m
                        dmin = float(np.min(np.hypot(
                            cont[:, 0] + cdx[i] - pcx,
                            cont[:, 1] + cdy[i] - pcy)))
                        gap[i] = dmin - float(probe["radius"])
                yy = np.clip(((cys + cdy[i]) * 0.5).astype(np.int32),
                             0, pmask.shape[0] - 1)
                xx = np.clip(((cxs + cdx[i]) * 0.5).astype(np.int32),
                             0, pmask.shape[1] - 1)
                overlap[i] = float(pmask[yy, xx].mean())

                n_done = i + 1
                if progress_callback is not None and (
                        i % 10 == 0 or i == n_total - 1):
                    if progress_callback(i + 1, n_total):
                        return None

            if n_done < 3:
                raise RuntimeError("Too few readable frames in the range.")
            sl = slice(0, n_done)
            frames_idx, t_ms, prof = frames_idx[sl], t_ms[sl], prof[sl]
            cdx, cdy, cscore = cdx[sl], cdy[sl], cscore[sl]
            pxs, pys, psharp = pxs[sl], pys[sl], psharp[sl]
            gap, overlap = gap[sl], overlap[sl]

            # v1.9.4: frames the file would not give us (damaged spot) were
            # stood in for by the last good picture: leave them out.
            bad_rel = np.array(sorted({int(b_) - start_frame
                                       for b_ in getattr(cap, "bad_frames",
                                                         [])
                                       if 0 <= int(b_) - start_frame
                                       < n_done}), dtype=np.int64)
            if bad_rel.size:
                cscore[bad_rel] = 0.0
                pxs[bad_rel] = np.nan
                pys[bad_rel] = np.nan
                psharp[bad_rel] = np.nan
                gap[bad_rel] = np.nan
                t_ms[bad_rel] = np.nan
            unread_ranges = []
            for b_ in bad_rel.tolist():
                if unread_ranges and b_ == unread_ranges[-1][1] + 1:
                    unread_ranges[-1][1] = b_
                else:
                    unread_ranges.append([b_, b_])
            unread_ranges = [(start_frame + a_, start_frame + c_)
                             for a_, c_ in unread_ranges]
            read_short = n_done < n_total

            # v1.9.5: repeated frames. Some recordings (screen or camera
            # software) store each picture twice; the copy shows no motion,
            # so frame-by-frame speeds would flick between 'moving' and
            # 'still'. Copies are found from how little the picture changes
            # and are dropped; the real pictures keep their real times.
            chg = chg[sl]
            if bad_rel.size:
                chg[bad_rel] = np.nan
            rep = A.find_repeats(chg)
            n_rep = int(rep.sum())
            if n_rep:
                keep = ~rep
                frames_idx, t_ms, prof = (frames_idx[keep], t_ms[keep],
                                          prof[keep])
                cdx, cdy, cscore = cdx[keep], cdy[keep], cscore[keep]
                pxs, pys, psharp = pxs[keep], pys[keep], psharp[keep]
                gap, overlap = gap[keep], overlap[keep]
                span = max(3, int(keep[:span].sum()))
                n_done = int(keep.sum())
                pos = np.cumsum(keep) - 1           # old index -> new index
                bad_rel = np.unique(pos[bad_rel[keep[bad_rel]]]) \
                    if bad_rel.size else bad_rel

            # ---------- time axis (real timestamps when sane) ----------
            t_nom = frames_idx / fps
            t = t_ms / 1000.0
            fin_t = np.isfinite(t)
            if 0 < bad_rel.size and fin_t.sum() >= max(3, 0.5 * t.size):
                t = np.interp(np.arange(t.size), np.nonzero(fin_t)[0],
                              t[fin_t])
            good = (np.all(np.isfinite(t)) and np.all(np.diff(t) >= -1e-6)
                    and (t[-1] - t[0]) > 0)
            agree = good and (abs((t[-1] - t[0]) - (t_nom[-1] - t_nom[0]))
                              <= 0.1 * (t_nom[-1] - t_nom[0]) + 0.5)
            if good and not agree:
                # v1.9.4: the header fps is wrong (e.g. a variable-rate
                # phone/screen recording); the time stamps are the truth
                # as long as they look like a real frame rate.
                f_ts = (t.size - 1) / float(t[-1] - t[0])
                good = bool(0.5 <= f_ts <= 20000 and
                            np.mean(np.diff(t) > 0) > 0.9)
            time_source = ("video timestamps" if good and agree else
                           f"video timestamps (the file header says "
                           f"{fps:.2f} fps, the frames say "
                           f"{(t.size - 1) / float(t[-1] - t[0]):.2f})"
                           if good else "frame / fps")
            if not good:
                t = t_nom.copy()

            # ---------- length change along the long axis ----------
            # Each end of the cell is followed separately: the piece of the
            # reference profile around that end is found again on every
            # frame (sub-pixel). Length change = how much the two ends
            # moved towards / away from each other. (Checked against a
            # direct end-to-end measurement: same result within 0.2 %.)
            H = geo["H"]
            L = float(cell["length_px"])
            hw, maxs = geo["hw"], geo["maxs"]
            P = A._highpass_rows(prof, max(9, 0.22 * H))
            n_ref = max(3, min(n_done, span))
            ref_prof = np.median(P[:n_ref], axis=0)
            centers = [hw + maxs, H - 1 - hw - maxs]
            if centers[1] - centers[0] < 2 * hw:
                raise RuntimeError(
                    "The detected cell is too small to measure its "
                    "length change.")
            shifts, scores = A._local_shifts(P, ref_prof, centers, hw, maxs)
            min_conf = float(Config.AUTO_END_MIN_CONF)
            ok_len = ((scores[:, 0] >= min_conf) & (scores[:, 1] >= min_conf)
                      & (cscore >= conf))
            length_pct = np.where(
                ok_len, 100.0 * (shifts[:, 1] - shifts[:, 0]) / L, np.nan)
            # movement of each end in the picture, along the long axis
            ax_, ay_ = cell["axis"]
            along = cdx * ax_ + cdy * ay_
            end_a = np.where(ok_len, shifts[:, 0] + along, np.nan)
            end_b = np.where(ok_len, shifts[:, 1] + along, np.nan)
            if abs(ay_) >= abs(ax_):
                end_names = ("upper", "lower")
            else:
                end_names = ("left", "right") if ax_ > 0 else ("right", "left")

            ev = A.detect_events(t, length_pct)
            # v1.9: sarcomere length from the striations, every frame
            try:
                slr = A.sarcomere_lengths(prof, geo, L, um, n_ref)
            except Exception as _e:
                slr = {"sl_px": np.full(n_done, np.nan), "sl_um": None,
                       "sl_snr": np.zeros(n_done), "sl_found": False,
                       "sl_ref_px": float("nan"),
                       "sl_reason": f"sarcomere analysis failed: {_e}"}
            if bad_rel.size:
                try:
                    slr["sl_px"] = np.asarray(slr["sl_px"], dtype=float)
                    slr["sl_px"][bad_rel] = np.nan
                    if slr.get("sl_um") is not None:
                        slr["sl_um"] = np.asarray(slr["sl_um"], dtype=float)
                        slr["sl_um"][bad_rel] = np.nan
                except Exception:
                    pass
            # time during which the length could actually be measured
            okf = np.isfinite(length_pct)
            dts = np.diff(t, prepend=t[0])
            measured_s = float(dts[okf].sum())
            gap_lim = 0.5 * float(Config.AUTO_EVENT_BASELINE_S)
            for kk in range(1, len(ev["events"])):
                i_a = ev["events"][kk - 1]["index"]
                i_b = ev["events"][kk]["index"]
                if float(dts[i_a:i_b + 1][~okf[i_a:i_b + 1]].sum()) > gap_lim:
                    # cell hidden in between: an unseen beat is possible
                    ev["events"][kk]["interval"] = float("nan")
            # how far each end travels in every contraction
            if ev["events"]:
                dtm = float(np.median(np.diff(t))) if n_done > 1 else 1.0
                dtm = dtm if dtm > 1e-6 else 1.0 / fps
                wb = float(Config.AUTO_EVENT_BASELINE_S) / dtm
                fa = np.where(np.isfinite(end_a), end_a, 0.0)
                fb = np.where(np.isfinite(end_b), end_b, 0.0)
                ba = A._rolling_median(fa, wb)
                bb = A._rolling_median(fb, wb)
                for e in ev["events"]:
                    j = e["index"]
                    e["end_a_move_px"] = float(fa[j] - ba[j])
                    e["end_b_move_px"] = float(fb[j] - bb[j])

            # ---------- probe / compression verdict ----------
            probe_found = np.isfinite(pxs)
            disp = np.full(n_done, np.nan)
            sharp_rel = np.full(n_done, np.nan)
            max_disp = float("nan")
            focus_ratio = float("nan")
            if probe is not None and probe_found.any():
                i0 = int(np.nonzero(probe_found)[0][0])
                disp = np.hypot(pxs - pxs[i0], pys - pys[i0])
                max_disp = float(np.nanmax(disp))
                s0 = float(np.nanmedian(psharp[:max(3, n_ref)]))
                if np.isfinite(s0) and s0 > 0:
                    sharp_rel = psharp / s0
                    lo_, hi_ = np.nanpercentile(sharp_rel, [5, 95])
                    focus_ratio = float(hi_ / lo_) if lo_ > 0 else float("nan")
            cell_disp = np.hypot(cdx, cdy)
            dt_med = float(np.median(np.diff(t))) if n_done > 1 else 1.0
            dt_med = dt_med if dt_med > 1e-6 else 1.0 / fps
            # slow part only: a contraction also shifts the cell briefly
            max_cell_disp = float(np.max(A._rolling_median(
                cell_disp, float(Config.AUTO_EVENT_BASELINE_S) / dt_med)))
            max_overlap = float(np.max(overlap))
            ov_idx = np.nonzero(
                overlap >= float(Config.AUTO_OVERLAP_MIN_FRAC))[0]
            t_first_overlap = float(t[ov_idx[0]]) if ov_idx.size else None
            min_gap = float(np.nanmin(gap)) if np.isfinite(gap).any() \
                else float("nan")
            move_thr = float(Config.AUTO_PROBE_MOVE_MIN_PX)
            probe_moved = bool(np.isfinite(max_disp) and max_disp > move_thr)
            cell_moved = bool(max_cell_disp > move_thr)

            def _len(v_px, nd=1):
                if um is not None:
                    return f"{v_px * um:.{nd}f} µm"
                return f"{v_px:.{nd}f} px"

            if probe is None:
                verdict = ("No probe found in the picture, so compression "
                           "cannot be judged.")
                code = "no_probe"
            elif (t_first_overlap is not None and not probe_moved
                  and t_first_overlap <= float(t[0]) + 0.5):
                # v1.9.6: probe resting on top of the cell all along
                verdict = (f"The probe rests on top of the cell for the "
                           f"whole range (covering up to "
                           f"{100 * max_overlap:.0f}% of it; the cell under "
                           f"it is rebuilt) and did not move (max "
                           f"{max_disp:.2f} px): no pressing movement in "
                           "this range.")
                code = "probe_on_cell"
            elif t_first_overlap is not None:
                verdict = (f"The probe is over the cell from t = "
                           f"{t_first_overlap:.2f} s (up to "
                           f"{100 * max_overlap:.0f}% of the cell covered): "
                           "compression is possible in this range.")
                code = "probe_over_cell"
            elif probe_moved or cell_moved:
                verdict = ("Probe and cell moved relative to the picture "
                           f"(probe up to {_len(max_disp)}, cell up to "
                           f"{_len(max_cell_disp)}) but the probe never "
                           f"came over the cell (closest {_len(min_gap)}).")
                code = "moved_no_contact"
            else:
                verdict = ("No compression in this range: the probe did "
                           f"not move (max {max_disp:.2f} px) and stays "
                           f"{_len(min_gap)} from the cell.")
                code = "no_compression"
            if (code != "probe_over_cell" and np.isfinite(focus_ratio)
                    and focus_ratio > 1.4):
                verdict += (" The probe edge changes sharpness by "
                            f"×{focus_ratio:.2f}, so it probably moved up "
                            "or down.")

            # ---------- pictures for the summary figure ----------
            ref_u8 = np.clip(ref * bg0, 0, 255).astype(np.uint8)
            peak_u8 = None
            peak_frame = None
            events = ev["events"]
            if events:
                big = max(events, key=lambda e: e["amp_pct"])
                peak_frame = int(frames_idx[big["index"]])
                cap.set(cv2.CAP_PROP_POS_FRAMES, peak_frame)
                ok, frame = cap.read()
                if ok and frame is not None:
                    peak_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            amps = np.array([e["amp_pct"] for e in events], dtype=float)
            ivals = np.array([e["interval"] for e in events[1:]], dtype=float)
            ivals = ivals[np.isfinite(ivals)]
            dur = float(t[-1] - t[0]) if n_done > 1 else 0.0
            return {
                "video_path": video_path, "fps": fps,
                "start_frame": int(start_frame),
                "end_frame": int(frames_idx[-1]),
                "n_frames": int(n_done), "duration_s": dur,
                "time_source": time_source,
                "unread_ranges": unread_ranges,
                "n_repeats": n_rep,
                "n_unread": int(bad_rel.size),
                "read_short": bool(read_short),
                "n_requested": int(n_total),
                "frames_total": int(total),
                "um_per_px": um, "scale_source": scale_source,
                "probe": probe, "cell": cell, "strip": geo,
                "sl_px": slr["sl_px"], "sl_um": slr["sl_um"],
                "sl_snr": slr["sl_snr"], "sl_found": slr["sl_found"],
                "sl_ref_px": slr["sl_ref_px"],
                "sl_reason": slr.get("sl_reason", ""),
                "sl_snr_ref": slr.get("sl_snr_ref", float("nan")),
                "frames": frames_idx, "t": t,
                "length_pct": length_pct,
                "end_a_px": end_a, "end_b_px": end_b,
                "end_names": end_names,
                "end_score": scores.min(axis=1),
                "baseline_pct": ev["baseline"],
                "noise_pct": ev["sigma"], "event_thr_pct": ev["thr"],
                "events": events,
                "n_events": len(events),
                "measured_s": measured_s,
                "rate_per_min": (60.0 * len(events) / measured_s
                                 if measured_s > 0 else float("nan")),
                "amp_mean_pct": float(amps.mean()) if amps.size else float("nan"),
                "amp_sd_pct": (float(amps.std(ddof=1)) if amps.size > 1
                               else float("nan")),
                "amp_max_pct": float(amps.max()) if amps.size else float("nan"),
                "interval_mean_s": (float(ivals.mean()) if ivals.size
                                    else float("nan")),
                "interval_sd_s": (float(ivals.std(ddof=1)) if ivals.size > 1
                                  else float("nan")),
                "valid_frac": float(np.mean(np.isfinite(length_pct))),
                "cell_dx": cdx, "cell_dy": cdy, "cell_score": cscore,
                "max_cell_disp_px": max_cell_disp,
                "probe_x": pxs, "probe_y": pys, "probe_disp_px": disp,
                "probe_sharp_rel": sharp_rel, "focus_ratio": focus_ratio,
                "max_probe_disp_px": max_disp,
                "gap_px": gap, "min_gap_px": min_gap,
                "overlap": overlap, "max_overlap": max_overlap,
                "t_first_overlap": t_first_overlap,
                "verdict": verdict, "verdict_code": code,
                "ref_u8": ref_u8, "peak_u8": peak_u8,
                "peak_frame": peak_frame,
            }
        finally:
            cap.release()

    # ------------------------------ reporting ------------------------------

    @staticmethod
    def suggest_points4(res: dict) -> np.ndarray:
        """Four spots inside the cell near its corners, for the 4-point
        tracker (so nothing has to be clicked)."""
        cell = res["cell"]
        ax, ay = cell["axis"]
        bx, by = -ay, ax
        cx, cy = cell["center"]
        a = 0.38 * float(cell["length_px"])
        b = 0.32 * float(cell["width_mean_px"])
        pts = []
        for sa, sb in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
            pts.append((cx + sa * a * ax + sb * b * bx,
                        cy + sa * a * ay + sb * b * by))
        return np.array(pts, dtype=np.float32)

    # ------------------------------ cell from your outline ------------------------------

    @staticmethod
    def _cell_dict(mask: np.ndarray, n_candidates: int = 1) -> Optional[dict]:
        """Measurements of a cell mask (same keys as find_cell)."""
        ys, xs = np.nonzero(mask)
        if xs.size < 20:
            return None
        cx, cy = float(xs.mean()), float(ys.mean())
        cov = np.cov(np.vstack([xs - cx, ys - cy]))
        evals, evecs = np.linalg.eigh(cov)
        ax, ay = float(evecs[0, 1]), float(evecs[1, 1])
        if ay < 0 or (abs(ay) < 1e-9 and ax < 0):
            ax, ay = -ax, -ay
        p = (xs - cx) * ax + (ys - cy) * ay
        q = -(xs - cx) * ay + (ys - cy) * ax
        length = float(p.max() - p.min() + 1.0)
        wmax = float(q.max() - q.min() + 1.0)
        pc = 0.5 * float(p.max() + p.min())
        qc = 0.5 * float(q.max() + q.min())
        ccx = cx + pc * ax - qc * ay
        ccy = cy + pc * ay + qc * ax
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        contour = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(
            np.float32)
        area = float(mask.sum())
        return {
            "mask": mask, "contour": contour, "ys": ys, "xs": xs,
            "center": (ccx, ccy), "centroid": (cx, cy), "axis": (ax, ay),
            "length_px": length, "width_max_px": wmax,
            "width_mean_px": area / max(1.0, length), "area_px": area,
            "angle_deg": float(math.degrees(math.atan2(ay, ax)) % 180.0),
            "bbox": (int(xs.min()), int(ys.min()),
                     int(xs.max()) + 1, int(ys.max()) + 1),
            "n_candidates": int(n_candidates),
        }

    @staticmethod
    def cell_from_outline(norm: np.ndarray, poly, probe: Optional[dict] = None,
                          snap: bool = True) -> Optional[dict]:
        """v1.9: the cell = the outline you drew (needed when part of the
        cell is under the probe, which the automatic search cuts away).

        Where the cell is visible (not under the probe) the outline is
        snapped to the cell's real edge: only pixels inside your outline
        (grown by AUTO_HINT_SNAP_PX) with strong edge / texture are kept,
        holes filled. Under the probe nothing can be seen, so your outline
        is used as drawn there. If snapping keeps too little of what you
        drew, your outline is used everywhere."""
        h, w = norm.shape[:2]
        pts = np.round(np.asarray(poly, dtype=np.float64)).astype(np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return None
        drawn = np.zeros((h, w), np.uint8)
        cv2.fillPoly(drawn, [pts.reshape(-1, 1, 2)], 1)
        drawn = drawn.astype(bool)
        if int(drawn.sum()) < 50:
            return None
        mask = drawn
        snapped_ok = False
        if snap:
            pm = probe["mask"] if probe is not None else np.zeros((h, w), bool)
            # where the probe is (grown a little: its blurred edge is not cell)
            pr = int(round(0.25 * float(probe["radius"]))) if probe is not None \
                else 0
            pm_g = (cv2.dilate(pm.astype(np.uint8), np.ones(
                (2 * pr + 1, 2 * pr + 1), np.uint8)) > 0) if pr > 0 else pm
            g = int(Config.AUTO_HINT_SNAP_PX)
            zone = cv2.dilate(drawn.astype(np.uint8),
                              cv2.getStructuringElement(
                                  cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
                              ) > 0
            s1 = float(Config.AUTO_CELL_FINE_SIGMA)
            f1 = cv2.GaussianBlur(norm, (0, 0), s1)
            gx = cv2.Sobel(f1, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(f1, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            gm = np.hypot(gx, gy)
            bg = ~zone & ~pm_g
            gmed, gsig = AutoCellAnalysis._robust(gm[bg][::7]) if bg.any() \
                else AutoCellAnalysis._robust(gm[::7, ::7])
            inside = drawn & ~pm_g
            ref90 = float(np.percentile(gm[inside], 90)) if inside.any() \
                else 0.0
            gthr = max(float(Config.AUTO_CELL_MIN_GRAD),
                       gmed + float(Config.AUTO_CELL_K_MAD) * gsig,
                       0.3 * ref90)
            fine = (gm > gthr) & zone & ~pm_g
            fine = cv2.morphologyEx(
                fine.astype(np.uint8), cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            # under the probe: your outline is all we have
            fine = (fine > 0) | (drawn & pm_g)
            fine = AutoCellAnalysis._fill_holes(fine)
            fine = cv2.erode(fine.astype(np.uint8), cv2.getStructuringElement(
                cv2.MORPH_CROSS, (3, 3))) > 0
            n2, lab2, st2, _ = cv2.connectedComponentsWithStats(
                fine.astype(np.uint8), 8)
            if n2 > 1:
                # the piece that overlaps your outline most
                best, best_ov = 0, 0
                for k2 in range(1, n2):
                    ov = int((drawn & (lab2 == k2)).sum())
                    if ov > best_ov:
                        best, best_ov = k2, ov
                cand = lab2 == best
                vis_drawn = int((drawn & ~pm_g).sum())
                vis_cand = int((cand & ~pm_g).sum())
                if (vis_drawn == 0 or vis_cand >= 0.6 * vis_drawn) and \
                        int(cand.sum()) <= 1.6 * int(drawn.sum()):
                    mask = cand
                    snapped_ok = True
        cell = AutoCellAnalysis._cell_dict(mask, 1)
        if cell is None:
            return None
        cell["source"] = "outline"
        cell["snapped"] = bool(snapped_ok)
        cell["outline"] = np.asarray(poly, dtype=np.float32)
        return cell

    # ------------------------------ sarcomere length ------------------------------

    @staticmethod
    def sarcomere_lengths(prof: np.ndarray, geo: dict, length_px: float,
                          um_per_px: Optional[float], n_ref: int) -> dict:
        """v1.9: sarcomere length on every frame from the striation pattern.

        prof[i] is the cell's brightness along its long axis on frame i
        (averaged across the width, following the cell). The sarcomeres
        make this profile ripple with a period = sarcomere length. The
        middle part of the cell (ends excluded) is detrended, Hann-windowed
        and Fourier transformed; the strongest peak in the sarcomere band
        gives the period, refined to a small fraction of a pixel (zero
        padding + parabola on the log power). The band is first found on
        the reference frames, then each frame is searched within ±25 % of
        that period, so the estimate cannot jump to a harmonic.
        Returns sl_px / sl_um per frame (NaN where the pattern is too weak),
        the signal-to-noise of the peak, and the resting period."""
        C = Config
        n, H = prof.shape
        out = {"sl_px": np.full(n, np.nan), "sl_um": None,
               "sl_snr": np.zeros(n), "sl_ref_px": float("nan"),
               "sl_found": False, "sl_reason": ""}
        m = int(geo.get("margin", 0))
        L = float(length_px)
        a = int(round(m + C.AUTO_SL_EDGE_FRAC * L))
        b = int(round(m + (1.0 - C.AUTO_SL_EDGE_FRAC) * L))
        a, b = max(0, a), min(H, b)
        if b - a < 24:
            out["sl_reason"] = "cell too short to measure sarcomeres"
            return out
        nyq = 2.05
        if um_per_px is not None and um_per_px > 0:
            pmin = float(C.AUTO_SL_MIN_UM) / um_per_px
            pmax = float(C.AUTO_SL_MAX_UM) / um_per_px
        else:
            pmin, pmax = nyq, 12.0
        if pmax < nyq * 1.05:
            out["sl_reason"] = (
                f"pixels too large ({um_per_px:.2f} µm/px): a sarcomere "
                "covers fewer than 2 pixels and cannot be resolved")
            return out
        pmin = max(pmin, nyq)
        raw = prof[:, a:b].astype(np.float64)
        w = AutoCellAnalysis._odd(max(5.0, 2.5 * pmax))
        min_run = int(max(16, 5 * pmax))
        def clean_runs(row):
            """Stretches of the profile NOT covered by the probe: darker
            than AUTO_SL_DARK_FRAC x the cell's usual brightness counts as
            covered (judged on the profile smoothed over ~3 sarcomeres, so
            texture and striations do not count), grown 3 px for the
            shadow's soft edge. -> list of (start, end)."""
            ks = int(max(5, round(3 * pmax))) | 1
            sm = np.convolve(np.pad(row, ks // 2, mode="edge"),
                             np.ones(ks) / ks, mode="valid")
            ref_lvl = float(np.percentile(sm, 75))
            bad = ~(sm >= float(C.AUTO_SL_DARK_FRAC) * ref_lvl)
            if not bad.any():
                return [(0, row.shape[0])]
            bad = np.convolve(bad.astype(float), np.ones(7), "same") > 0
            runs, i0 = [], None
            for i_, g_ in enumerate(np.r_[~bad, False]):
                if g_ and i0 is None:
                    i0 = i_
                elif not g_ and i0 is not None:
                    runs.append((i0, i_))
                    i0 = None
            return runs

        def prep(row):
            """every clear stretch long enough, detrended and Hann
            windowed -> list of 1-D arrays (empty = nothing usable)"""
            segs = []
            for r0, r1 in clean_runs(row):
                if r1 - r0 < min_run:
                    continue
                x = row[r0:r1]
                h = w // 2
                xp = np.pad(x, (h, h), mode="reflect")
                cs = np.cumsum(np.r_[0.0, xp])
                x = x - (cs[w:] - cs[:-w]) / float(w)
                segs.append(x * np.hanning(x.shape[0]))
            return segs

        nfft = 1 << int(math.ceil(math.log2(max(64, 32 * raw.shape[1]))))
        f = np.fft.rfftfreq(nfft)

        def power_of(segs):
            """summed power spectrum of all clear stretches"""
            pw = None
            for x in segs:
                p_ = np.abs(np.fft.rfft(x, n=nfft)) ** 2
                pw = p_ if pw is None else pw + p_
            return pw

        def peak(pw, f_lo, f_hi):
            """-> (period px, snr) of the strongest peak in [f_lo, f_hi]."""
            sel = np.nonzero((f >= f_lo) & (f <= f_hi))[0]
            if sel.size < 3:
                return float("nan"), 0.0
            j = sel[int(np.argmax(pw[sel]))]
            if j <= 0 or j >= f.size - 1:
                return float("nan"), 0.0
            y0, y1, y2 = (math.log(pw[j - 1] + 1e-30),
                          math.log(pw[j] + 1e-30),
                          math.log(pw[j + 1] + 1e-30))
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            fp = f[j] + float(np.clip(d, -0.5, 0.5)) * (f[1] - f[0])
            wide = np.nonzero((f >= 0.5 * f_lo) & (f <= min(0.5, 1.5 * f_hi)))[0]
            noise = float(np.median(pw[wide])) if wide.size else 0.0
            snr = float(pw[j] / noise) if noise > 0 else 0.0
            return (1.0 / fp if fp > 0 else float("nan")), snr

        nr = int(max(1, min(n, n_ref)))
        acc, cnt = None, 0
        for i in range(nr):
            x = prep(raw[i])
            if not x:
                continue
            pw = power_of(x) / max(1.0, float(sum((v * v).sum()
                                                  for v in x)))
            acc = pw if acc is None else acc + pw
            cnt += 1
        if acc is None:
            out["sl_reason"] = ("the visible part of the cell is too short "
                                "(the probe covers too much of it)")
            return out
        p_ref, snr_ref = peak(acc / cnt, 1.0 / pmax, 1.0 / pmin)
        if not np.isfinite(p_ref) or snr_ref < float(C.AUTO_SL_REF_SNR):
            out["sl_reason"] = ("no clear striation pattern on the "
                                f"reference frames (peak/noise {snr_ref:.1f}, "
                                f"needs {C.AUTO_SL_REF_SNR:g})")
            out["sl_snr_ref"] = snr_ref
            return out
        lo, hi = 1.0 / (1.25 * p_ref), 1.0 / max(nyq, 0.75 * p_ref)
        sl = np.full(n, np.nan)
        snr = np.zeros(n)
        for i in range(n):
            x = prep(raw[i])
            if not x:
                continue
            p, q = peak(power_of(x), lo, hi)
            snr[i] = q
            sl[i] = p
        sl[~(snr >= float(C.AUTO_SL_MIN_SNR))] = np.nan
        # single-frame spikes (noise picked a neighbouring peak): a value
        # more than AUTO_SL_SPIKE_FRAC of the resting period away from the
        # median of its ±2 neighbours is dropped. Real contractions change
        # sarcomere length far less than that from one frame to the next.
        if n >= 5:
            pad = np.pad(sl, 2, mode="edge")
            win = np.lib.stride_tricks.sliding_window_view(pad, 5)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med5 = np.nanmedian(win, axis=1)
            spike = np.abs(sl - med5) > float(C.AUTO_SL_SPIKE_FRAC) * p_ref
            sl[spike & np.isfinite(med5)] = np.nan
        # resting period reported = median of the measured reference
        # frames (the band search above can stop at its edge)
        ref_meas = sl[:nr][np.isfinite(sl[:nr])]
        if ref_meas.size:
            p_ref = float(np.median(ref_meas))
        out.update(sl_px=sl, sl_snr=snr, sl_ref_px=float(p_ref),
                   sl_snr_ref=snr_ref, sl_found=True,
                   sl_window_px=(a - m, b - m))
        if um_per_px is not None and um_per_px > 0:
            out["sl_um"] = sl * float(um_per_px)
        return out

    # ------------------------------ speed over time ------------------------------

    @staticmethod
    def speed_series(res: dict, smooth_frames: Optional[int] = None) -> dict:
        """v1.9: speed over time, in display units.

        rate = d(cell length)/dt from a Savitzky-Golay derivative
        (`smooth_frames` wide, polynomial order 4 from 5 frames up, else
        2; 1 or 3 = plain central difference). Below 0 the
        cell is shortening, above 0 it is re-lengthening. The same is done
        for each end's position along the long axis. Frames where the
        length could not be measured stay empty (the derivative is never
        taken across a gap). Also returns, per contraction, the peak
        shortening and relaxation speed read from this curve."""
        if smooth_frames is None:
            smooth_frames = int(Config.AUTO_SPEED_SMOOTH_FRAMES)
        n = int(max(1, smooth_frames))
        if n > 1 and n % 2 == 0:
            n += 1
        um = res.get("um_per_px")
        k = um if um is not None else 1.0
        L = float(res["cell"]["length_px"])
        t = np.asarray(res["t"], dtype=np.float64)
        dts = np.diff(t)
        dts = dts[np.isfinite(dts) & (dts > 0)]
        dt = float(np.median(dts)) if dts.size else 1.0 / float(
            res.get("fps") or Config.DEFAULT_FPS)

        # order 4 keeps the peak of a fast upstroke (order 2 over the same
        # window cut a 3-frame upstroke's peak speed by a third in tests)
        order = int(Config.AUTO_SPEED_POLYORDER) if n >= 5 else 2

        def deriv(y):
            y = np.asarray(y, dtype=np.float64)
            if n >= 3:
                return Signal.smooth_derivative(y, n, order, dt)
            return Signal.plain_derivative(y, dt)

        pct = np.asarray(res["length_pct"], dtype=np.float64)
        length_u = pct / 100.0 * L * k
        rate_pct = deriv(pct)
        rate_pct_raw = Signal.plain_derivative(pct, dt)
        ea = np.asarray(res["end_a_px"], dtype=np.float64) * k
        eb = np.asarray(res["end_b_px"], dtype=np.float64) * k
        out = {
            "t": t, "dt": dt, "smooth": n, "order": order,
            "length_pct": pct, "length_u": length_u,
            "rate_pct": rate_pct, "rate_u": rate_pct / 100.0 * L * k,
            "rate_pct_raw": rate_pct_raw,
            "rate_u_raw": rate_pct_raw / 100.0 * L * k,
            "end_a_u": ea, "end_b_u": eb,
            "end_a_speed": deriv(ea), "end_b_speed": deriv(eb),
            "probe_u": np.asarray(res["probe_disp_px"], dtype=np.float64) * k,
            "gap_u": np.asarray(res["gap_px"], dtype=np.float64) * k,
        }
        # per contraction: fastest shortening (most negative rate) between
        # onset and peak, fastest relaxation between peak and end
        vs, vr, ivs, ivr = [], [], [], []
        N = t.shape[0]
        ru = out["rate_u"]
        for e in res["events"]:
            ip = int(e["index"])
            a = (int(np.searchsorted(t, e["t_onset"])) - 1
                 if e.get("t_onset") is not None else max(0, ip - int(
                     round(1.0 / dt))))
            b = (int(np.searchsorted(t, e["t_end"])) + 1
                 if e.get("t_end") is not None else min(N - 1, ip + int(
                     round(2.0 / dt))))
            a = int(np.clip(a, 0, ip))
            b = int(np.clip(b, ip, N - 1))
            seg1, seg2 = ru[a:ip + 1], ru[ip:b + 1]
            if np.isfinite(seg1).any():
                j = a + int(np.nanargmin(seg1))
                vs.append(float(-ru[j]))
                ivs.append(j)
            else:
                vs.append(float("nan"))
                ivs.append(ip)
            if np.isfinite(seg2).any():
                j = ip + int(np.nanargmax(seg2))
                vr.append(float(ru[j]))
                ivr.append(j)
            else:
                vr.append(float("nan"))
                ivr.append(ip)
        out.update(ev_vshort=vs, ev_vrelax=vr, ev_ivshort=ivs, ev_ivrelax=ivr)

        # ---- v1.9: sarcomere length (µm, or px without a scale) ----
        slp = res.get("sl_px")
        if slp is None:
            slp = np.full(N, np.nan)
        sl_u = np.asarray(slp, dtype=np.float64) * k
        sl_base = np.full(N, np.nan)
        sl_pct = np.full(N, np.nan)
        okv = np.isfinite(sl_u)
        if int(okv.sum()) >= 3:
            fill = np.interp(t, t[okv], sl_u[okv])
            sl_base = AutoCellAnalysis._rolling_median(
                fill, float(Config.AUTO_EVENT_BASELINE_S) / dt)
            sl_pct = 100.0 * (sl_u / sl_base - 1.0)
        sl_rest, sl_min, sl_short, sl_imin = [], [], [], []
        for j, e in enumerate(res["events"]):
            ip = int(e["index"])
            a = (int(np.searchsorted(t, e["t_onset"])) - 1
                 if e.get("t_onset") is not None else max(0, ip - int(
                     round(1.0 / dt))))
            b = (int(np.searchsorted(t, e["t_end"])) + 1
                 if e.get("t_end") is not None else min(N - 1, ip + int(
                     round(2.0 / dt))))
            a, b = int(np.clip(a, 0, ip)), int(np.clip(b, ip, N - 1))
            seg = sl_u[a:b + 1]
            if np.isfinite(seg).any() and np.isfinite(sl_base[ip]):
                jm = a + int(np.nanargmin(seg))
                rest = float(sl_base[ip])
                sl_rest.append(rest)
                sl_min.append(float(sl_u[jm]))
                sl_short.append(100.0 * (rest - float(sl_u[jm])) / rest)
                sl_imin.append(jm)
            else:
                sl_rest.append(float("nan"))
                sl_min.append(float("nan"))
                sl_short.append(float("nan"))
                sl_imin.append(ip)
        out.update(sl_u=sl_u, sl_pct=sl_pct, sl_base=sl_base,
                   sl_found=bool(res.get("sl_found", False)),
                   ev_sl_rest=sl_rest, ev_sl_min=sl_min,
                   ev_sl_short=sl_short, ev_sl_imin=sl_imin)
        return out

    # ------------------------------ method description ------------------------------

    @staticmethod
    def method_text(res: dict, smooth_frames: Optional[int] = None):
        """v1.9: plain-language description of the algorithm, filled in
        with the numbers used for THIS result. -> [(kind, text)] where kind
        is 'h1', 'h2', 'p' or 'num'."""
        C = Config
        um = res.get("um_per_px")
        cell = res["cell"]
        probe = res.get("probe")
        fps = float(res.get("fps") or C.DEFAULT_FPS)
        n = int(smooth_frames or C.AUTO_SPEED_SMOOTH_FRAMES)
        L = float(cell["length_px"])
        geo = res.get("strip", {})
        unit = "µm" if um is not None else "px"
        k = um if um is not None else 1.0
        o = [("h1", "How the automatic analysis works"),
             ("p", "Everything below is done by the program from the video "
                   "alone; nothing is clicked. Numbers in grey are the "
                   "values used for this video "
                   f"({Path(res['video_path']).name}, frames "
                   f"{res['start_frame']}–{res['end_frame']}, "
                   f"{res['n_frames']} frames at {fps:.2f} fps).")]

        o += [("h2", "1. Brightness made comparable"),
              ("p", "Every frame is turned to grey and divided by its own "
                    "background level (the median grey value), so lamp "
                    "flicker or automatic exposure does not change any "
                    "measurement. The background is then ≈ 1.0.")]

        o += [("h2", "2. A clean reference picture"),
              ("p", "The median of frames from the first "
                    f"{C.AUTO_REF_SECONDS:g} s of the range (up to "
                    f"{C.AUTO_REF_MAX_FRAMES} frames) is the reference. The "
                    "median removes noise and anything that only passes by "
                    "briefly. The cell and the probe are found on this "
                    "picture, and every later frame is compared with it — "
                    "so errors cannot add up over time.")]

        ptxt = ("The probe is the largest region darker than "
                f"{C.AUTO_PROBE_DARK_FRAC:g} × the background (at least "
                f"{100 * C.AUTO_PROBE_MIN_AREA_FRAC:.2f}% of the picture). "
                "If it touches the picture border it is treated as a rod and "
                "its round tip is its free end; otherwise as a free disk. "
                "The tip centre is refined to a fraction of a pixel from "
                "where its soft edge crosses half darkness, and the tip "
                "diameter is measured edge to edge.")
        if probe is not None and probe.get("source") == "step2":
            ptxt += (" For this video the tip is the sphere you marked in "
                     "Step 2 (its centre and size used as you set them; "
                     "it is then followed from frame to frame).")
        if probe is not None:
            ptxt += (f" Found: a {probe['kind']}, tip Ø "
                     f"{2 * probe['radius'] * k:.1f} {unit}.")
        else:
            ptxt += " No probe was found in this video."
        o += [("h2", "3. The probe and the scale"), ("p", ptxt),
              ("num", f"Scale: {res['scale_source']}"
                      + (f" → {um:.4f} µm per pixel" if um is not None
                         else " → sizes are in pixels"))]

        o += [("h2", "4. Finding the cell"),
              ("p", "A cardiomyocyte is full of fine, sharp texture "
                    "(sarcomeres, mitochondria); the background and the "
                    "blurred probe halo are not. A band-pass filter "
                    f"(σ = {C.AUTO_CELL_FINE_SIGMA:g} px vs "
                    f"{2 * C.AUTO_CELL_FINE_SIGMA:g} px) keeps that fine "
                    "texture, its local energy is averaged over "
                    f"{C.AUTO_CELL_ENERGY_SIGMA:g} px, and everything more "
                    f"than {C.AUTO_CELL_K_MAD:g} robust standard deviations "
                    "above the background texture is a candidate. Near the "
                    "probe the cut-off is raised ring by ring so its blurred "
                    "edge is not taken for cell. Holes are filled; objects "
                    f"smaller than {100 * C.AUTO_CELL_MIN_AREA_FRAC:.1f}% of "
                    "the picture (debris) are ignored. Of the remaining "
                    "objects the one nearest the probe tip (or nearest your "
                    "picked points, if any) is the cell."),
              ("p", "The outline is then tightened on the edge strength "
                    "(gradient magnitude) inside a thin zone around the "
                    "candidate, and the long axis is the main direction of "
                    "the cell's pixels (principal-component analysis). "
                    "Length and maximum width are the extents along and "
                    "across that axis; mean width = area / length."),
              ("p", "The probe lies on top of the cell, so the cell carries "
                    "on underneath it. Where the probe covers part of the "
                    "cell, the pieces seen on either side of the probe are "
                    "joined, and the hidden part is rebuilt: along the long "
                    "axis, every cross-section clear of the probe gives the "
                    "cell's two side edges, and across the covered stretch "
                    "both edges are carried straight from the last clear "
                    "section before the probe to the first one after it. "
                    "Only pixels inside the probe's footprint are added; "
                    "the visible outline is kept as found."
                    + (f" Here {100 * cell['hidden_px'] / max(1.0, cell['area_px']):.0f}%"
                       " of the cell's area was under the probe and was "
                       "rebuilt this way (dashed orange outline)."
                       if cell.get("hidden_px", 0) > 0 else
                       " Here no part of the cell was under the probe.")),
              ("p", {"outline": "For this video you drew the cell's "
                          "outline, so the search above was skipped: your "
                          "outline is the cell. Where the cell is visible it "
                          "was snapped to the real edge (strong edge "
                          "strength within "
                          f"{C.AUTO_HINT_SNAP_PX} px of your line); under the "
                          "probe, where nothing can be seen, your line is "
                          "used as drawn.",
                     "click": "For this video you clicked on the cell, so of "
                              "the candidate objects the one nearest your "
                              "click was taken.",
                     }.get(cell.get("source", "auto"),
                           "The cell was found fully automatically (no "
                           "hint given).")),
              ("num", f"Cell: length {L * k:.1f} {unit}, mean width "
                      f"{cell['width_mean_px'] * k:.1f} {unit}, long axis at "
                      f"{cell['angle_deg']:.0f}°, "
                      f"{cell.get('n_candidates', 1)} candidate(s) in view.")]

        o += [("h2", "5. Following the cell on every frame"),
              ("p", "The cell is cut into three pieces along its longer "
                    "side. Each piece of the reference picture is searched "
                    f"for within ±{C.AUTO_CELL_SEARCH_PAD} px of its last "
                    "position by normalised cross-correlation (brightness "
                    "independent), with a parabola fit for sub-pixel "
                    "position. A piece only counts if it matches with a "
                    f"score ≥ {C.AUTO_CELL_MATCH_CONF:g}; if none does, the "
                    f"search widens to ±{C.AUTO_CELL_RECOVER_PAD} px. The "
                    "cell's shift is the median of the pieces found, so the "
                    "cell can be followed when the probe hides one piece or "
                    "the stage moves.")]

        o += [("h2", "6. Length change along the long axis"),
              ("p", "On every frame an upright strip of the cell is cut "
                    "out along its long axis (following the shift from "
                    "step 5) and averaged across its width, giving a "
                    "brightness profile along the cell. Slow brightness "
                    "trends are removed from the profile. Around each END "
                    "of the cell a window of the reference profile "
                    f"(±{geo.get('hw', '?')} px) is found again on that "
                    "frame by correlation, with sub-pixel accuracy, within "
                    f"±{geo.get('maxs', '?')} px. The length change is how "
                    "far the two ends moved towards or away from each "
                    "other, as a percentage of the resting length. A frame "
                    "is only used when both ends match with a score ≥ "
                    f"{C.AUTO_END_MIN_CONF:g}."),
              ("num", f"Length measured on {100 * res['valid_frac']:.0f}% "
                      "of the frames.")]

        sl_ok = bool(res.get("sl_found"))
        sl_line = ("Resting period found: "
                   f"{res.get('sl_ref_px', float('nan')):.3f} px"
                   + (f" = {res['sl_ref_px'] * um:.3f} µm"
                      if (um is not None and sl_ok) else "")
                   + f", striation peak {res.get('sl_snr_ref', float('nan')):.0f}× "
                     "above the noise; measured on "
                   f"{100 * np.mean(np.isfinite(res['sl_px'])):.0f}% of "
                   "the frames.") if sl_ok else (
            "Not measured for this video: " + str(res.get("sl_reason", "")))
        o += [("h2", "6b. Sarcomere length"),
              ("p", "The same brightness profile along the cell (step 6) "
                    "ripples with the striations: one dark–light cycle per "
                    "sarcomere. On every frame the middle "
                    f"{100 * (1 - 2 * C.AUTO_SL_EDGE_FRAC):.0f}% of the cell "
                    "(the ends are left out) is taken; any part covered by "
                    "the probe (darker than "
                    f"{C.AUTO_SL_DARK_FRAC:g} × the cell's usual brightness) "
                    "is skipped "
                    "and the clear stretches on either side are combined. It is "
                    "detrended, tapered with a Hann window and Fourier "
                    "transformed. The strongest peak "
                    "between "
                    f"{C.AUTO_SL_MIN_UM:g} and {C.AUTO_SL_MAX_UM:g} µm gives "
                    "the sarcomere length; its position is refined far "
                    "below one pixel (zero padding + parabola fit), "
                    "because it averages over dozens of sarcomeres. The "
                    "period is found first on the reference frames; every "
                    "frame is then searched within ±25% of it, so a "
                    "harmonic can never be picked. A frame is left empty "
                    f"if its peak is weaker than {C.AUTO_SL_MIN_SNR:g}× the "
                    "noise, or if it jumps more than "
                    f"{100 * C.AUTO_SL_SPIKE_FRAC:.0f}% away from its "
                    "neighbours. Sarcomere shortening = (resting length − "
                    "shortest length in the contraction) / resting length, "
                    "with the resting length taken as a running median "
                    f"over {C.AUTO_EVENT_BASELINE_S:g} s, like the cell "
                    "length. At 10x a sarcomere spans only about 3 pixels, "
                    "so this needs the striations to be in focus; the "
                    "program says so when they are not."),
              ("num", sl_line)]

        o += [("h2", "7. Speed over time"),
              ("p", "Speed is the rate of change of the cell length, "
                    "d(length)/dt. It is computed with a Savitzky–Golay "
                    "derivative: over a sliding window of "
                    f"{n} frames a polynomial (order "
                    f"{Config.AUTO_SPEED_POLYORDER if n >= 5 else 2}) is "
                    "fitted and its slope taken at the centre frame, so the "
                    "curve is not shifted in time. A 4th-order fit is used "
                    "because the upstroke of a contraction often lasts only "
                    "a few frames: in tests a 2nd-order fit over 5 frames "
                    "read the peak shortening speed about a third too low, "
                    "the 4th-order fit within a few percent. Below zero "
                    "the cell is shortening, above zero it is "
                    "re-lengthening. The derivative is never taken across "
                    "frames where the length could not be measured. Each "
                    "end's speed is the same derivative of that end's "
                    "position along the long axis. The window can be "
                    "changed on the Graphs tab (1 or 3 = plain central "
                    "difference, 5 = accurate peaks, larger = smoother "
                    "curve but lower peaks of fast contractions)."),
              ("num", f"Speed smoothing: {n} frames = "
                      f"{1000.0 * n / fps:.0f} ms at {fps:.1f} fps.")]

        o += [("h2", "8. Counting contractions"),
              ("p", "A running baseline (median over "
                    f"{C.AUTO_EVENT_BASELINE_S:g} s) is subtracted from the "
                    "length trace, so slow drift does not count. The trace "
                    "noise is the robust standard deviation of what is "
                    "left. A contraction is a dip below the baseline that "
                    f"is at least {C.AUTO_EVENT_K_SIGMA:g} × the noise "
                    f"deep, and never less than {C.AUTO_EVENT_MIN_PCT:g}% of "
                    "the length; dips closer than "
                    f"{C.AUTO_EVENT_MERGE_S:g} s are merged. For each one: "
                    "peak = deepest point; onset and end = where the trace "
                    "is back within 10% of the peak depth; duration at 50% "
                    "= time spent below half the depth; time to peak = "
                    "onset to peak; relaxation 90% = peak to end. Peak "
                    "shortening / relaxation speed = the most negative / "
                    "most positive value of the speed curve (step 7) "
                    "between onset and peak / peak and end. An interval is "
                    "left empty when the cell was hidden long enough in "
                    "between for a beat to be missed."),
              ("num", f"Trace noise {res['noise_pct']:.3f}% → smallest "
                      f"contraction counted {res['event_thr_pct']:.2f}% of "
                      f"the length; {res['n_events']} found.")]

        o += [("h2", "9. Did the probe come over the cell?"),
              ("p", "On every frame the probe tip is re-measured near its "
                    "last position, the shortest distance from the tip "
                    "edge to the cell outline is the gap, and the share of "
                    "the cell's pixels that lie under the dark probe is "
                    "the coverage. The probe 'came over the cell' when "
                    f"coverage reaches {100 * C.AUTO_OVERLAP_MIN_FRAC:.0f}% "
                    "of the cell; it 'moved' when it travelled more than "
                    f"{C.AUTO_PROBE_MOVE_MIN_PX:g} px. A change in the "
                    "sharpness of the probe edge means it moved up or down "
                    "(out of focus)."),
              ("num", f"Result: {res['verdict']}")]

        o += [("h2", "Limits worth knowing"),
              ("p", "• The length is measured along one straight axis; a "
                    "cell that bends or turns a lot during a beat is "
                    "measured less exactly.\n"
                    "• The cell must be in focus in the first "
                    f"{C.AUTO_REF_SECONDS:g} s of the range (the reference).\n"
                    "• The time axis comes from the video's own timestamps "
                    "when they are consistent, otherwise from frame number "
                    f"/ fps (here: {res['time_source']}).\n"
                    "• Contractions shorter than a few frames cannot be "
                    "resolved; the peak speed depends on the frame rate and "
                    "on the smoothing window.")]
        return o

    @staticmethod
    def summary_lines(res: dict) -> List[str]:
        um = res["um_per_px"]
        cell = res["cell"]

        def ln(v_px, nd=1):
            return (f"{v_px * um:.{nd}f} µm" if um is not None
                    else f"{v_px:.{nd}f} px")

        L = float(cell["length_px"])
        lines = [
            f"File: {Path(res['video_path']).name}",
            (f"Range: frames {res['start_frame']}–{res['end_frame']} "
             f"({res['n_frames']} frames, {res['duration_s']:.2f} s, "
             f"time from {res['time_source']})"),
            (f"Scale: {um:.4f} µm/px  [{res['scale_source']}]"
             if um is not None else
             "Scale: not set — sizes are in pixels"),
            (f"Cell: length {ln(L)}, width {ln(cell['width_mean_px'])} mean "
             f"/ {ln(cell['width_max_px'])} max, area "
             + (f"{cell['area_px'] * um * um:.0f} µm²" if um is not None
                else f"{cell['area_px']:.0f} px²")
             + f", length/width {L / max(1e-9, cell['width_mean_px']):.1f}, "
               f"long axis at {cell['angle_deg']:.0f}°"),
        ]
        n = res["n_events"]
        if n == 0:
            lines.append(
                "Contractions: none found (smallest counted = "
                f"{res['event_thr_pct']:.2f}% of the length; trace noise "
                f"{res['noise_pct']:.3f}%).")
        else:
            a_um = (f" = {res['amp_mean_pct'] / 100 * L * um:.1f} µm"
                    if um is not None else "")
            s = (f"Contractions: {n} in {res['measured_s']:.1f} s "
                 f"({res['rate_per_min']:.1f} per min); shortening "
                 f"{res['amp_mean_pct']:.1f}"
                 + (f" ± {res['amp_sd_pct']:.1f}" if n > 1 else "")
                 + f"% of length{a_um} (largest {res['amp_max_pct']:.1f}%)")
            if np.isfinite(res["interval_mean_s"]):
                s += (f"; one every {res['interval_mean_s']:.1f}"
                      + (f" ± {res['interval_sd_s']:.1f}"
                         if np.isfinite(res["interval_sd_s"]) else "")
                      + " s")
            lines.append(s + ".")
            ma = float(np.mean([abs(e.get("end_a_move_px", 0.0))
                                for e in res["events"]]))
            mb = float(np.mean([abs(e.get("end_b_move_px", 0.0))
                                for e in res["events"]]))
            na, nb = res["end_names"]
            s2 = (f"Ends: the {na} end moves {ln(ma)} per contraction, "
                  f"the {nb} end {ln(mb)}")
            if min(ma, mb) < 0.25 * max(ma, mb):
                s2 += (f" — the cell looks held near its "
                       f"{nb if mb < ma else na} end")
            lines.append(s2 + ".")
        if res.get("sl_found"):
            slv = np.asarray(res["sl_px"], dtype=float)
            fin = slv[np.isfinite(slv)]
            if fin.size:
                if um is not None:
                    s3 = (f"Sarcomeres: {np.median(fin) * um:.2f} µm (median; "
                          f"resting period {res['sl_ref_px'] * um:.2f} µm)")
                else:
                    s3 = (f"Sarcomeres: period {np.median(fin):.2f} px "
                          "(set a scale for µm)")
                if res["events"]:
                    sp_ = AutoCellAnalysis.speed_series(res)
                    sh = np.array(sp_["ev_sl_short"], dtype=float)
                    sh = sh[np.isfinite(sh)]
                    if sh.size:
                        s3 += (f", shortening {sh.mean():.1f}"
                               + (f" ± {sh.std(ddof=1):.1f}" if sh.size > 1
                                  else "") + "% per contraction")
                lines.append(s3 + ".")
        else:
            lines.append("Sarcomeres: not measured — "
                         + str(res.get("sl_reason", "")) + ".")
        lines.append(f"Probe: {res['verdict']}")
        if res["valid_frac"] < 0.98:
            lines.append(
                f"Note: the cell length could be measured on only "
                f"{100 * res['valid_frac']:.0f}% of the frames (hidden or "
                "too blurred on the rest).")
        lines.extend(AutoCellAnalysis.read_notes(res))
        if cell.get("hidden_px", 0) > 0:
            lines.append(
                f"Note: {100 * cell['hidden_px'] / max(1.0, cell['area_px']):.0f}"
                "% of the cell's area was under the probe; that part was "
                "rebuilt by carrying the cell's edges straight across "
                "beneath the probe (dashed orange outline).")
        if cell.get("source") == "outline":
            lines.append(
                "Note: the cell is the outline you drew"
                + (" (snapped to the cell's edge where it is visible)."
                   if cell.get("snapped") else " (used exactly as drawn)."))
        elif cell.get("n_candidates", 1) > 1:
            lines.append(
                f"Note: {cell['n_candidates']} cell-sized objects were in "
                "view; the one nearest the probe (or your picked points) "
                "was analysed.")
        return lines

    @staticmethod
    def find_repeats(chg) -> np.ndarray:
        """v1.9.5: True for frames that are copies of the frame before.
        `chg` = mean picture change vs the previous frame (NaN = unknown).
        Real frames always differ a little (camera noise, motion); copies
        differ only by compression noise, a several times smaller amount.
        Only called repeats when the changes clearly split into those two
        groups and the copies are a sizeable share of the frames."""
        chg = np.asarray(chg, dtype=float)
        rep = np.zeros(chg.size, dtype=bool)
        ok = np.isfinite(chg)
        if ok.sum() < 20:
            return rep
        # 1) the usual case: every picture stored twice, so every other
        #    frame barely changes. Judged in short windows so a slip in the
        #    pattern (a dropped frame) is followed.
        n = chg.size
        idx = np.arange(n)
        votes = np.zeros(n)
        seen = np.zeros(n)
        for a0 in range(0, max(1, n - 10), 20):
            sl_ = slice(a0, min(n, a0 + 40))
            ii, cc = idx[sl_], chg[sl_]
            meds = []
            for par in (0, 1):
                v_ = cc[(ii % 2 == par) & np.isfinite(cc)]
                meds.append(float(np.median(v_)) if v_.size >= 6 else np.nan)
            if not np.all(np.isfinite(meds)) or max(meds) <= 0:
                continue
            seen[sl_] += 1
            if min(meds) < 0.6 * max(meds):
                low = int(np.argmin(meds))
                votes[ii[ii % 2 == low]] += 1
        par_rep = (seen > 0) & (votes > 0.5 * seen) & ok
        if par_rep.sum() >= 0.3 * ok.sum():
            return par_rep
        # 2) otherwise: two clearly separate groups of change sizes
        v = np.log(np.maximum(chg[ok], 1e-6))
        vs = np.sort(v)
        best, thr = -1.0, None
        for k in range(int(0.05 * vs.size), int(0.95 * vs.size)):
            lo, hi = vs[:k], vs[k:]
            if lo.size < 3 or hi.size < 3:
                continue
            w0 = lo.size / vs.size
            bv = w0 * (1 - w0) * (hi.mean() - lo.mean()) ** 2
            if bv > best:
                best, thr = bv, 0.5 * (lo[-1] + hi[0])
        if thr is None:
            return rep
        low, high = v[v < thr], v[v >= thr]
        share = low.size / v.size
        ratio = float(np.exp(np.median(high) - np.median(low)))
        sep = (np.percentile(high, 10) - np.percentile(low, 90))
        if ratio >= 3.0 and sep > 0 and 0.15 <= share <= 0.8:
            idx = np.nonzero(ok)[0]
            rep[idx[v < thr]] = True
        return rep

    @staticmethod
    def index_of_frame(res: dict, frame: int) -> int:
        """Index into the result arrays for a video frame number (the
        nearest analysed frame at or before it)."""
        fr = np.asarray(res["frames"])
        i = int(np.searchsorted(fr, int(frame), side="right")) - 1
        return int(np.clip(i, 0, len(fr) - 1))

    @staticmethod
    def read_notes(res: dict) -> List[str]:
        """v1.9.4: plain-words notes when part of the video could not be
        read from the file."""
        out = []
        fps = float(res.get("fps") or Config.DEFAULT_FPS)

        def ts(f):
            return Utils.format_hhmmss(float(f) / fps)
        rngs = res.get("unread_ranges") or []
        if rngs:
            nb = int(res.get("n_unread", 0))
            where = ", ".join(
                (f"{ts(a_)}" if a_ == b_ else f"{ts(a_)}–{ts(b_)}")
                for a_, b_ in rngs[:4]) + (" …" if len(rngs) > 4 else "")
            out.append(
                f"Note: {nb} frame(s) could not be read from the video file "
                f"(damaged or unusual spot at {where}). They were skipped "
                "and left out of the measurements; the analysis carried on "
                "after them.")
        nrep = int(res.get("n_repeats", 0) or 0)
        if nrep:
            tt = np.asarray(res["t"], dtype=float)
            dts = np.diff(tt)
            dts = dts[np.isfinite(dts) & (dts > 0)]
            ufps = 1.0 / float(np.median(dts)) if dts.size else float("nan")
            s_ = (f"Note: {nrep} of the {nrep + int(res['n_frames'])} frames "
                  "in this range are repeats of the picture before (the "
                  "recording stored pictures twice). The repeats were left "
                  f"out, so the cell is measured on {int(res['n_frames'])} "
                  f"real pictures, one every {1000 / ufps:.0f} ms "
                  f"({ufps:.1f} per second).")
            if ufps < 20:
                s_ += (" That is slow for a contraction: the peak and the "
                       "speeds can be missed between pictures. Record at 30 "
                       "or more real frames per second if you can.")
            out.append(s_)
        if res.get("read_short"):
            last = int(np.asarray(res["frames"])[-1])
            out.append(
                f"Warning: the video file could only be read up to frame "
                f"{last} ({ts(last)}) of the {int(res.get('n_requested', 0))}"
                f" frames asked for (range ends at {ts(res['end_frame'])}). "
                "The rest of the file cannot be decoded. Converting the "
                "video again (e.g. with HandBrake or ffmpeg) usually fixes "
                "this.")
        return out

    @staticmethod
    def summary_row(res: dict) -> dict:
        """One line per video for the batch table."""
        um = res["um_per_px"]
        cell = res["cell"]
        k = um if um is not None else float("nan")
        return {
            "file": Path(res["video_path"]).name,
            "frames": res["n_frames"],
            "duration_s": round(res["duration_s"], 3),
            "um_per_px": (round(um, 5) if um is not None else ""),
            "cell_length_px": round(cell["length_px"], 2),
            "cell_width_mean_px": round(cell["width_mean_px"], 2),
            "cell_width_max_px": round(cell["width_max_px"], 2),
            "cell_area_px2": round(cell["area_px"], 1),
            "cell_length_um": round(cell["length_px"] * k, 2),
            "cell_width_mean_um": round(cell["width_mean_px"] * k, 2),
            "cell_area_um2": round(cell["area_px"] * k * k, 1),
            "cell_area_rebuilt_under_probe_pct": round(
                100 * cell.get("hidden_px", 0.0)
                / max(1.0, cell["area_px"]), 1),
            "cell_axis_deg": round(cell["angle_deg"], 1),
            "n_contractions": res["n_events"],
            "contractions_per_min": round(res["rate_per_min"], 3),
            "shortening_mean_pct": round(res["amp_mean_pct"], 3),
            "shortening_sd_pct": round(res["amp_sd_pct"], 3),
            "shortening_max_pct": round(res["amp_max_pct"], 3),
            "interval_mean_s": round(res["interval_mean_s"], 3),
            "interval_sd_s": round(res["interval_sd_s"], 3),
            "probe_found": int(res["probe"] is not None),
            "probe_max_move_px": round(res["max_probe_disp_px"], 2),
            "cell_max_move_px": round(res["max_cell_disp_px"], 2),
            "probe_cell_min_gap_px": round(res["min_gap_px"], 2),
            "probe_cell_min_gap_um": round(res["min_gap_px"] * k, 2),
            "cell_covered_max_pct": round(100 * res["max_overlap"], 1),
            "verdict": res["verdict_code"],
            **AutoCellAnalysis._sl_row(res),
        }

    @staticmethod
    def _sl_row(res: dict) -> dict:
        """v1.9: sarcomere columns for the folder table."""
        um = res.get("um_per_px")
        sl = np.asarray(res.get("sl_px", []), dtype=float)
        fin = sl[np.isfinite(sl)]
        row = {"sarcomere_found": int(bool(res.get("sl_found"))),
               "sarcomere_median_um": "", "sarcomere_shortening_mean_pct": ""}
        if fin.size and um is not None:
            row["sarcomere_median_um"] = round(float(np.median(fin)) * um, 4)
        try:
            sh = np.array(AutoCellAnalysis.speed_series(res)["ev_sl_short"],
                          dtype=float)
            sh = sh[np.isfinite(sh)]
            if sh.size:
                row["sarcomere_shortening_mean_pct"] = round(float(sh.mean()),
                                                             3)
        except Exception:
            pass
        return row

    @staticmethod
    def save_csvs(res: dict, base: str) -> List[str]:
        um = res["um_per_px"]
        L = float(res["cell"]["length_px"])
        k = um if um is not None else float("nan")

        def f(v, nd=4):
            return "" if (v is None or not np.isfinite(v)) else f"{v:.{nd}f}"

        p_ts = base + "_timeseries.csv"
        with open(p_ts, "w", encoding="utf-8") as fh:
            fh.write(",".join([
                "frame", "time_s", "length_change_pct", "length_change_um",
                f"{res['end_names'][0]}_end_move_px",
                f"{res['end_names'][1]}_end_move_px",
                "cell_dx_px", "cell_dy_px", "cell_match",
                "probe_x_px", "probe_y_px", "probe_move_px",
                "probe_cell_gap_px", "probe_cell_gap_um",
                "cell_covered_frac", "probe_edge_sharpness_rel",
                "length_rate_pct_per_s", "length_rate_um_per_s",
                "length_rate_px_per_s",
                f"{res['end_names'][0]}_end_speed_px_per_s",
                f"{res['end_names'][1]}_end_speed_px_per_s",
                "sarcomere_length_px", "sarcomere_length_um",
                "sarcomere_length_change_pct"]) + "\n")
            sp_u = AutoCellAnalysis.speed_series(res)
            sp = AutoCellAnalysis.speed_series(dict(res, um_per_px=None))
            for i in range(res["n_frames"]):
                pct = float(res["length_pct"][i])
                fh.write(",".join([
                    str(int(res["frames"][i])), f(float(res["t"][i])),
                    f(pct), f(pct / 100.0 * L * k),
                    f(float(res["end_a_px"][i]), 3),
                    f(float(res["end_b_px"][i]), 3),
                    f(float(res["cell_dx"][i]), 2),
                    f(float(res["cell_dy"][i]), 2),
                    f(float(res["cell_score"][i]), 3),
                    f(float(res["probe_x"][i]), 2),
                    f(float(res["probe_y"][i]), 2),
                    f(float(res["probe_disp_px"][i]), 2),
                    f(float(res["gap_px"][i]), 2),
                    f(float(res["gap_px"][i]) * k, 2),
                    f(float(res["overlap"][i]), 3),
                    f(float(res["probe_sharp_rel"][i]), 3),
                    f(float(sp["rate_pct"][i]), 4),
                    f(float(sp["rate_pct"][i]) / 100.0 * L * k, 4),
                    f(float(sp["rate_u"][i]), 3),
                    f(float(sp["end_a_speed"][i]), 3),
                    f(float(sp["end_b_speed"][i]), 3),
                    f(float(res["sl_px"][i]), 4),
                    f(float(res["sl_px"][i]) * k, 4),
                    f(float(sp_u["sl_pct"][i]), 3)]) + "\n")
        p_ev = base + "_events.csv"
        with open(p_ev, "w", encoding="utf-8") as fh:
            fh.write(",".join([
                "n", "t_peak_s", "frame", "shortening_pct", "shortening_um",
                "time_to_peak_s", "duration_half_s", "relax90_s",
                "shortening_speed_um_per_s", "relaxation_speed_um_per_s",
                f"{res['end_names'][0]}_end_move_px",
                f"{res['end_names'][1]}_end_move_px",
                "interval_s", "cut_off_by_range",
                "sarcomere_rest_um", "sarcomere_shortest_um",
                "sarcomere_shortening_pct"]) + "\n")
            sp_e = AutoCellAnalysis.speed_series(res)
            for j, e in enumerate(res["events"]):
                fh.write(",".join([
                    str(j + 1), f(e["t_peak"]),
                    str(int(res["frames"][e["index"]])),
                    f(e["amp_pct"]), f(e["amp_pct"] / 100.0 * L * k),
                    f(e["time_to_peak"]), f(e["dur50"]), f(e["relax90"]),
                    f(-e["v_short_pct_s"] / 100.0 * L * k, 3),
                    f(e["v_relax_pct_s"] / 100.0 * L * k, 3),
                    f(e.get("end_a_move_px", float("nan")), 3),
                    f(e.get("end_b_move_px", float("nan")), 3),
                    f(e["interval"]), str(int(e["partial"])),
                    f(sp_e["ev_sl_rest"][j] if um is not None else
                      float("nan")),
                    f(sp_e["ev_sl_min"][j] if um is not None else
                      float("nan")),
                    f(sp_e["ev_sl_short"][j], 3)]) + "\n")
        p_tx = base + "_report.txt"
        with open(p_tx, "w", encoding="utf-8") as fh:
            fh.write("\n".join(AutoCellAnalysis.summary_lines(res)) + "\n")
        return [p_ts, p_ev, p_tx]

    FIGSIZE = (12.0, 10.2)

    @staticmethod
    def save_figure(res: dict, path: str) -> str:
        """Write the summary figure to a file. Uses no window toolkit, so
        it is safe from a worker thread (batch mode)."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=AutoCellAnalysis.FIGSIZE, facecolor="white")
        FigureCanvasAgg(fig)
        AutoCellAnalysis.build_figure(res, fig)
        fig.savefig(path, dpi=160, facecolor="white")
        return path

    @staticmethod
    def build_figure(res: dict, fig):
        """One-page summary drawn onto `fig`: where the cell and probe were
        found, the cell relaxed vs at its largest contraction, the length
        trace with every contraction marked, and what the probe did."""
        import matplotlib
        um = res["um_per_px"]
        cell = res["cell"]
        probe = res["probe"]
        t = res["t"]
        L = float(cell["length_px"])
        ink, mut, line = "#1f2328", "#5f6875", "#2f6fb5"
        unit = "µm" if um is not None else "px"
        k = um if um is not None else 1.0
        rc = {
            "font.family": "sans-serif", "font.size": 9,
            "axes.titlesize": 9.5, "axes.labelsize": 9,
            "axes.labelweight": "normal", "axes.labelpad": 4,
            "axes.titlepad": 6, "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
            "xtick.major.width": 0.8, "ytick.major.width": 0.8,
            "xtick.major.size": 3.5, "ytick.major.size": 3.5,
            "xtick.major.pad": 3.5, "ytick.major.pad": 3.5,
            "xtick.direction": "out", "ytick.direction": "out",
            "xtick.top": False, "ytick.right": False,
            "xtick.minor.visible": False, "ytick.minor.visible": False,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": False, "lines.linewidth": 1.4,
            "legend.fontsize": 8.5, "legend.frameon": False,
            "figure.dpi": 100, "savefig.dpi": 160,
        }
        with matplotlib.rc_context(rc):
            fig.set_facecolor("white")
            gs = fig.add_gridspec(4, 4,
                                  height_ratios=[1.45, 1.0, 0.8, 0.72],
                                  hspace=0.45, wspace=0.28,
                                  left=0.06, right=0.94, top=0.885,
                                  bottom=0.06)
            fig.suptitle(
                f"{Path(res['video_path']).name}   ·   automatic analysis",
                x=0.06, y=0.978, ha="left", fontsize=11.5, color=ink,
                fontweight="bold")
            fig.text(0.06, 0.951, res["verdict"], ha="left", va="center",
                     fontsize=9.5, color=ink)
            sizes = (f"Cell {L * k:.0f} × {cell['width_mean_px'] * k:.0f} "
                     f"{unit} (length × mean width)")
            if probe is not None:
                sizes += (f"   ·   probe tip Ø "
                          f"{2 * probe['radius'] * k:.0f} {unit}")
            sizes += (f"   ·   {um:.3f} µm/px" if um is not None
                      else "   ·   no scale set (pixels)")
            fig.text(0.06, 0.928, sizes, ha="left", va="center",
                     fontsize=9.5, color=mut)

            # ---- A: whole picture with what was found ----
            axA = fig.add_subplot(gs[0, 0:2])
            axA.imshow(res["ref_u8"], cmap="gray", vmin=0, vmax=255)
            c = cell["contour"]
            axA.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]],
                     color="#00b3a4", lw=1.3)
            for hc in AutoCellAnalysis.hidden_contours(cell):
                axA.plot(np.r_[hc[:, 0], hc[0, 0]], np.r_[hc[:, 1], hc[0, 1]],
                         color="#ff9f1c", lw=1.1, ls=(0, (3, 2)))
            ax_, ay_ = cell["axis"]
            cx, cy = cell["center"]
            axA.plot([cx - 0.5 * L * ax_, cx + 0.5 * L * ax_],
                     [cy - 0.5 * L * ay_, cy + 0.5 * L * ay_],
                     color="#00b3a4", lw=0.9, ls=(0, (4, 3)))
            if probe is not None:
                th = np.linspace(0, 2 * np.pi, 120)
                px, py = probe["center"]
                r = float(probe["radius"])
                axA.plot(px + r * np.cos(th), py + r * np.sin(th),
                         color="#ffd23f", lw=1.3, ls=(0, (4, 2)))
            hh, ww = res["ref_u8"].shape[:2]
            if um is not None:
                bar_um = float(Config.SCALE_BAR_UM)
                bar = bar_um / um
                if bar < 0.6 * ww:
                    xb, yb = ww - 24 - bar, hh - 26
                    axA.plot([xb, xb + bar], [yb, yb], color="white", lw=5,
                             solid_capstyle="butt")
                    axA.plot([xb, xb + bar], [yb, yb], color=ink, lw=3,
                             solid_capstyle="butt")
                    axA.text(xb + bar / 2, yb - 10, f"{bar_um:g} µm",
                             ha="center", va="bottom", fontsize=8.5,
                             color=ink)
            axA.set_xlim(0, ww)
            axA.set_ylim(hh, 0)
            axA.set_xticks([])
            axA.set_yticks([])
            axA.set_title(
                ("Cell = your outline (snapped to its edge); probe tip"
                 if cell.get("source") == "outline" else
                 "Found automatically: cell outline and probe tip"),
                loc="left", color=ink)

            # ---- B, C: the cell relaxed vs at its largest contraction ----
            x0, y0, x1, y1 = cell["bbox"]
            m = int(max(12, 0.25 * max(x1 - x0, y1 - y0)))
            xa, xb_ = max(0, x0 - m), min(ww, x1 + m)
            ya, yb_ = max(0, y0 - m), min(hh, y1 + m)
            axB = fig.add_subplot(gs[0, 2])
            axB.imshow(res["ref_u8"][ya:yb_, xa:xb_], cmap="gray",
                       vmin=0, vmax=255)
            axB.set_title("Relaxed", loc="left", color=ink)
            axC = fig.add_subplot(gs[0, 3])
            if res["peak_u8"] is not None and res["events"]:
                big = max(res["events"], key=lambda e: e["amp_pct"])
                # same window, moved with the cell if the stage moved
                ox = int(round(float(res["cell_dx"][big["index"]])))
                oy = int(round(float(res["cell_dy"][big["index"]])))
                ox = int(np.clip(ox, -xa, ww - xb_))
                oy = int(np.clip(oy, -ya, hh - yb_))
                axC.imshow(res["peak_u8"][ya + oy:yb_ + oy,
                                          xa + ox:xb_ + ox],
                           cmap="gray", vmin=0, vmax=255)
                axC.set_title(
                    f"Shortest: −{big['amp_pct']:.1f}% at "
                    f"{big['t_peak']:.1f} s", loc="left", color=ink)
                # same two guide lines on both pictures: the cell ends
                for a in (axB, axC):
                    for sgn in (-1, 1):
                        ex = cx + sgn * 0.5 * L * ax_ - xa
                        ey = cy + sgn * 0.5 * L * ay_ - ya
                        bx_, by_ = -ay_, ax_
                        hwid = 0.75 * float(cell["width_max_px"])
                        a.plot([ex - hwid * bx_, ex + hwid * bx_],
                               [ey - hwid * by_, ey + hwid * by_],
                               color="#00b3a4", lw=1.0, ls=(0, (4, 3)))
            else:
                axC.text(0.5, 0.5, "No contraction\nfound in this range",
                         ha="center", va="center", color=mut, fontsize=10,
                         transform=axC.transAxes)
                axC.set_facecolor("#f3f4f6")
            for a in (axB, axC):
                a.set_xticks([])
                a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(True)
                    sp.set_color("#c5c9d0")
            for sp in axA.spines.values():
                sp.set_visible(True)
                sp.set_color("#c5c9d0")

            # ---- D: length change, contractions marked ----
            axL = fig.add_subplot(gs[1, :])
            pct = res["length_pct"]
            axL.axhline(0, color="#d0d4da", lw=0.8)
            # grey band = the probe is over the cell
            ov = res["overlap"] >= float(Config.AUTO_OVERLAP_MIN_FRAC)
            if ov.any():
                edges = np.flatnonzero(np.diff(np.r_[0, ov.astype(int), 0]))
                for a_i, b_i in zip(edges[::2], edges[1::2]):
                    axL.axvspan(float(t[a_i]), float(t[b_i - 1]),
                                color="#e3e5e9", lw=0)
                axL.text(float(t[edges[0]]), 1.0,
                         " probe over the cell", transform=axL.
                         get_xaxis_transform(), ha="left", va="top",
                         fontsize=8, color=mut)
            axL.plot(t, pct, color=line, lw=1.4)
            for j, e in enumerate(res["events"]):
                axL.plot([e["t_peak"]], [pct[e["index"]]], marker="v",
                         ms=5, color=ink, ls="none")
            fin = pct[np.isfinite(pct)]
            if fin.size:
                lo, hi = float(fin.min()), float(fin.max())
                padv = max(0.5, 0.12 * (hi - lo))
                axL.set_ylim(min(lo - padv, -1.0), max(hi + padv, 0.6))
            axL.set_xlim(float(t[0]), float(t[-1]))
            axL.set_xlabel("Time in video (s)", color=mut)
            axL.set_ylabel("Cell length change (%)", color=mut)
            axL.grid(axis="y", color="#eceef1", lw=0.8)
            axL.set_axisbelow(True)
            n = res["n_events"]
            if n == 0:
                ttl = ("Cell length along its long axis: no contraction "
                       f"found (smallest counted {res['event_thr_pct']:.2f}%)")
            else:
                ttl = (f"Cell length along its long axis: {n} "
                       f"contraction{'s' if n != 1 else ''}, shortening "
                       f"{res['amp_mean_pct']:.1f}"
                       + (f" ± {res['amp_sd_pct']:.1f}" if n > 1 else "")
                       + "%")
                if um is not None:
                    ttl += f" ({res['amp_mean_pct'] / 100 * L * um:.1f} µm)"
                if np.isfinite(res["interval_mean_s"]):
                    ttl += (f", one every {res['interval_mean_s']:.1f}"
                            + (f" ± {res['interval_sd_s']:.1f}"
                               if np.isfinite(res["interval_sd_s"])
                               else "") + " s")
            axL.set_title(ttl, loc="left", color=ink)
            if um is not None:
                sec = axL.secondary_yaxis(
                    "right",
                    functions=(lambda p: p / 100.0 * L * um,
                               lambda u: u / (L * um) * 100.0))
                sec.set_ylabel("µm", color=mut)
                sec.tick_params(colors=mut)
                sec.spines["right"].set_color("#c5c9d0")

            # ---- v1.9: speed over time ----
            axS = fig.add_subplot(gs[2, :], sharex=axL)
            spd = AutoCellAnalysis.speed_series(res)
            axS.axhline(0, color="#d0d4da", lw=0.8)
            axS.plot(t, spd["rate_u"], color=line, lw=1.3)
            for j, e in enumerate(res["events"]):
                jj = spd["ev_ivshort"][j]
                if np.isfinite(spd["rate_u"][jj]):
                    axS.plot([t[jj]], [spd["rate_u"][jj]], marker="o",
                             ms=4, color=ink, ls="none")
            sfin = spd["rate_u"][np.isfinite(spd["rate_u"])]
            if sfin.size:
                slo, shi = float(sfin.min()), float(sfin.max())
                sp_ = max(1e-6, 0.12 * (shi - slo))
                axS.set_ylim(slo - sp_, shi + sp_)
            axS.set_xlabel("Time in video (s)", color=mut)
            axS.set_ylabel(f"Speed ({unit}/s)", color=mut)
            axS.grid(axis="y", color="#eceef1", lw=0.8)
            axS.set_axisbelow(True)
            vs_ = np.array(spd["ev_vshort"], dtype=float)
            vs_ = vs_[np.isfinite(vs_)]
            sttl = ("Speed over time (rate of length change; below 0 = "
                    "shortening)")
            if vs_.size:
                sttl += (f": peak shortening speed {vs_.mean():.1f}"
                         + (f" ± {vs_.std(ddof=1):.1f}" if vs_.size > 1
                            else "") + f" {unit}/s")
            axS.set_title(sttl, loc="left", color=ink)
            axS.tick_params(colors=mut)
            for side in ("left", "bottom"):
                axS.spines[side].set_color("#c5c9d0")
            axL.set_xlabel("")

            # ---- E, F: what the probe did ----
            axP = fig.add_subplot(gs[3, 0:2])
            axG = fig.add_subplot(gs[3, 2:4])
            if probe is not None and np.isfinite(res["probe_disp_px"]).any():
                axP.plot(t, res["probe_disp_px"] * k, color=line, lw=1.3)
                top = float(np.nanmax(res["probe_disp_px"])) * k
                lim = max(float(Config.AUTO_PROBE_MOVE_MIN_PX) * k * 1.5,
                          1.15 * top)
                axP.set_ylim(-0.04 * lim, lim)
                axP.axhline(float(Config.AUTO_PROBE_MOVE_MIN_PX) * k,
                            color="#b9bec7", lw=0.8, ls=(0, (4, 3)))
                axP.text(float(t[-1]),
                         float(Config.AUTO_PROBE_MOVE_MIN_PX) * k,
                         " counted as movement above this line ",
                         ha="right", va="bottom", fontsize=7.5, color=mut)
                axG.plot(t, res["gap_px"] * k, color=line, lw=1.3)
                axG.axhline(0, color="#b9bec7", lw=0.8, ls=(0, (4, 3)))
                gfin = res["gap_px"][np.isfinite(res["gap_px"])] * k
                if gfin.size:
                    glo, ghi = float(gfin.min()), float(gfin.max())
                    gp = max(1.0, 0.15 * (ghi - glo))
                    axG.set_ylim(min(glo - gp, -gp), ghi + gp)
                axG.text(float(t[0]), 0, "  0 or less = probe touching "
                         "/ over the cell", ha="left", va="bottom",
                         fontsize=7.5, color=mut)
            else:
                for a in (axP, axG):
                    a.text(0.5, 0.5, "No probe found", ha="center",
                           va="center", color=mut, transform=a.transAxes)
            axP.set_title("Probe tip: distance moved from its first position",
                          loc="left", color=ink)
            axG.set_title("Gap between probe tip and cell edge",
                          loc="left", color=ink)
            for a, yl in ((axP, f"Moved ({unit})"), (axG, f"Gap ({unit})")):
                a.set_xlim(float(t[0]), float(t[-1]))
                a.set_xlabel("Time in video (s)", color=mut)
                a.set_ylabel(yl, color=mut)
                a.grid(axis="y", color="#eceef1", lw=0.8)
                a.set_axisbelow(True)
            for a in (axL, axP, axG):
                a.tick_params(colors=mut)
                for side in ("left", "bottom"):
                    a.spines[side].set_color("#c5c9d0")
        return fig

    # ------------------------------ files / batch ------------------------------

    @staticmethod
    def output_base(res: dict) -> str:
        """<video without extension>_auto, plus the frame range when only
        a section was analysed (so sections never overwrite each other)."""
        base = str(Path(res["video_path"]).with_suffix("")) + "_auto"
        try:
            cap = RobustCapture(res["video_path"])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception:
            total = 0
        whole = (res["start_frame"] == 0
                 and (total <= 0 or res["end_frame"] >= total - 2))
        if not whole:
            base += f"_f{res['start_frame']}-{res['end_frame']}"
        return base

    @staticmethod
    def save_all(res: dict, base: Optional[str] = None) -> List[str]:
        """Summary figure + time series + contraction list + report."""
        if base is None:
            base = AutoCellAnalysis.output_base(res)
        out = [AutoCellAnalysis.save_figure(res, base + "_summary.png")]
        out += AutoCellAnalysis.save_csvs(res, base)
        return out

    @staticmethod
    def list_videos(folder: str) -> List[str]:
        exts = tuple(e.lower() for e in Config.AUTO_VIDEO_EXTS)
        return sorted(str(p) for p in Path(folder).iterdir()
                      if p.is_file() and p.suffix.lower() in exts)

    @staticmethod
    def run_batch(files: List[str], out_csv: str,
                  probe_diameter_um: Optional[float] = None,
                  um_per_px: Optional[float] = None,
                  progress_callback=None) -> Tuple[List[dict], str]:
        """Analyse every video in `files`, save each one's figure/CSVs
        next to it, and write ONE table (one line per video) to out_csv.
        progress_callback(file_no, n_files, done, total, name) -> True to
        stop. A video that fails gets a line with the reason; the batch
        carries on."""
        rows: List[dict] = []
        n = len(files)
        for k, path in enumerate(files):
            name = Path(path).name

            def cb(done, total, _k=k, _name=name):
                if progress_callback is None:
                    return False
                return bool(progress_callback(_k, n, done, total, _name))

            try:
                res = AutoCellAnalysis.analyze(
                    path, um_per_px=um_per_px,
                    probe_diameter_um=probe_diameter_um,
                    progress_callback=cb)
                if res is None:          # stopped by the user
                    break
                AutoCellAnalysis.save_all(res)
                row = AutoCellAnalysis.summary_row(res)
                row["error"] = ""
            except Exception as e:
                row = {"file": name, "error": str(e).replace("\n", " ")}
            rows.append(row)
        cols: List[str] = []
        for r in rows:
            for c in r:
                if c not in cols:
                    cols.append(c)
        if "error" in cols:
            cols.remove("error")
            cols.append("error")
        with open(out_csv, "w", encoding="utf-8") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                vals = []
                for c in cols:
                    v = r.get(c, "")
                    if isinstance(v, float) and not np.isfinite(v):
                        v = ""
                    v = str(v)
                    if "," in v or '"' in v:
                        v = '"' + v.replace('"', '""') + '"'
                    vals.append(v)
                fh.write(",".join(vals) + "\n")
        return rows, out_csv


# ---------------------------- Auto analysis: results window ----------------------------



# ------------------------------------- the probe sphere, as is (desktop) ---

class SphereEdgeRefiner:
    @staticmethod
    def refine(
        gray: np.ndarray,
        center_pred: Tuple[float, float],
        r_pred: float,
        num_rays: int = Config.SPHERE_REFINE_NUM_RAYS,
        band: float = Config.SPHERE_REFINE_BAND_PX,
        min_grad: float = Config.SPHERE_REFINE_MIN_GRAD,
        min_valid_frac: float = Config.SPHERE_REFINE_MIN_VALID_RAYS_FRAC,
        blur_ksize: int = Config.SPHERE_REFINE_BLUR_KSIZE,
    ) -> dict:
        h, w = gray.shape[:2]
        cx, cy = float(center_pred[0]), float(center_pred[1])
        r_pred = float(r_pred)

        result = {
            "ok": False,
            "center": (cx, cy),
            "radius": r_pred,
            "edge": (cx + r_pred, cy),
            "confidence": 0.0,
            "num_rays_valid": 0,
        }

        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(r_pred)) or r_pred <= 2.0:
            return result

        ksize = max(1, int(blur_ksize))
        if ksize % 2 == 0:
            ksize += 1
        if ksize >= 3:
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        else:
            blurred = gray

        r_min = max(1.0, r_pred - band)
        r_max = r_pred + band
        n_samples = max(5, int(round(r_max - r_min)) + 1)
        radii = np.linspace(r_min, r_max, n_samples)

        thetas = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)

        refined_radii = []
        offset_x_list = []
        offset_y_list = []

        for theta in thetas:
            ct, st = np.cos(theta), np.sin(theta)
            xs = cx + radii * ct
            ys = cy + radii * st

            if (xs.min() < 0 or xs.max() >= w - 1 or
                    ys.min() < 0 or ys.max() >= h - 1):
                continue

            samples = cv2.remap(
                blurred,
                xs.astype(np.float32).reshape(1, -1),
                ys.astype(np.float32).reshape(1, -1),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).reshape(-1).astype(np.float64)

            if samples.size < 3:
                continue

            grad = np.abs(np.gradient(samples))
            peak_idx = int(np.argmax(grad))
            peak_val = float(grad[peak_idx])

            if peak_val < min_grad:
                continue

            r_found = float(radii[peak_idx])
            refined_radii.append(r_found)
            residual = r_found - r_pred
            offset_x_list.append(residual * ct)
            offset_y_list.append(residual * st)

        n_valid = len(refined_radii)
        confidence = n_valid / float(num_rays) if num_rays > 0 else 0.0

        if n_valid == 0:
            return result

        radius_refined = float(np.median(refined_radii))

        if n_valid >= max(6, int(0.5 * num_rays)):
            dx = float(np.mean(offset_x_list))
            dy = float(np.mean(offset_y_list))
            cap = 0.5 * band
            dx = float(Utils.clamp(dx, -cap, cap))
            dy = float(Utils.clamp(dy, -cap, cap))
            cx_ref = cx + dx
            cy_ref = cy + dy
        else:
            cx_ref, cy_ref = cx, cy

        ok = confidence >= min_valid_frac

        ex = cx_ref + radius_refined
        ey = cy_ref

        return {
            "ok": ok,
            "center": (cx_ref, cy_ref),
            "radius": radius_refined,
            "edge": (ex, ey),
            "confidence": confidence,
            "num_rays_valid": n_valid,
        }

    @staticmethod
    def snap_edge_to_gradient(
        gray: np.ndarray,
        center_xy: Tuple[float, float],
        edge_click_xy: Tuple[float, float],
        search_band_px: float = Config.SPHERE_REFINE_BAND_PX,
        min_grad: float = Config.SPHERE_REFINE_MIN_GRAD,
        blur_ksize: int = Config.SPHERE_REFINE_BLUR_KSIZE,
    ) -> Tuple[Tuple[float, float], bool]:
        cx, cy = float(center_xy[0]), float(center_xy[1])
        ex, ey = float(edge_click_xy[0]), float(edge_click_xy[1])

        dx, dy = ex - cx, ey - cy
        r_click = float(np.hypot(dx, dy))
        if r_click < 2.0:
            return (ex, ey), False

        ux, uy = dx / r_click, dy / r_click
        h, w = gray.shape[:2]

        ksize = max(1, int(blur_ksize))
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(
            gray, (ksize, ksize), 0) if ksize >= 3 else gray

        r_min = max(1.0, r_click - search_band_px)
        r_max = r_click + search_band_px
        n_samples = max(7, int(round(r_max - r_min)) + 1)
        radii = np.linspace(r_min, r_max, n_samples)

        xs = cx + radii * ux
        ys = cy + radii * uy
        if (xs.min() < 0 or xs.max() >= w - 1
                or ys.min() < 0 or ys.max() >= h - 1):
            return (ex, ey), False

        samples = cv2.remap(
            blurred,
            xs.astype(np.float32).reshape(1, -1),
            ys.astype(np.float32).reshape(1, -1),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1).astype(np.float64)
        if samples.size < 3:
            return (ex, ey), False

        grad = np.abs(np.gradient(samples))
        peak_idx = int(np.argmax(grad))
        if float(grad[peak_idx]) < min_grad:
            return (ex, ey), False

        r_snapped = float(radii[peak_idx])
        return (cx + r_snapped * ux, cy + r_snapped * uy), True

    @staticmethod
    def _radial_edge_sharpness(
        gray: np.ndarray,
        cx: float,
        cy: float,
        r: float,
        band_frac: float = Config.SPHERE_DETECT_EDGE_PROBE_BAND_FRAC,
        num_rays: int = Config.SPHERE_DETECT_EDGE_PROBE_RAYS,
    ) -> Tuple[float, float]:
        h, w = gray.shape[:2]
        band = max(2.0, r * band_frac)
        r_min = max(1.0, r - band)
        r_max = r + band
        n_samples = max(7, int(round(r_max - r_min)) + 1)
        radii = np.linspace(r_min, r_max, n_samples)

        thetas = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
        peak_radii = []
        centered_scores = []

        for theta in thetas:
            ct, st = np.cos(theta), np.sin(theta)
            xs = cx + radii * ct
            ys = cy + radii * st

            if (xs.min() < 0 or xs.max() >= w - 1
                    or ys.min() < 0 or ys.max() >= h - 1):
                continue

            samples = cv2.remap(
                gray,
                xs.astype(np.float32).reshape(1, -1),
                ys.astype(np.float32).reshape(1, -1),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).reshape(-1).astype(np.float64)
            if samples.size < 5:
                continue

            grad = np.abs(np.gradient(samples))
            if grad.max() < 2.0:
                continue

            peak_idx = int(np.argmax(grad))
            peak_radii.append(float(radii[peak_idx]))

            third = max(1, n_samples // 3)
            central_slice = grad[third:n_samples - third] if (
                n_samples - 2 * third) > 0 else grad
            central = float(central_slice.sum())
            total = float(grad.sum()) + 1e-9
            centered_scores.append(central / total)

        if not peak_radii:
            return 0.0, r

        sharpness = float(np.median(centered_scores))
        r_refined = float(np.median(peak_radii))
        return sharpness, r_refined

    @staticmethod
    def _hough_detect_single(
        gray: np.ndarray,
        top_k: int = 3,
        hint_xy: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[float, float, float]]:
        h, w = gray.shape[:2]

        mksize = int(getattr(Config, "SPHERE_DETECT_MEDIAN_KSIZE", 5))
        if mksize % 2 == 0:
            mksize += 1
        if mksize >= 3:
            smoothed = cv2.medianBlur(gray, mksize)
        else:
            smoothed = gray

        min_dim = min(h, w)
        if hint_xy is not None:
            r_max_search = max(
                10, int(round(Config.SPHERE_DETECT_HINT_RADIUS_FRAC * min_dim)))
            min_r = max(5, int(round(
                Config.SPHERE_DETECT_HINT_RMIN_FRAC * r_max_search)))
            max_r = max(min_r + 5, r_max_search)
        else:
            min_r = max(5, int(round(
                Config.SPHERE_DETECT_RADIUS_FRAC_MIN * min_dim)))
            max_r = max(min_r + 5, int(round(
                Config.SPHERE_DETECT_RADIUS_FRAC_MAX * min_dim)))

        min_dist = max(min_r * 2, min_dim // 4)

        circles = cv2.HoughCircles(
            smoothed,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=Config.SPHERE_DETECT_HOUGH_PARAM1,
            param2=Config.SPHERE_DETECT_HOUGH_PARAM2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None or circles.shape[1] == 0:
            return []

        k_pre = int(getattr(
            Config, "SPHERE_DETECT_TOP_K_BEFORE_SCORE", 8))
        cand: List[Tuple[float, float, float]] = []
        if hint_xy is not None:
            hx, hy = float(hint_xy[0]), float(hint_xy[1])
            tol = float(max_r)
            for c in circles[0]:
                cx, cy, cr = float(c[0]), float(c[1]), float(c[2])
                if np.hypot(cx - hx, cy - hy) <= tol:
                    cand.append((cx, cy, cr))
                if len(cand) >= k_pre:
                    break
        else:
            for c in circles[0, :k_pre]:
                cand.append((float(c[0]), float(c[1]), float(c[2])))

        if not cand:
            return []

        band_frac = float(getattr(
            Config, "SPHERE_DETECT_EDGE_PROBE_BAND_FRAC", 0.12))
        num_rays = int(getattr(
            Config, "SPHERE_DETECT_EDGE_PROBE_RAYS", 24))
        sharp_min = float(getattr(
            Config, "SPHERE_DETECT_EDGE_SHARPNESS_MIN", 0.55))

        scored: List[Tuple[float, Tuple[float, float, float]]] = []
        fallback: List[Tuple[float, Tuple[float, float, float]]] = []

        for cx, cy, cr in cand:
            sharpness, r_ref = SphereEdgeRefiner._radial_edge_sharpness(
                smoothed, cx, cy, cr,
                band_frac=band_frac, num_rays=num_rays)
            refined = (cx, cy, r_ref)
            if sharpness >= sharp_min:
                scored.append((sharpness, refined))
            else:
                fallback.append((sharpness, refined))

        if scored:
            scored.sort(key=lambda t: t[0], reverse=True)
            return [t[1] for t in scored[:int(top_k)]]

        fallback.sort(key=lambda t: t[0], reverse=True)
        return [t[1] for t in fallback[:int(top_k)]]

    @staticmethod
    def _suppress_specular(gray: np.ndarray) -> np.ndarray:
        """Remove the bright specular dot many spheres show at 60x. That white
        highlight otherwise fools HoughCircles into a tiny/false detection.
        Inpaint the near-saturated pixels with surrounding intensity."""
        thr = int(getattr(Config, "SPECULAR_THRESH", 235))
        mask = (gray >= thr).astype(np.uint8)
        if float(mask.mean()) > 0.0002:
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
            try:
                gray = cv2.inpaint(
                    gray, mask * 255,
                    int(getattr(Config, "SPECULAR_INPAINT_RADIUS", 4)),
                    cv2.INPAINT_TELEA)
            except Exception:
                gray = cv2.medianBlur(gray, 5)
        return gray

    @staticmethod
    def _detect_dark_disk(gray: np.ndarray,
                          hint_xy=None):
        """Fallback when Hough finds nothing: the sphere is a large dark disk
        on a lighter background. Otsu-threshold the dark region, take the most
        circular blob in the plausible size range, fit an enclosing circle."""
        h, w = gray.shape[:2]
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(
            g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(
            th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        min_dim = min(h, w)
        rmin = Config.SPHERE_DETECT_RADIUS_FRAC_MIN * min_dim
        rmax = Config.SPHERE_DETECT_RADIUS_FRAC_MAX * min_dim
        best = None
        best_score = -1.0
        for c in cnts:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if r < rmin or r > rmax:
                continue
            area = cv2.contourArea(c)
            circ = area / (math.pi * r * r + 1e-9)
            if circ < 0.55:
                continue
            score = circ
            if hint_xy is not None:
                d = math.hypot(cx - float(hint_xy[0]), cy - float(hint_xy[1]))
                score -= 0.001 * d
            if score > best_score:
                best_score = score
                best = (float(cx), float(cy), float(r))
        return best

    @staticmethod
    def _radius_from_center(gray, cx, cy, num_rays=48, min_grad=3.0):
        """Given a trusted center (the user's hint), find the sphere radius by
        shooting rays outward and taking the median radius of the strongest
        intensity edge. Guarantees the detection sits AT the hint."""
        h, w = gray.shape[:2]
        min_dim = min(h, w)
        r_min = max(4.0, Config.SPHERE_DETECT_RADIUS_FRAC_MIN * min_dim)
        r_max = max(r_min + 5.0, Config.SPHERE_DETECT_RADIUS_FRAC_MAX * min_dim)
        radii = np.arange(r_min, r_max, 1.0, dtype=np.float32)
        if radii.size < 5:
            return None
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        thetas = np.linspace(0.0, 2.0 * np.pi, int(num_rays), endpoint=False)
        found = []
        for th in thetas:
            ct, st = math.cos(th), math.sin(th)
            xs = cx + radii * ct
            ys = cy + radii * st
            inb = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
            if int(inb.sum()) < 5:
                continue
            xs2 = xs[inb]
            ys2 = ys[inb]
            rr = radii[inb]
            samples = cv2.remap(
                g, xs2.astype(np.float32).reshape(1, -1),
                ys2.astype(np.float32).reshape(1, -1),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE).reshape(-1).astype(np.float64)
            if samples.size < 5:
                continue
            grad = np.abs(np.gradient(samples))
            gmax = float(grad.max()) if grad.size else 0.0
            if gmax < min_grad:
                continue
            # Take the OUTERMOST strong edge, so we reach the outer boundary of
            # the dark shadow halo (the sphere can be transparent inside with a
            # dark rim). Require it to be >= half the strongest edge on the ray.
            thr = max(min_grad, 0.5 * gmax)
            sig = np.where(grad >= thr)[0]
            if sig.size:
                found.append(float(rr[int(sig[-1])]))
        if len(found) >= max(6, int(0.25 * num_rays)):
            return float(np.median(found))
        return None

    @staticmethod
    def auto_detect_multiframe(
        video_path: str,
        center_frame: int,
        num_frames: int = Config.SPHERE_DETECT_NUM_FRAMES,
        stride: int = Config.SPHERE_DETECT_FRAME_STRIDE,
        agreement_px: float = Config.SPHERE_DETECT_AGREEMENT_PX,
        hint_xy: Optional[Tuple[float, float]] = None,
    ) -> Optional[dict]:
        cap = RobustCapture(video_path)
        if not cap.isOpened():
            return None
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return None
            half = num_frames // 2
            offsets = [(i - half) * stride for i in range(num_frames)]

            # Hint given -> TRUST the clicked center, find radius radially so
            # the detection lands exactly where the user clicked.
            if hint_xy is not None:
                hx, hy = float(hint_xy[0]), float(hint_xy[1])
                hradii = []
                for off in offsets:
                    idx = int(Utils.clamp(center_frame + off, 0, total - 1))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    g = SphereEdgeRefiner._suppress_specular(g)
                    rr = SphereEdgeRefiner._radius_from_center(g, hx, hy)
                    if rr is not None:
                        hradii.append(rr)
                if hradii:
                    r_med = float(np.median(hradii))
                    if np.isfinite(r_med) and r_med >= 3:
                        return {
                            "center": (hx, hy),
                            "radius": r_med,
                            "n_frames_used": len(hradii),
                            "n_frames_agreed": len(hradii),
                        }

            detections = []
            for off in offsets:
                idx = int(Utils.clamp(center_frame + off, 0, total - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = SphereEdgeRefiner._suppress_specular(gray)
                hits = SphereEdgeRefiner._hough_detect_single(
                    gray, top_k=1, hint_xy=hint_xy)
                cand = hits[0] if hits else \
                    SphereEdgeRefiner._detect_dark_disk(gray, hint_xy=hint_xy)
                if cand is None:
                    continue
                cx, cy, r0 = cand
                # The 60x sphere is transparent in the middle with a dark shadow
                # rim, so Hough tends to lock the small bright inner circle. The
                # center is still concentric, so push the radius OUT to the outer
                # edge of the dark shadow (the true sphere boundary).
                r_out = SphereEdgeRefiner._radius_from_center(gray, cx, cy)
                if r_out is not None and r_out >= r0:
                    r0 = float(r_out)
                detections.append((cx, cy, r0))
        finally:
            cap.release()

        if not detections:
            return None

        arr = np.array(detections, dtype=np.float64)
        med_cx = float(np.median(arr[:, 0]))
        med_cy = float(np.median(arr[:, 1]))
        med_r = float(np.median(arr[:, 2]))

        d = np.hypot(arr[:, 0] - med_cx, arr[:, 1] - med_cy)
        keep = d <= float(agreement_px)
        n_agreed = int(keep.sum())
        if n_agreed >= 2:
            kept = arr[keep]
            cx = float(np.median(kept[:, 0]))
            cy = float(np.median(kept[:, 1]))
            r = float(np.median(kept[:, 2]))
        else:
            cx, cy, r = med_cx, med_cy, med_r

        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(r)) or r < 3:
            return None

        return {
            "center": (cx, cy),
            "radius": r,
            "n_frames_used": int(arr.shape[0]),
            "n_frames_agreed": n_agreed,
        }


# ---------------------------- Reference Image Saver (4-point mode) ----------------------------

_PT_BGR = [
    (107, 107, 255),
    (196, 205,  78),
    (209, 183,  69),
    (36, 202, 249),
]
_PT_LABELS = ["P1", "P2", "P3", "P4"]

_SPHERE_CENTER_BGR = (107, 107, 255)
_SPHERE_EDGE_BGR = (196, 205,  78)
_SPHERE_RING_BGR = (36, 202, 249)


def snap_sphere(gray, circle):
    """
    Fit a marked circle to the sphere's edge (SphereEdgeRefiner).

    Each refine moves the circle only part of the way, so it is repeated
    while it is converging: on a round edge every step is about half the
    last. When the steps stop shrinking the circle is creeping somewhere
    with no edge to stop it (into the cantilever, on a sphere it partly
    hides), so it stops there and keeps the last converging fit.
    """
    cx, cy, r = (float(v) for v in circle)
    fit = SphereEdgeRefiner.refine(gray, (cx, cy), r)
    if not fit.get("ok"):
        return fit
    last = (float(np.hypot(fit["center"][0] - cx, fit["center"][1] - cy))
            + abs(float(fit["radius"]) - r))
    for _ in range(8):
        if last < 0.05:
            break
        cx, cy = fit["center"]
        r = float(fit["radius"])
        step = SphereEdgeRefiner.refine(gray, (cx, cy), r)
        if not step.get("ok"):
            break
        size = (float(np.hypot(step["center"][0] - cx,
                               step["center"][1] - cy))
                + abs(float(step["radius"]) - r))
        if size >= 0.9 * last:
            break
        fit, last = step, size
    return fit


def _hough_near(gray, click, half):
    """The best-voted circle around the click that contains it, or None."""
    g = np.asarray(gray)
    cx, cy = int(round(click[0])), int(round(click[1]))
    x0, y0 = max(0, cx - half), max(0, cy - half)
    roi = g[y0:cy + half, x0:cx + half]
    if roi.size == 0 or min(roi.shape[:2]) < 12:
        return None
    blur = cv2.GaussianBlur(roi, (0, 0), 1.5)
    for votes in (14, 10, 7):
        found = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=2,
                                 param1=60, param2=votes, minRadius=4,
                                 maxRadius=max(8, half - 4))
        if found is None:
            continue
        for x, y, r in found[0]:
            x, y = float(x) + x0, float(y) + y0
            if math.hypot(x - click[0], y - click[1]) < 0.8 * float(r):
                return (x, y, float(r))
    return None


def sphere_from_click(gray, click, r_guess=None):
    """
    The probe sphere from one click anywhere on it.

    First a Hough circle search in a window round the click (the circle
    with the most edge votes that contains the click), snapped to the
    outline when the snap agrees with it. That holds for a sphere on the
    end of a rod too, where the top of the circle has no edge. Failing
    that, a few starting points round the click are each snapped
    (snap_sphere) and the fit whose edge most rays find wins. Returns
    snap_sphere's dict (ok, center, radius, confidence) or {"ok": False}.
    """
    g0 = np.asarray(gray)
    cx0, cy0 = float(click[0]), float(click[1])
    half = int(max(90, 0.2 * max(g0.shape[:2])))
    if r_guess:
        half = int(max(half, 2.5 * float(r_guess)))
    circle = _hough_near(g0, (cx0, cy0), half)
    if circle is not None:
        fit = snap_sphere(gray, circle)
        if (fit.get("ok") and math.hypot(fit["center"][0] - circle[0],
                                         fit["center"][1] - circle[1])
                < 0.3 * circle[2]
                and abs(fit["radius"] - circle[2]) < 0.2 * circle[2]):
            return fit
        return {"ok": True, "center": (circle[0], circle[1]),
                "radius": circle[2],
                "confidence": float(fit.get("confidence", 0.5) or 0.5)}
    g = SphereEdgeRefiner._suppress_specular(g0)
    r0 = r_guess or SphereEdgeRefiner._radius_from_center(g, cx0, cy0) or 15.0
    step = max(3.0, 0.45 * float(r0))
    starts = [(cx0, cy0)] + [
        (cx0 + step * math.cos(a), cy0 + step * math.sin(a))
        for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    best = None
    for sx, sy in starts:
        r = SphereEdgeRefiner._radius_from_center(g, sx, sy) or r0
        fit = snap_sphere(gray, (sx, sy, r))
        if not fit.get("ok"):
            continue
        key = (round(float(fit["confidence"]), 2),
               -math.hypot(fit["center"][0] - cx0, fit["center"][1] - cy0))
        if best is None or key > best[0]:
            best = (key, fit)
    return best[1] if best else {"ok": False}

def available():
    """True when OpenCV is importable."""
    return cv2 is not None
